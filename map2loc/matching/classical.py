"""Rotation-aware normalised cross-correlation (classical baseline).

The observation is a BEV crop of the same map that is being matched against.
For every heading bin we resample the observation into the map's axis-aligned
frame at the matching resolution, then run masked NCC over the whole map.  The
result is a dense log-likelihood over ``(yaw, y, x)``.

The validity mask is important: border padding, scratches, smudges and
occluders are *not* map content.  Masked NCC computes all moments over valid
template pixels only and therefore does not reward unexplained bright/dark
blobs.
"""

from __future__ import annotations

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from .grid import LikelihoodVolume, PoseGrid, map_raster


def _as_gray(image) -> np.ndarray:
    """Return a float32 ``(H, W)`` grayscale image from common input types."""
    if isinstance(image, torch.Tensor):
        x = image.detach().cpu().float()
        if x.ndim == 3 and x.shape[0] in (1, 3):
            x = x.permute(1, 2, 0)
        x = x.numpy()
    else:
        x = np.asarray(image)
        if x.ndim == 3 and x.shape[0] in (1, 3) and x.shape[-1] not in (1, 3):
            x = np.transpose(x, (1, 2, 0))
    if x.ndim == 2:
        return x.astype(np.float32)
    if x.shape[-1] == 1:
        return x[..., 0].astype(np.float32)
    if x.dtype == np.uint8:
        return cv2.cvtColor(x, cv2.COLOR_RGB2GRAY).astype(np.float32)
    # Float RGB, possibly in [-1, 1] from dataset normalisation.
    x = x.astype(np.float32)
    if x.min() < -1e-3:
        x = x * 0.5 + 0.5
    return x[..., 0] * 0.299 + x[..., 1] * 0.587 + x[..., 2] * 0.114


class ClassicalNCCMatcher:
    """Exhaustive rotation-aware masked NCC template matcher."""

    def __init__(
        self,
        grid: PoseGrid,
        *,
        fov_px: float,
        obs_width: int,
        obs_height: int,
        temperature: float = 0.10,
        device: str | torch.device = "cpu",
    ):
        self.grid = grid
        self.fov_px = float(fov_px)
        self.obs_width = int(obs_width)
        self.obs_height = int(obs_height)
        self.temperature = float(temperature)
        self.device = torch.device(device)
        self._map_cache: dict[bool, dict[str, torch.Tensor]] = {}
        # Template size is the map span covered by the observation, expressed
        # in grid cells.  Odd sizes keep the convolution centre aligned.
        w = int(round(self.fov_px / grid.resolution))
        h = int(round(self.fov_px * self.obs_height / self.obs_width / grid.resolution))
        self.template_w = max(3, w if w % 2 == 1 else w + 1)
        self.template_h = max(3, h if h % 2 == 1 else h + 1)

    # ------------------------------------------------------------------ map
    def _map_tensors(self, grayscale: bool = True) -> dict[str, torch.Tensor]:
        key = bool(grayscale)
        if key not in self._map_cache:
            # Cached by the caller via ``set_map``; see below.
            raise RuntimeError("call set_map() before matching")
        return self._map_cache[key]

    def set_map(self, map_) -> None:
        """Precompute the map raster on the matching grid."""
        r = map_raster(map_, self.grid, grayscale=True, normalize=True, device=self.device)
        gray = r["gray"]  # 1,1,ny,nx
        valid = r["valid"]
        self._map_cache[True] = {"gray": gray, "valid": valid}
        self._map = map_

    # ------------------------------------------------------------ templates
    def _yaw_matrices(self, yaws: np.ndarray) -> np.ndarray:
        """Affine (Y,2,3) mapping template pixels -> observation pixels."""
        s = self.obs_width / self.fov_px  # observation px per map px
        c_obs = np.array(
            [(self.obs_width - 1) / 2.0, (self.obs_height - 1) / 2.0],
            dtype=np.float64,
        )
        c_t = np.array(
            [(self.template_w - 1) / 2.0, (self.template_h - 1) / 2.0],
            dtype=np.float64,
        )
        mats = np.empty((len(yaws), 2, 3), dtype=np.float64)
        for k, yaw in enumerate(yaws):
            c, sn = np.cos(yaw), np.sin(yaw)
            # R^-1 for R = [[-sin, -cos], [cos, -sin]]
            rinv = np.array([[-sn, c], [-c, -sn]], dtype=np.float64)
            a = s * self.grid.resolution * rinv
            b = c_obs - a @ c_t
            mats[k, :, :2] = a
            mats[k, :, 2] = b
        return mats

    def build_templates(
        self, image, valid_mask=None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Resample an observation into map-aligned templates for every yaw."""
        gray = _as_gray(image).astype(np.float32)
        if valid_mask is None:
            mask = np.ones(gray.shape, dtype=np.float32)
        elif isinstance(valid_mask, torch.Tensor):
            mask = valid_mask.detach().cpu().float().numpy()
        else:
            mask = np.asarray(valid_mask, dtype=np.float32)
        mask = (mask > 0.5).astype(np.float32)

        mats = self._yaw_matrices(self.grid.yaws)
        templates = np.empty(
            (len(mats), 1, self.template_h, self.template_w), dtype=np.float32
        )
        valid = np.empty_like(templates)
        for k, a in enumerate(mats):
            templates[k, 0] = cv2.warpAffine(
                gray,
                a,
                (self.template_w, self.template_h),
                flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            )
            valid[k, 0] = cv2.warpAffine(
                mask,
                a,
                (self.template_w, self.template_h),
                flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            )
        t = torch.from_numpy(templates).to(self.device)
        v = torch.from_numpy(valid).to(self.device)
        # Zero-mean templates; masked moments handle the rest.
        v_sum = v.sum(dim=(2, 3), keepdim=True).clamp_min(1e-6)
        t = (t - (t * v).sum(dim=(2, 3), keepdim=True) / v_sum) * v
        return t, v

    # ------------------------------------------------------------ matching
    def match(self, image, valid_mask=None) -> LikelihoodVolume:
        if not self._map_cache:
            raise RuntimeError("call set_map() before match()")
        maps = self._map_cache[True]
        m = maps["gray"]
        map_valid = maps["valid"]

        t, v = self.build_templates(image, valid_mask)  # Y,1,th,tw
        y, _, th, tw = t.shape
        pad = (th // 2, tw // 2)

        def conv(inp, weight):
            return F.conv2d(inp, weight, padding=pad)

        n = conv(map_valid, v).clamp_min(1e-6)
        m_valid = m * map_valid
        sum_p = conv(m_valid, v)
        sum_p2 = conv(m_valid * m_valid, v)
        sum_tp = conv(m_valid, t * v)
        sum_tv = (t * v).sum(dim=(1, 2, 3), keepdim=True)  # Y,1,1,1
        sum_t2v = ((t * t) * v).sum(dim=(1, 2, 3), keepdim=True)

        numerator = sum_tp - sum_tv * sum_p / n
        var_t = (sum_t2v - sum_tv * sum_tv / n).clamp_min(0.0)
        var_p = (sum_p2 - sum_p * sum_p / n).clamp_min(0.0)
        denom = torch.sqrt(var_t * var_p + 1e-6)
        ncc = numerator / denom

        # Penalise templates that are mostly outside the map or degenerate.
        valid_count = v.sum(dim=(1, 2, 3), keepdim=True)
        enough = n >= 0.25 * valid_count
        # Near-constant map patches (featureless grass) carry no information
        # and their variance suffers catastrophic cancellation; reject them.
        min_var_p = 1e-3 * n
        min_var_t = 1e-3 * valid_count
        good = (
            (var_t > min_var_t)
            & (var_p > min_var_p)
            & enough
        )
        ncc = torch.clamp(ncc, -1.0, 1.0)
        ncc = torch.where(good, ncc, torch.full_like(ncc, -1.0))
        ncc = ncc[0]  # Y,ny,nx
        logits = ncc / max(self.temperature, 1e-6)
        return LikelihoodVolume(logits=logits, grid=self.grid)