"""Classical masked NCC on rectified BEV templates.

Each perspective observation is rectified into a map-aligned BEV patch using
the measured attitude (see :mod:`map2loc.bev`).  Matching is then a
translation search, with a small residual heading search for the attitude
noise and several focal-scale hypotheses for the unmeasured zoom.  The result
is a dense log-likelihood over map position.

Masked NCC uses only valid template pixels, so the 50 % random invalid pixels,
scratches and out-of-view regions do not create false peaks.
"""

from __future__ import annotations

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from .grid import LikelihoodVolume, PoseGrid, map_raster
from .learned import merge_residual_yaw


def _as_gray(image) -> torch.Tensor:
    if isinstance(image, torch.Tensor):
        x = image.detach().cpu().float()
    else:
        x = torch.from_numpy(np.asarray(image)).float()
    if x.ndim == 2:
        return x
    if x.shape[0] in (1, 3):
        x = x.permute(1, 2, 0)
    if x.shape[-1] == 1:
        return x[..., 0]
    if x.max() > 1.5:
        x = x / 255.0
    if x.min() < -1e-3:
        x = x * 0.5 + 0.5
    return x[..., 0] * 0.299 + x[..., 1] * 0.587 + x[..., 2] * 0.114


class ClassicalNCCMatcher:
    """Masked NCC over position with residual-yaw and scale search."""

    def __init__(
        self,
        grid: PoseGrid,
        *,
        patch_px: int = 164,
        map_px_per_px: float = 2.0,
        temperature: float = 0.10,
        device: str | torch.device = "cpu",
    ):
        self.grid = grid
        self.patch_px = int(patch_px)
        self.map_px_per_px = float(map_px_per_px)
        self.temperature = float(temperature)
        self.device = torch.device(device)
        t = int(round(self.patch_px * self.map_px_per_px / grid.resolution))
        self.template = t if t % 2 == 1 else t + 1
        self._map: dict[str, torch.Tensor] | None = None

    def set_map(self, map_) -> None:
        r = map_raster(
            map_, self.grid, grayscale=True, normalize=True, device=self.device
        )
        self._map = {"gray": r["gray"], "valid": r["valid"]}
        self.map_ = map_

    def _template(self, patch: torch.Tensor, dyaw: float) -> tuple[torch.Tensor, torch.Tensor]:
        t = self.template
        c = (t - 1) / 2.0
        ys, xs = torch.meshgrid(
            torch.arange(t, device=patch.device, dtype=patch.dtype),
            torch.arange(t, device=patch.device, dtype=patch.dtype),
            indexing="ij",
        )
        cth, sth = np.cos(dyaw), np.sin(dyaw)
        qx = (xs - c) * self.grid.resolution / self.map_px_per_px
        qy = (ys - c) * self.grid.resolution / self.map_px_per_px
        # Same convention as the learned matcher.
        rx = cth * qx - sth * qy
        ry = sth * qx + cth * qy
        ref = self.patch_px // 2
        px = ref + rx
        py = ref + ry
        gx = 2.0 * px / max(self.patch_px - 1, 1) - 1.0
        gy = 2.0 * py / max(self.patch_px - 1, 1) - 1.0
        grid = torch.stack([gx, gy], dim=-1)[None]
        return F.grid_sample(
            patch[None, None], grid, align_corners=True, padding_mode="zeros"
        )[0, 0]

    def _template_mask(self, mask: torch.Tensor, dyaw: float) -> torch.Tensor:
        t = self.template
        c = (t - 1) / 2.0
        ys, xs = torch.meshgrid(
            torch.arange(t, device=mask.device, dtype=mask.dtype),
            torch.arange(t, device=mask.device, dtype=mask.dtype),
            indexing="ij",
        )
        cth, sth = np.cos(dyaw), np.sin(dyaw)
        qx = (xs - c) * self.grid.resolution / self.map_px_per_px
        qy = (ys - c) * self.grid.resolution / self.map_px_per_px
        rx = cth * qx - sth * qy
        ry = sth * qx + cth * qy
        ref = self.patch_px // 2
        gx = 2.0 * (ref + rx) / max(self.patch_px - 1, 1) - 1.0
        gy = 2.0 * (ref + ry) / max(self.patch_px - 1, 1) - 1.0
        g = torch.stack([gx, gy], dim=-1)[None]
        out = F.grid_sample(
            mask[None, None], g, align_corners=True, padding_mode="zeros"
        )[0, 0]
        return (out > 0.5).to(mask.dtype)

    def _ncc(self, template: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        m = self._map["gray"]
        map_valid = self._map["valid"]
        th, tw = template.shape
        pad = (th // 2, tw // 2)

        def conv(inp, weight):
            return F.conv2d(inp, weight, padding=pad)

        t = template[None, None]
        v = weight[None, None]
        n = conv(map_valid, v).clamp_min(1e-6)
        m_valid = m * map_valid
        sum_p = conv(m_valid, v)
        sum_p2 = conv(m_valid * m_valid, v)
        sum_tp = conv(m_valid, t * v)
        sum_tv = (t * v).sum()
        sum_t2v = ((t * t) * v).sum()
        numerator = sum_tp - sum_tv * sum_p / n
        var_t = (sum_t2v - sum_tv * sum_tv / n).clamp_min(0.0)
        var_p = (sum_p2 - sum_p * sum_p / n).clamp_min(0.0)
        ncc = numerator / torch.sqrt(var_t * var_p + 1e-6)
        valid_count = v.sum()
        good = (
            (var_t > 1e-3 * valid_count)
            & (var_p > 1e-3 * n)
            & (n >= 0.25 * valid_count)
        )
        ncc = torch.clamp(ncc, -1.0, 1.0)
        return torch.where(good, ncc, torch.full_like(ncc, -1.0))[0, 0]

    def match(self, patches, masks, yaw=None) -> LikelihoodVolume:
        """``patches``: ``(S,P,P,C)`` or ``(S,C,P,P)``; ``masks`` ``(S,P,P)``."""
        if self._map is None:
            raise RuntimeError("call set_map() before match()")
        if isinstance(patches, torch.Tensor):
            p = patches.detach().to(self.device)
            if p.ndim == 3:
                p = p[None]
            if p.shape[1] in (1, 3) and p.shape[-1] not in (1, 3):
                p = p.permute(0, 2, 3, 1)
        else:
            p = torch.from_numpy(np.asarray(patches)).to(self.device)
            if p.ndim == 3:
                p = p[None]
            if p.shape[1] in (1, 3) and p.shape[-1] not in (1, 3):
                p = p.permute(0, 2, 3, 1)
        if isinstance(masks, torch.Tensor):
            m = masks.detach().to(self.device).float()
        else:
            m = torch.from_numpy(np.asarray(masks)).to(self.device).float()
        if m.ndim == 2:
            m = m[None]

        scores = []
        for s in range(p.shape[0]):
            gray = _as_gray(p[s]).to(self.device)
            mask = m[s]
            v_sum = mask.sum().clamp_min(1.0)
            gray = (gray - (gray * mask).sum() / v_sum) * mask
            per_yaw = []
            for dyaw in (
                -self.grid.dyaw,
                0.0,
                self.grid.dyaw,
            ):
                tpl = self._template(gray, dyaw)
                tmask = self._template_mask(mask, dyaw)
                tpl = (tpl - (tpl * tmask).sum() / tmask.sum().clamp_min(1.0)) * tmask
                per_yaw.append(self._ncc(tpl, tmask))
            scores.append(torch.stack(per_yaw, dim=0))
        scores = torch.stack(scores, dim=0).amax(dim=0)[None]  # 1,R,ny,nx
        if yaw is None:
            yaw_t = torch.zeros(1, device=self.device)
        else:
            yaw_t = torch.as_tensor(yaw, device=self.device, dtype=torch.float32)
            if yaw_t.ndim == 0:
                yaw_t = yaw_t[None]
        logits = merge_residual_yaw(
            scores / self.temperature,
            self.grid,
            yaw_t,
            self.grid.dyaw,
        )
        logits.logits = logits.logits[0]
        return logits