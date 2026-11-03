"""Learned BEV-to-map matching with measured attitude.

The observation is first rectified into a map-aligned BEV patch (see
:mod:`map2loc.bev`).  Unlike the original OrienterNet-style matcher, the
observation encoder does **not** need to search over a full 360 deg of
heading: yaw, pitch and roll are measured inputs (5 deg sigma).  The network
therefore only searches

* a small residual heading around the measured yaw, and
* a set of focal-scale hypotheses (the camera zoom is not measured).

The dense correlation still produces a likelihood over map position.  A
reliability head is supervised with the rectified validity mask so the
encoder learns to ignore invalid pixels, scratches, border padding and
out-of-view regions produced by the oblique-to-BEV warp.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .grid import LikelihoodVolume, PoseGrid


def _norm(ch: int) -> nn.Module:
    return nn.GroupNorm(num_groups=min(8, ch), num_channels=ch)


class ConvBlock(nn.Sequential):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1, dilation: int = 1):
        super().__init__(
            nn.Conv2d(
                in_ch,
                out_ch,
                kernel_size=3,
                stride=stride,
                padding=dilation,
                dilation=dilation,
                bias=False,
            ),
            _norm(out_ch),
            nn.ReLU(inplace=True),
        )


class ObservationEncoder(nn.Module):
    """Local BEV encoder with a reliability / corruption head."""

    def __init__(
        self,
        *,
        in_channels: int = 3,
        base_channels: int = 32,
        feature_dim: int = 32,
        stride: int = 4,
    ):
        super().__init__()
        if stride < 1:
            raise ValueError("stride must be >= 1")
        self.stride = int(stride)
        self.stem = ConvBlock(in_channels, base_channels, stride=1)
        blocks = []
        ch = base_channels
        remaining = self.stride
        while remaining > 1:
            step = 2 if remaining % 2 == 0 else 1
            blocks.append(ConvBlock(ch, min(base_channels * 2, 128), stride=step))
            ch = min(base_channels * 2, 128)
            remaining //= step
        self.down = nn.Sequential(*blocks) if blocks else nn.Identity()
        self.body = nn.Sequential(
            ConvBlock(ch, ch, dilation=1),
            ConvBlock(ch, ch, dilation=2),
        )
        self.feat = nn.Conv2d(ch, feature_dim, 1)
        self.reliability = nn.Conv2d(ch, 1, 1)

    def forward(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.body(self.down(self.stem(image)))
        return self.feat(x), self.reliability(x)


class MapEncoder(nn.Module):
    """Global map encoder at the matching-grid resolution."""

    def __init__(
        self,
        *,
        in_channels: int = 3,
        base_channels: int = 32,
        feature_dim: int = 32,
    ):
        super().__init__()
        self.net = nn.Sequential(
            ConvBlock(in_channels, base_channels, stride=1),
            ConvBlock(base_channels, base_channels, stride=1),
            ConvBlock(base_channels, base_channels, stride=1, dilation=2),
            ConvBlock(base_channels, feature_dim, stride=1, dilation=2),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.net(image)


@dataclass
class LearnedMatcherConfig:
    patch_px: int = 164
    map_px_per_px: float = 2.0
    feature_dim: int = 32
    base_channels: int = 32
    obs_stride: int = 4
    grid_resolution: float = 8.0
    n_yaw: int = 64
    temperature_init: float = 0.10

    def to_dict(self) -> dict:
        return dict(self.__dict__)

    @classmethod
    def from_dict(cls, d: dict) -> "LearnedMatcherConfig":
        return cls(**{k: d[k] for k in cls.__dataclass_fields__ if k in d})


class LearnedMatcher(nn.Module):
    """Dense correlation matcher over position with attitude residual search."""

    def __init__(self, cfg: LearnedMatcherConfig, grid: PoseGrid):
        super().__init__()
        self.cfg = cfg
        self.grid = grid
        self.obs_encoder = ObservationEncoder(
            base_channels=cfg.base_channels,
            feature_dim=cfg.feature_dim,
            stride=cfg.obs_stride,
        )
        self.map_encoder = MapEncoder(
            base_channels=cfg.base_channels,
            feature_dim=cfg.feature_dim,
        )
        self.log_temperature = nn.Parameter(
            torch.tensor(float(np.log(max(cfg.temperature_init, 1e-3))))
        )
        # 41 template cells for the default 164 px / 2 map-px patch at 8 px
        # resolution.  Always odd so the convolution centre is unambiguous.
        t = int(round(cfg.patch_px * cfg.map_px_per_px / cfg.grid_resolution))
        self.template = t if t % 2 == 1 else t + 1
        self._rot_cache: dict[tuple, torch.Tensor] = {}

    @property
    def temperature(self) -> torch.Tensor:
        return F.softplus(self.log_temperature) + 0.02

    @property
    def residual_yaw(self) -> float:
        return self.grid.dyaw

    # ----------------------------------------------------------- templates
    def _template_grids(
        self,
        feat_h: int,
        feat_w: int,
        device,
        dtype,
        dyaw: float,
    ) -> torch.Tensor:
        """Normalised sample coords ``(T,T,2)`` for one residual heading."""
        key = (int(feat_h), int(feat_w), round(float(dyaw), 6), str(device))
        if key in self._rot_cache:
            return self._rot_cache[key]
        t = self.template
        c = (t - 1) / 2.0
        ys, xs = torch.meshgrid(
            torch.arange(t, device=device, dtype=dtype),
            torch.arange(t, device=device, dtype=dtype),
            indexing="ij",
        )
        cth, sth = np.cos(dyaw), np.sin(dyaw)
        # One template cell per matching-grid cell, i.e. ``grid_resolution``
        # map pixels.  Convert that to patch pixels (``map_px_per_px``) and
        # then to observation-encoder feature pixels (``obs_stride``).
        qx = (xs - c) * self.cfg.grid_resolution
        qy = (ys - c) * self.cfg.grid_resolution
        # Rotate into the patch frame.
        rx = cth * qx - sth * qy
        ry = sth * qx + cth * qy
        # Patch pixel -> feature pixel index (stride conversion).
        ref = self.cfg.patch_px // 2
        fx = (ref + rx / self.cfg.map_px_per_px) / self.cfg.obs_stride
        fy = (ref + ry / self.cfg.map_px_per_px) / self.cfg.obs_stride
        gx = 2.0 * fx / max(feat_w - 1, 1) - 1.0
        gy = 2.0 * fy / max(feat_h - 1, 1) - 1.0
        out = torch.stack([gx, gy], dim=-1)
        self._rot_cache[key] = out
        return out

    def _templates(
        self,
        feats: torch.Tensor,
        reliability: torch.Tensor,
        dyaw: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        b, c, hf, wf = feats.shape
        g = self._template_grids(hf, wf, feats.device, feats.dtype, dyaw)
        g = g[None].expand(b, -1, -1, -1)
        sampled = F.grid_sample(feats, g, align_corners=True, padding_mode="zeros")
        conf = F.grid_sample(
            reliability, g, align_corners=True, padding_mode="zeros"
        )
        weight = conf.clamp(0.0, 1.0)
        return F.normalize(sampled, dim=1) * weight, weight

    # ------------------------------------------------------------- matching
    def encode_map(self, map_image: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.map_encoder(map_image), dim=1)

    def forward(
        self,
        bev: torch.Tensor,
        map_feats: torch.Tensor,
        *,
        valid_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Match ``(B,S,3,P,P)`` BEV hypotheses against map features.

        ``valid_mask`` is ``(B,S,P,P)``.  Returns per-hypothesis residual
        scores, the yaw-marginal position logits and the reliability loss.
        """
        if bev.ndim == 5:
            b, s = bev.shape[:2]
        else:
            b, s = bev.shape[0], 1
            bev = bev[:, None]
        flat = bev.reshape(b * s, *bev.shape[2:])
        feats, rel_logits = self.obs_encoder(flat)
        rel = torch.sigmoid(rel_logits)

        th, tw = self.template, self.template
        pad = (th // 2, tw // 2)
        ones_map = torch.ones(
            (1, 1, map_feats.shape[-2], map_feats.shape[-1]),
            device=map_feats.device,
            dtype=map_feats.dtype,
        )

        scores = []
        for dyaw in (-self.residual_yaw, 0.0, self.residual_yaw):
            tpl, wt = self._templates(feats, rel, dyaw)
            corr = F.conv2d(map_feats, tpl, padding=pad)
            denom = F.conv2d(ones_map, wt, padding=pad).clamp_min(1e-4)
            scores.append((corr / denom).reshape(b, s, *corr.shape[-2:]))
        # (B,S,R,ny,nx): the zoom is a nuisance variable -> max over scales.
        scores = torch.stack(scores, dim=2).amax(dim=1)  # B,R,ny,nx
        xy_logits = torch.logsumexp(scores, dim=1) / self.temperature

        out: dict[str, torch.Tensor] = {
            "scores": scores / self.temperature,
            "xy_logits": xy_logits,
            "reliability_logits": rel_logits.reshape(b, s, *rel_logits.shape[1:]),
            "features": feats.reshape(b, s, *feats.shape[1:]),
        }
        if valid_mask is not None:
            target = F.adaptive_avg_pool2d(
                valid_mask.reshape(b * s, 1, *valid_mask.shape[-2:]).float(),
                rel_logits.shape[-2:],
            )
            weight = 1.0 + 4.0 * (1.0 - target)
            bce = F.binary_cross_entropy_with_logits(
                rel_logits, target, reduction="none"
            )
            out["mask_loss"] = (bce * weight).mean()
        return out

    @torch.no_grad()
    def match(
        self,
        bev: torch.Tensor,
        map_feats: torch.Tensor,
        *,
        yaw: torch.Tensor | None = None,
        valid_mask: torch.Tensor | None = None,
    ) -> LikelihoodVolume:
        out = self.forward(bev, map_feats, valid_mask=valid_mask)
        return merge_residual_yaw(
            out["scores"], self.grid, yaw, self.residual_yaw
        )

    # -------------------------------------------------------------- targets
    def nll(
        self,
        xy_logits: torch.Tensor,
        xy_px: torch.Tensor,
    ) -> torch.Tensor:
        """Position NLL at the ground-truth (patch-centre) location."""
        b = xy_logits.shape[0]
        logp = torch.log_softmax(xy_logits.reshape(b, -1), dim=-1)
        idx = grid_indices(self.grid, xy_px)
        return -logp.gather(1, idx[:, None]).squeeze(1).mean()


def grid_indices(grid: PoseGrid, xy_px: torch.Tensor) -> torch.Tensor:
    xs = xy_px[:, 0].detach().cpu().numpy()
    ys = xy_px[:, 1].detach().cpu().numpy()
    out = np.empty(len(xs), dtype=np.int64)
    for i in range(len(xs)):
        ix = int(np.clip(round(xs[i] / grid.resolution - 0.5), 0, grid.nx - 1))
        iy = int(np.clip(round(ys[i] / grid.resolution - 0.5), 0, grid.ny - 1))
        out[i] = iy * grid.nx + ix
    return torch.from_numpy(out).to(xy_px.device)


def merge_residual_yaw(
    scores: torch.Tensor,
    grid: PoseGrid,
    yaw: torch.Tensor | None,
    residual_yaw: float,
    *,
    floor_drop: float = 12.0,
) -> LikelihoodVolume:
    """Scatter three residual-yaw scores into a full ``(Y,ny,nx)`` volume."""
    b, r, ny, nx = scores.shape
    if yaw is None:
        yaw = torch.zeros(b, device=scores.device)
    yaw = yaw.reshape(-1)
    mins = scores.amin(dim=(1, 2, 3), keepdim=True)
    vol = (mins - floor_drop).expand(b, grid.n_yaw, ny, nx).clone()
    offsets = torch.arange(r, device=scores.device, dtype=scores.dtype) - (r // 2)
    for k in range(r):
        iyaw = torch.round((yaw + offsets[k] * residual_yaw + np.pi) / grid.dyaw)
        iyaw = iyaw.long() % grid.n_yaw
        vol.scatter_(
            1,
            iyaw.reshape(b, 1, 1, 1).expand(b, 1, ny, nx),
            scores[:, k : k + 1],
        )
    return LikelihoodVolume(logits=vol, grid=grid)