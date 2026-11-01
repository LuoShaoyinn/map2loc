"""Learned BEV-to-map matching (OrienterNet-style dense correlation).

Two small CNN encoders embed the local BEV observation and the global map.
The local feature map is rotated into every heading bin and correlated with
the global map feature map by a single batched convolution, which yields a
dense score volume over ``(x, y, yaw)``.

The observation encoder also predicts a *reliability* map.  Features are
weighted by that reliability before correlation, and the reliability head is
supervised with the corruption validity mask produced by the observation
generator (scratches, occlusion, out-of-map border, salt-and-pepper noise).
This is the mechanism that lets the observation model learn to ignore
unreliable regions.  When comparing temporal filters the encoder is frozen.
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
    """Everything needed to rebuild a trained observation/map pair."""

    fov_px: float = 320.0
    obs_width: int = 256
    obs_height: int = 256
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
    """Dense correlation matcher with a learned reliability mask."""

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
        self._grid_cache: dict[tuple[int, int], torch.Tensor] = {}
        self._inside_cache: dict[tuple[int, int], torch.Tensor] = {}

        w = int(round(cfg.fov_px / grid.resolution))
        h = int(
            round(cfg.fov_px * cfg.obs_height / cfg.obs_width / grid.resolution)
        )
        self.template_w = max(3, w if w % 2 == 1 else w + 1)
        self.template_h = max(3, h if h % 2 == 1 else h + 1)

    @property
    def temperature(self) -> torch.Tensor:
        return F.softplus(self.log_temperature) + 0.02

    # ------------------------------------------------------------ templates
    def _make_grids(
        self, feat_h: int, feat_w: int, device, dtype
    ) -> tuple[torch.Tensor, torch.Tensor]:
        key = (int(feat_h), int(feat_w))
        if key in self._grid_cache and self._grid_cache[key].device == device:
            return self._grid_cache[key], self._inside_cache[key]

        res = float(self.grid.resolution)
        fov = float(self.cfg.fov_px)
        ow, oh = float(self.cfg.obs_width), float(self.cfg.obs_height)
        s = ow / fov  # observation px per map px
        c_obs = np.array([(ow - 1) / 2.0, (oh - 1) / 2.0])
        c_t = np.array(
            [(self.template_w - 1) / 2.0, (self.template_h - 1) / 2.0]
        )
        # Feature-map pixel -> observation pixel scale.
        sx = ow / float(feat_w)
        sy = oh / float(feat_h)

        grids = np.empty(
            (self.grid.n_yaw, self.template_h, self.template_w, 2),
            dtype=np.float32,
        )
        ys, xs = np.meshgrid(
            np.arange(self.template_h, dtype=np.float32),
            np.arange(self.template_w, dtype=np.float32),
            indexing="ij",
        )
        for k, yaw in enumerate(self.grid.yaws):
            c, sn = np.cos(yaw), np.sin(yaw)
            rinv = np.array([[-sn, c], [-c, -sn]], dtype=np.float32)
            q = np.stack(
                [(xs - c_t[0]) * res, (ys - c_t[1]) * res], axis=-1
            )  # (Ht,Wt,2)
            uv = c_obs.astype(np.float32) + s * (q @ rinv.T)
            # Observation pixel -> feature pixel -> normalised sample coords.
            fx = (uv[..., 0] + 0.5) / sx - 0.5
            fy = (uv[..., 1] + 0.5) / sy - 0.5
            gx = 2.0 * fx / max(feat_w - 1, 1) - 1.0
            gy = 2.0 * fy / max(feat_h - 1, 1) - 1.0
            grids[k, ..., 0] = gx
            grids[k, ..., 1] = gy

        grid_t = torch.from_numpy(grids).to(device=device, dtype=dtype)
        inside = F.grid_sample(
            torch.ones(
                (self.grid.n_yaw, 1, feat_h, feat_w), device=device, dtype=dtype
            ),
            grid_t,
            align_corners=True,
            padding_mode="zeros",
        )
        self._grid_cache[key] = grid_t
        self._inside_cache[key] = inside
        return grid_t, inside

    def _templates(
        self,
        feats: torch.Tensor,
        reliability: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        b, c, hf, wf = feats.shape
        y = self.grid.n_yaw
        grid_t, inside = self._make_grids(hf, wf, feats.device, feats.dtype)
        g = grid_t[None].expand(b, -1, -1, -1, -1).reshape(
            b * y, self.template_h, self.template_w, 2
        )
        f = feats[:, None].expand(b, y, c, hf, wf).reshape(b * y, c, hf, wf)
        sampled = F.grid_sample(
            f, g, align_corners=True, padding_mode="zeros"
        )
        r = (
            reliability[:, None]
            .expand(b, y, 1, hf, wf)
            .reshape(b * y, 1, hf, wf)
        )
        conf = F.grid_sample(
            r, g, align_corners=True, padding_mode="zeros"
        )
        i = inside[None].expand(b, -1, -1, -1, -1).reshape(
            b * y, 1, self.template_h, self.template_w
        )
        weight = (conf * i).clamp(0.0, 1.0)
        feat = F.normalize(sampled, dim=1) * weight
        return feat, weight

    # ------------------------------------------------------------- matching
    def encode_map(self, map_image: torch.Tensor) -> torch.Tensor:
        """``map_image`` is ``(1,3,H,W)`` at grid resolution."""
        return F.normalize(self.map_encoder(map_image), dim=1)

    def forward(
        self,
        image: torch.Tensor,
        map_feats: torch.Tensor,
        *,
        valid_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Match a batch of observations against precomputed map features."""
        feats, rel_logits = self.obs_encoder(image)
        templates, weights = self._templates(feats, torch.sigmoid(rel_logits))
        th, tw = templates.shape[-2:]
        pad = (th // 2, tw // 2)
        b = image.shape[0]
        y = self.grid.n_yaw

        ones_map = torch.ones(
            (1, 1, map_feats.shape[-2], map_feats.shape[-1]),
            device=map_feats.device,
            dtype=map_feats.dtype,
        )
        scores = F.conv2d(map_feats, templates, padding=pad)
        denom = F.conv2d(ones_map, weights, padding=pad).clamp_min(1e-4)
        scores = scores / denom
        scores = scores.reshape(b, y, self.grid.ny, self.grid.nx)
        logits = scores / self.temperature

        out: dict[str, torch.Tensor] = {
            "logits": logits,
            "reliability_logits": rel_logits,
            "features": feats,
        }
        if valid_mask is not None:
            target = F.adaptive_avg_pool2d(
                valid_mask[:, None].float(), rel_logits.shape[-2:]
            )
            # Invalid pixels are rarer; emphasise them so the head does not
            # collapse to "everything is valid".
            weight = 1.0 + 4.0 * (1.0 - target)
            bce = F.binary_cross_entropy_with_logits(
                rel_logits, target, reduction="none"
            )
            out["mask_loss"] = (bce * weight).mean()
            out["mask_target"] = target
        return out

    @torch.no_grad()
    def match(
        self,
        image: torch.Tensor,
        map_feats: torch.Tensor,
        *,
        valid_mask: torch.Tensor | None = None,
    ) -> LikelihoodVolume:
        out = self.forward(image, map_feats, valid_mask=valid_mask)
        return LikelihoodVolume(logits=out["logits"][0], grid=self.grid)

    # -------------------------------------------------------------- targets
    def nll(
        self,
        logits: torch.Tensor,
        xy_px: torch.Tensor,
        yaw: torch.Tensor,
    ) -> torch.Tensor:
        """Negative log-likelihood of the observed pose under the volume."""
        b = logits.shape[0]
        flat = logits.reshape(b, -1)
        logp = torch.log_softmax(flat, dim=-1)
        idx = torch.empty(b, dtype=torch.long, device=logits.device)
        xs = xy_px[:, 0].detach().cpu().numpy()
        ys = xy_px[:, 1].detach().cpu().numpy()
        yaw_np = yaw.detach().cpu().numpy().reshape(-1)
        g = self.grid
        for i in range(b):
            ix = int(np.clip(round(xs[i] / g.resolution - 0.5), 0, g.nx - 1))
            iy = int(np.clip(round(ys[i] / g.resolution - 0.5), 0, g.ny - 1))
            iyaw = int(round((yaw_np[i] + np.pi) / g.dyaw)) % g.n_yaw
            idx[i] = (iyaw * g.ny + iy) * g.nx + ix
        return -logp.gather(1, idx[:, None]).squeeze(1).mean()