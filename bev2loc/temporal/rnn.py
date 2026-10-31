"""Recurrent sequential baseline on top of the learned observation features.

The observation model (encoders + correlation) is frozen and shared with the
per-frame learned matcher / histogram filter.  This module only adds temporal
integration: a ConvGRU operates on a coarse likelihood volume plus a context
vector pooled from the learned observation features, and predicts a refined
dense likelihood for every frame.  Decoding is the same argmax over
``(yaw, y, x)`` used by every other method.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..matching.grid import PoseGrid


class ConvGRUCell(nn.Module):
    def __init__(self, input_ch: int, hidden_ch: int, kernel_size: int = 3):
        super().__init__()
        pad = kernel_size // 2
        self.hidden_ch = hidden_ch
        self.conv_zr = nn.Conv2d(
            input_ch + hidden_ch, 2 * hidden_ch, kernel_size, padding=pad
        )
        self.conv_h = nn.Conv2d(
            input_ch + hidden_ch, hidden_ch, kernel_size, padding=pad
        )

    def forward(self, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        zr = self.conv_zr(torch.cat([x, h], dim=1))
        z, r = torch.chunk(zr, 2, dim=1)
        z = torch.sigmoid(z)
        r = torch.sigmoid(r)
        hh = torch.tanh(self.conv_h(torch.cat([x, r * h], dim=1)))
        return (1.0 - z) * h + z * hh


class ConvGRULocalizer(nn.Module):
    """Sequential localizer over coarse likelihood volumes."""

    def __init__(
        self,
        grid: PoseGrid,
        *,
        feature_dim: int = 32,
        hidden_ch: int = 32,
        context_ch: int = 4,
        latent_yaw: int = 8,
        latent_xy: int = 24,
        num_layers: int = 1,
    ):
        super().__init__()
        self.grid = grid
        self.latent_shape = (latent_yaw, latent_xy, latent_xy)
        self.context_ch = int(context_ch)
        self.feature_proj = (
            nn.Linear(feature_dim, context_ch) if context_ch > 0 else None
        )
        in_ch = (1 + self.context_ch) * int(latent_yaw)
        self.cells = nn.ModuleList(
            [
                ConvGRUCell(in_ch if i == 0 else hidden_ch, hidden_ch)
                for i in range(num_layers)
            ]
        )
        self.head = nn.Sequential(
            nn.Conv2d(hidden_ch, hidden_ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_ch, 1, 1),
        )
        self.yaw_head = nn.Sequential(
            nn.Linear(hidden_ch, hidden_ch),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_ch, latent_yaw),
        )
        self.residual_scale = nn.Parameter(torch.tensor(1.0))

    # ------------------------------------------------------------- helpers
    def _downsample(self, logits: torch.Tensor) -> torch.Tensor:
        # logits: (B, Y, ny, nx) -> (B, 1, ly, ly, lx)
        x = logits.unsqueeze(1)
        # adaptive_avg_pool3d handles (B,C,D,H,W)
        return F.adaptive_avg_pool3d(x, self.latent_shape)

    def _context(self, features: torch.Tensor | None, b: int, device) -> torch.Tensor:
        ly, ly2, lx = self.latent_shape
        if self.feature_proj is None or features is None:
            return torch.zeros((b, 0, ly, ly2, lx), device=device)
        pooled = features.mean(dim=(-2, -1))  # B,C
        ctx = self.feature_proj(pooled)  # B,context_ch
        return ctx[:, :, None, None, None].expand(-1, -1, ly, ly2, lx)

    def _latent_input(
        self, logits: torch.Tensor, features: torch.Tensor | None
    ) -> torch.Tensor:
        b = logits.shape[0]
        x = self._downsample(logits)
        ctx = self._context(features, b, logits.device)
        x = torch.cat([x, ctx], dim=1)
        # Flatten the yaw dimension into channels so ConvGRU sees a 2D field.
        return x.reshape(b, x.shape[1] * x.shape[2], x.shape[3], x.shape[4])

    def _upsample(self, x: torch.Tensor, b: int) -> torch.Tensor:
        # Spatial residual: (B,1,latent_xy,latent_xy) -> (B,1,ny,nx)
        spatial = self.head(x)
        spatial = F.interpolate(
            spatial, size=(self.grid.ny, self.grid.nx), mode="bilinear", align_corners=False
        )
        # Heading residual: (B,latent_yaw) -> (B,Y,1,1)
        yaw = self.yaw_head(x.mean(dim=(-2, -1)))  # B, latent_yaw
        yaw = F.interpolate(
            yaw[:, None], size=self.grid.n_yaw, mode="linear", align_corners=False
        )[:, 0]  # B, Y
        return spatial + yaw[:, :, None, None]

    def init_hidden(self, batch_size: int, device) -> list[torch.Tensor]:
        ly, ly2, lx = self.latent_shape
        return [
            torch.zeros(
                (batch_size, cell.hidden_ch, ly2, lx), device=device
            )
            for cell in self.cells
        ]

    # ------------------------------------------------------------- forward
    def step(
        self,
        logits: torch.Tensor,
        features: torch.Tensor | None,
        hidden: list[torch.Tensor],
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Advance one frame.  ``logits`` is ``(B, Y, ny, nx)``."""
        x = self._latent_input(logits, features)
        new_hidden = []
        for cell, h in zip(self.cells, hidden):
            h = cell(x, h)
            x = h
            new_hidden.append(h)
        residual = self._upsample(new_hidden[-1], logits.shape[0])
        refined = logits + self.residual_scale * residual
        return refined, new_hidden

    def forward_sequence(
        self,
        logits_seq: torch.Tensor,
        features_seq: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """``logits_seq``: ``(B,T,Y,ny,nx)`` -> refined ``(B,T,Y,ny,nx)``."""
        b, t = logits_seq.shape[:2]
        hidden = self.init_hidden(b, logits_seq.device)
        outs = []
        for i in range(t):
            feats = None if features_seq is None else features_seq[:, i]
            out, hidden = self.step(logits_seq[:, i], feats, hidden)
            outs.append(out)
        return torch.stack(outs, dim=1)


def pose_nll(
    logits: torch.Tensor,
    grid: PoseGrid,
    xy_px: torch.Tensor,
    yaw: torch.Tensor,
) -> torch.Tensor:
    """NLL of ground-truth pose bins under a ``(B,Y,ny,nx)`` volume."""
    b = logits.shape[0]
    logp = torch.log_softmax(logits.reshape(b, -1), dim=-1)
    xs = xy_px[:, 0].detach().cpu().numpy()
    ys = xy_px[:, 1].detach().cpu().numpy()
    yaws = yaw.detach().cpu().numpy().reshape(-1)
    idx = np.empty(b, dtype=np.int64)
    for i in range(b):
        ix = int(np.clip(round(xs[i] / grid.resolution - 0.5), 0, grid.nx - 1))
        iy = int(np.clip(round(ys[i] / grid.resolution - 0.5), 0, grid.ny - 1))
        iyaw = int(round((yaws[i] + np.pi) / grid.dyaw)) % grid.n_yaw
        idx[i] = (iyaw * grid.ny + iy) * grid.nx + ix
    idx_t = torch.from_numpy(idx).to(logits.device)
    return -logp.gather(1, idx_t[:, None]).squeeze(1).mean()