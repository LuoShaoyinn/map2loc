"""F3Loc-style SE(2) histogram filter for BEV likelihood volumes.

This keeps the useful part of F3Loc: the transition is a grouped 2D
convolution (one group per heading, the Gaussian translation kernel expressed
in that heading's body frame) followed by a circular 1D convolution over
heading.  The observation model is *not* F3Loc's perspective-depth front end;
it is any per-frame likelihood volume over the map grid, e.g. classical NCC or
the learned OrienterNet-style matcher.

The filter lives in log space so that fusing many frames cannot underflow.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from ..matching.grid import PoseGrid
from ..types import Pose


@dataclass
class FilterConfig:
    sigma_xy_px: float = 3.0
    sigma_yaw_rad: float = math.radians(4.0)
    kernel_radius_px: float = 24.0
    keep_free_only: bool = False


class SE2HistogramFilter:
    """Discrete Bayes filter on ``(x, y, yaw)`` with ego-motion transition."""

    def __init__(
        self,
        grid: PoseGrid,
        cfg: FilterConfig | None = None,
        *,
        free_mask: torch.Tensor | None = None,
        device: str | torch.device = "cpu",
    ):
        self.grid = grid
        self.cfg = cfg or FilterConfig()
        self.device = torch.device(device)
        self.n_phi = grid.n_yaw
        self.ny, self.nx = grid.ny, grid.nx
        if free_mask is None:
            self.free = torch.ones(
                (self.ny, self.nx), dtype=torch.bool, device=self.device
            )
        else:
            self.free = free_mask.to(self.device).bool()
        self.log_volume = torch.empty(
            (self.n_phi, self.ny, self.nx),
            dtype=torch.float32,
            device=self.device,
        )
        self.reset()

    # --------------------------------------------------------------- state
    @property
    def volume(self) -> torch.Tensor:
        """Posterior probabilities, shape ``(n_yaw, ny, nx)``."""
        return self.log_volume.exp()

    def reset(self, prior: torch.Tensor | Pose | None = None) -> None:
        logv = torch.full(
            (self.n_phi, self.ny, self.nx),
            -math.log(self.n_phi * self.ny * self.nx),
            dtype=torch.float32,
            device=self.device,
        )
        if isinstance(prior, Pose):
            logv = self._pose_prior(prior)
        elif prior is not None:
            p = prior.to(self.device).float()
            if p.shape != logv.shape:
                raise ValueError(
                    f"prior shape {tuple(p.shape)} != {tuple(logv.shape)}"
                )
            # Accept either probabilities or unnormalised logits.
            if p.min() < 0.0:
                logv = torch.log_softmax(p.flatten(), dim=0).reshape(p.shape)
            else:
                logv = torch.log(p.flatten().clamp_min(1e-12)).reshape(p.shape)
                logv = torch.log_softmax(logv.flatten(), dim=0).reshape(p.shape)
        if self.cfg.keep_free_only:
            logv = logv.masked_fill(~self.free[None], float("-inf"))
            logv = torch.log_softmax(logv.flatten(), dim=0).reshape(logv.shape)
        self.log_volume = logv

    def _pose_prior(self, pose: Pose, *, sigma_xy: float | None = None, sigma_yaw: float | None = None) -> torch.Tensor:
        sigma_xy = sigma_xy or self.cfg.sigma_xy_px
        sigma_yaw = sigma_yaw or self.cfg.sigma_yaw_rad
        coords = self.grid.to(self.device)
        x = coords["x"]
        y = coords["y"]
        yaw = coords["yaw"]
        d2 = (x - pose.x) ** 2 + (y - pose.y) ** 2
        # Circular yaw difference.
        dyaw = torch.atan2(torch.sin(yaw - pose.yaw), torch.cos(yaw - pose.yaw))
        logp = -0.5 * d2 / max(sigma_xy, 1e-6) ** 2 - 0.5 * dyaw**2 / max(
            sigma_yaw, 1e-6
        ) ** 2
        if self.cfg.keep_free_only:
            logp = logp.masked_fill(~self.free[None], float("-inf"))
        return torch.log_softmax(logp.flatten(), dim=0).reshape(logp.shape)

    def set_prior_pose(
        self,
        pose: Pose,
        *,
        sigma_xy: float | None = None,
        sigma_yaw: float | None = None,
    ) -> None:
        self.log_volume = self._pose_prior(pose, sigma_xy=sigma_xy, sigma_yaw=sigma_yaw)

    # ---------------------------------------------------------- observation
    def update(self, log_likelihood: torch.Tensor) -> None:
        """Fuse a per-frame log-likelihood ``(Y, ny, nx)`` (or probabilities)."""
        ll = log_likelihood.to(self.device).float()
        if ll.shape != self.log_volume.shape:
            raise ValueError(
                f"likelihood shape {tuple(ll.shape)} != {tuple(self.log_volume.shape)}"
            )
        if ll.min() >= 0.0:
            ll = torch.log(ll.clamp_min(1e-12))
        logv = self.log_volume + ll
        if self.cfg.keep_free_only:
            logv = logv.masked_fill(~self.free[None], float("-inf"))
        self.log_volume = torch.log_softmax(logv.flatten(), dim=0).reshape(logv.shape)

    # ----------------------------------------------------------- transition
    def _translation_kernels(self, ego_motion: torch.Tensor) -> torch.Tensor:
        res = float(self.grid.resolution)
        t_fwd = float(ego_motion[0])
        t_right = float(ego_motion[1])
        radius_px = max(
            self.cfg.kernel_radius_px,
            math.hypot(t_fwd, t_right) + 3.0 * self.cfg.sigma_xy_px,
        )
        r = max(1, int(math.ceil(radius_px / res)))
        axis = torch.arange(-r, r + 1, device=self.device, dtype=torch.float32) * res
        gx, gy = torch.meshgrid(axis, axis, indexing="xy")  # world offsets
        phi = torch.as_tensor(self.grid.yaws, device=self.device, dtype=torch.float32)
        c, s = torch.cos(phi), torch.sin(phi)
        # Body-frame coordinates of each world offset.
        bx = c[:, None, None] * gx[None] + s[:, None, None] * gy[None]
        by = -s[:, None, None] * gx[None] + c[:, None, None] * gy[None]
        sigma2 = max(self.cfg.sigma_xy_px, 1e-6) ** 2
        k = torch.exp(-0.5 * ((bx - t_fwd) ** 2 + (by - t_right) ** 2) / sigma2)
        k = k / k.sum(dim=(1, 2), keepdim=True).clamp_min(1e-12)
        # conv2d is cross-correlation: flip so the posterior moves *by* +t.
        k = torch.flip(k, dims=(-2, -1))
        return k[:, None]

    def _rotation_kernel(self, dphi: float) -> torch.Tensor:
        sigma = max(self.cfg.sigma_yaw_rad, 1e-6)
        dstep = self.grid.dyaw
        kk = max(1, int(math.ceil((3.0 * sigma + abs(dphi)) / dstep)))
        off = torch.arange(-kk, kk + 1, device=self.device, dtype=torch.float32)
        k = torch.exp(-0.5 * ((off * dstep - dphi) / sigma) ** 2)
        k = k / k.sum().clamp_min(1e-12)
        # conv1d is cross-correlation; flip so heading moves by +dphi.
        k = torch.flip(k, dims=(-1,))
        return k.view(1, 1, -1)

    def predict(self, ego_motion) -> None:
        """Apply ego-motion ``(forward, right, dyaw)`` in the previous frame."""
        if isinstance(ego_motion, np.ndarray):
            t = torch.from_numpy(np.asarray(ego_motion, dtype=np.float32))
        else:
            t = torch.as_tensor(ego_motion, dtype=torch.float32)
        t = t.to(self.device).reshape(-1)
        if t.numel() < 3:
            t = torch.cat([t, torch.zeros(3 - t.numel(), device=self.device)])

        prob = self.log_volume.exp()
        kernels = self._translation_kernels(t)
        r = kernels.shape[-1] // 2
        vol = F.conv2d(prob[None], kernels, padding=r, groups=self.n_phi)[0]

        rk = self._rotation_kernel(float(t[2]))
        kk = rk.shape[-1] // 2
        v = vol.permute(1, 2, 0).reshape(self.ny * self.nx, 1, self.n_phi)
        v = F.pad(v, (kk, kk), mode="circular")
        v = F.conv1d(v, rk)
        vol = v.reshape(self.ny, self.nx, self.n_phi).permute(2, 0, 1)
        vol = vol.clamp_min(0.0)
        if self.cfg.keep_free_only:
            vol = vol * self.free[None]
        s = vol.sum()
        if float(s) <= 0 or not torch.isfinite(s):
            self.reset()
        else:
            logv = torch.log(vol.clamp_min(1e-30))
            self.log_volume = torch.log_softmax(logv.flatten(), dim=0).reshape(logv.shape)

    # ---------------------------------------------------------------- utils
    def posterior_xy(self) -> torch.Tensor:
        p = self.volume.sum(0)
        return p / p.sum().clamp_min(1e-12)

    def posterior_yaw(self) -> torch.Tensor:
        p = self.volume.sum((1, 2))
        return p / p.sum().clamp_min(1e-12)

    def entropy(self) -> float:
        p = self.volume.flatten()
        p = p[p > 0]
        return float(-(p * torch.log(p)).sum())

    def best(self) -> Pose:
        idx = int(self.log_volume.flatten().argmax())
        nx, ny = self.grid.nx, self.grid.ny
        ix = idx % nx
        iy = (idx // nx) % ny
        iyaw = idx // (nx * ny)
        return self.grid.pose_at(iyaw, iy, ix)

    def best_log_prob(self) -> float:
        return float(self.log_volume.max())

    def map_image(self, base: np.ndarray | None = None) -> np.ndarray:
        import cv2

        p = self.posterior_xy().detach().cpu().numpy().astype(np.float32)
        p = cv2.resize(p, (self.grid.map_width, self.grid.map_height), interpolation=cv2.INTER_LINEAR)
        p = p / max(float(p.max()), 1e-12)
        if base is None:
            return (np.stack([p, 0.2 * p, 1.0 - p], axis=-1) * 255).astype(np.uint8)
        heat = (np.stack([p, 0.2 * p, 1.0 - p], axis=-1) * 255).astype(np.uint8)
        return cv2.addWeighted(base.astype(np.uint8), 0.6, heat, 0.4, 0)