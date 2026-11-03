"""Position-only histogram Bayes filter.

In the updated task the camera attitude is a measured input (yaw / pitch /
roll, 5 deg sigma), so there is no reason to integrate heading into the
temporal state.  The filter therefore tracks the camera ground position
``(x, y)`` only:

* **predict** translates the discrete posterior by the available ego-motion,
  expressed in the measured heading frame, and blurs it with a small Gaussian;
* **update** adds the (yaw-marginalised) image likelihood in log space.

Heading is taken from the per-frame attitude measurement (optionally refined
by the residual-heading peak of the image likelihood).  This keeps the
temporal model comparable with the per-frame matchers while removing the
180 deg map-symmetry failure mode of heading search.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from ..matching.grid import PoseGrid
from ..types import Pose


@dataclass
class FilterConfig:
    # Process noise on the translation, in map pixels.
    sigma_xy_px: float = 3.0
    # Extra inflation applied after each update (prevents overconfidence).
    floor_fraction: float = 1e-4
    # Fraction of the posterior reset to the current frame's global
    # likelihood each step.  This lets the filter escape a wrong lock (e.g.
    # after distractor frames) while still using all temporal evidence.
    global_mix: float = 0.0


class XYHistogramFilter:
    """Discrete Bayes filter on a ``(ny, nx)`` position lattice."""

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
        self.free_mask = free_mask
        lp = torch.zeros((grid.ny, grid.nx), device=self.device)
        if free_mask is not None:
            lp = lp + torch.log(free_mask.clamp_min(1e-6).to(self.device))
        self.log_prob = torch.log_softmax(lp.flatten(), dim=0).reshape_as(lp)
        self._yaw = 0.0
        self._kernel: torch.Tensor | None = None
        self._kernel_sigma: float | None = None

    # ---------------------------------------------------------------- prior
    def set_prior_pose(
        self,
        pose: Pose,
        sigma_xy: float = 3.0,
        sigma_yaw: float = 0.1,
    ) -> None:
        coords = self.grid.to(self.device)
        d2 = (coords["x"] - pose.x) ** 2 + (coords["y"] - pose.y) ** 2
        lp = -0.5 * d2 / max(float(sigma_xy), 1e-3) ** 2
        if self.free_mask is not None:
            lp = lp + torch.log(self.free_mask.clamp_min(1e-6).to(self.device))
        self.log_prob = torch.log_softmax(lp.flatten(), dim=0).reshape(
            self.grid.ny, self.grid.nx
        )
        self._yaw = float(pose.yaw)

    def set_prior_uniform(self) -> None:
        self.log_prob = torch.full(
            (self.grid.ny, self.grid.nx),
            -np.log(self.grid.ny * self.grid.nx),
            device=self.device,
        )

    # -------------------------------------------------------------- predict
    def _gaussian_kernel(self, sigma_px: float) -> torch.Tensor:
        sigma = max(sigma_px / self.grid.resolution, 0.25)
        if self._kernel is not None and self._kernel_sigma == sigma:
            return self._kernel
        radius = int(np.ceil(3.0 * sigma))
        xs = torch.arange(-radius, radius + 1, device=self.device, dtype=torch.float32)
        k = torch.exp(-0.5 * (xs / sigma) ** 2)
        k = (k[:, None] * k[None, :]).float()
        k = k / k.sum()
        self._kernel = k[None, None]
        self._kernel_sigma = sigma
        return self._kernel

    def predict(
        self,
        ego_motion,
        *,
        yaw: float | None = None,
        sigma_xy_px: float | None = None,
    ) -> None:
        """Translate by ``(forward, right, dyaw)`` in the measured body frame."""
        if yaw is not None:
            self._yaw = float(yaw)
        e = torch.as_tensor(ego_motion, device=self.device, dtype=torch.float32)
        fwd, right = float(e[0]), float(e[1])
        c, s = np.cos(self._yaw), np.sin(self._yaw)
        dx = fwd * c - right * s
        dy = fwd * s + right * c
        x = self.log_prob[None, None]
        # Integer part of the translation.
        ix = int(np.floor(dx / self.grid.resolution + 0.5))
        iy = int(np.floor(dy / self.grid.resolution + 0.5))
        if ix or iy:
            x = torch.roll(x, shifts=(iy, ix), dims=(2, 3))
        sigma = self.cfg.sigma_xy_px if sigma_xy_px is None else sigma_xy_px
        k = self._gaussian_kernel(sigma)
        pad = k.shape[-1] // 2
        x = F.pad(x, (pad, pad, pad, pad), mode="replicate")
        x = F.conv2d(x, k)
        self.log_prob = x[0, 0]

    # --------------------------------------------------------------- update
    def update(self, logits, *, weight: float = 1.0) -> None:
        if isinstance(logits, torch.Tensor):
            lp = logits.to(self.device)
            if lp.ndim == 4:  # (B,Y,ny,nx)
                lp = lp[0]
            if lp.ndim == 3:  # (Y,ny,nx)
                lp = torch.logsumexp(lp, dim=0)
        else:
            lp = torch.as_tensor(logits, device=self.device, dtype=torch.float32)
        lp = lp - lp.max()
        if self.cfg.floor_fraction > 0:
            lp = torch.logaddexp(
                lp,
                torch.full_like(lp, float(np.log(self.cfg.floor_fraction))),
            )
        self.log_prob = self.log_prob + float(weight) * lp
        self.log_prob = torch.log_softmax(self.log_prob.flatten(), dim=0).reshape_as(
            self.log_prob
        )
        mix = float(self.cfg.global_mix)
        if mix > 0.0:
            import math

            cand = torch.log_softmax(lp.flatten(), dim=0).reshape_as(lp)
            self.log_prob = torch.logaddexp(
                self.log_prob + math.log1p(-mix),
                cand + math.log(mix),
            )
            self.log_prob = torch.log_softmax(
                self.log_prob.flatten(), dim=0
            ).reshape_as(self.log_prob)

    @property
    def log_volume(self) -> torch.Tensor:
        """Compatibility view: ``(1, ny, nx)`` log posterior."""
        return self.log_prob[None]

    def best(self) -> Pose:
        idx = int(self.log_prob.flatten().argmax())
        ix = idx % self.grid.nx
        iy = idx // self.grid.nx
        xy = self.grid.xy_at(iy, ix)
        return Pose(float(xy[0]), float(xy[1]), self._yaw)

    def marginal_xy(self) -> torch.Tensor:
        return self.log_prob.exp()