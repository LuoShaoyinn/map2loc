"""Batched, GPU-friendly camera observation generator.

The numpy/cv2 generator in :mod:`data_gen.observation` is the reference
implementation (and the one used for previews / evaluation).  This module
renders a whole batch of pinhole camera views with torch operations instead,
which keeps online training fed while the GPU is busy.  It implements the
same camera model, Gaussian-noise + random-invalid-pixel corruption, and
validity masks.

Usage::

    gen = TensorObservationGenerator(map_.image, geometry, photometric)
    poses = sample_poses(...)                    # (B, 5) x,y,yaw,pitch,roll
    batch = gen.sample(poses)
    batch["image"]        # (B, 3, H, W), dataset-normalised
    batch["valid_mask"]   # (B, H, W), 1 = trustworthy
    batch["gt_xy"], batch["yaw_input"], batch["pitch_input"]
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F

from .config import (
    GeometryConfig,
    MeasurementConfig,
    PhotometricConfig,
    TrajectoryConfig,
)


def _uniform(cfg, gen, size, device, dtype=torch.float32):
    lo, hi = float(cfg[0]), float(cfg[1])
    return lo + (hi - lo) * torch.rand(size, device=device, dtype=dtype, generator=gen)


def _wrap_torch(angle: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(angle), torch.cos(angle))


class TensorObservationGenerator:
    def __init__(
        self,
        map_image,  # np.uint8 HWC or torch CHW in [0, 1] or [0, 255]
        geometry: GeometryConfig | None = None,
        photometric: PhotometricConfig | None = None,
        measurement: MeasurementConfig | None = None,
        *,
        device: str | torch.device = "cpu",
    ):
        self.geometry = geometry or GeometryConfig()
        self.photometric = photometric or PhotometricConfig()
        self.measurement = measurement or MeasurementConfig()
        self.device = torch.device(device)

        if isinstance(map_image, np.ndarray):
            x = torch.from_numpy(np.ascontiguousarray(map_image)).float()
            if x.ndim == 3 and x.shape[-1] == 3:
                x = x.permute(2, 0, 1)
            if x.max() > 1.5:
                x = x / 255.0
        else:
            x = map_image.float()
            if x.max() > 1.5:
                x = x / 255.0
        self.map = x.to(self.device)
        h, w = self.map.shape[-2:]
        self.map_h, self.map_w = int(h), int(w)
        k = max(2, min(h, w) // 40)
        corners = torch.cat(
            [
                self.map[:, :k, :k].reshape(3, -1),
                self.map[:, :k, -k:].reshape(3, -1),
                self.map[:, -k:, :k].reshape(3, -1),
                self.map[:, -k:, -k:].reshape(3, -1),
            ],
            dim=1,
        )
        self.background = corners.median(dim=1).values  # (3,)
        self.width = int(self.geometry.obs_width)
        self.height = int(self.geometry.obs_height)
        self._base_grid = None

    # ------------------------------------------------------------- geometry
    def _base(self):
        if self._base_grid is None:
            us = torch.arange(self.width, device=self.device, dtype=torch.float32)
            vs = torch.arange(self.height, device=self.device, dtype=torch.float32)
            gy, gx = torch.meshgrid(vs, us, indexing="ij")
            ones = torch.ones_like(gx)
            self._base_grid = torch.stack([gx, gy, ones], dim=-1)  # H,W,3
        return self._base_grid

    def _camera_basis(self, yaw, pitch, roll):
        cy, sy = torch.cos(yaw), torch.sin(yaw)
        cp, sp = torch.cos(pitch), torch.sin(pitch)
        # Pitch is measured from horizontal (0 = ahead, -90 deg = down).
        forward = torch.stack([cp * cy, cp * sy, sp], dim=-1)
        up = torch.stack([-sp * cy, -sp * sy, cp], dim=-1)
        right = torch.cross(up, forward, dim=-1)

        cr, sr = torch.cos(roll), torch.sin(roll)
        right_rolled = cr[..., None] * right + sr[..., None] * up
        up_rolled = cr[..., None] * up - sr[..., None] * right
        return right_rolled, up_rolled, forward

    def _augmentation(self, b: int, gen, enabled: bool):
        dev = self.device
        if not enabled:
            zeros = torch.zeros(b, device=dev)
            return (
                torch.ones(b, device=dev),
                zeros,
                zeros.clone(),
                zeros.clone(),
            )
        g = self.geometry
        scale = _uniform(g.scale, gen, (b,), dev)
        head_pitch = (
            _uniform((-g.head_pitch_deg, g.head_pitch_deg), gen, (b,), dev)
            * math.pi
            / 180.0
        )
        head_yaw = (
            _uniform((-g.head_yaw_deg, g.head_yaw_deg), gen, (b,), dev)
            * math.pi
            / 180.0
        )
        head_roll = (
            _uniform((-g.head_roll_deg, g.head_roll_deg), gen, (b,), dev)
            * math.pi
            / 180.0
        )
        return scale, head_pitch, head_yaw, head_roll

    def _homography(self, poses: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        """Map pixels -> view pixels, ``(B, 3, 3)``."""
        b = poses.shape[0]
        dev = self.device
        x, y, yaw, pitch, roll = (poses[:, i] for i in range(5))
        height = (
            poses[:, 5]
            if poses.shape[1] > 5
            else torch.full((b,), float(self.geometry.height_px), device=dev)
        )
        height = torch.where(
            height > 0,
            height,
            torch.full_like(height, float(self.geometry.height_px)),
        )

        right, up, forward = self._camera_basis(yaw, pitch, roll)
        centre = torch.stack([x, y, height], dim=-1)

        focal = height * self.width / float(self.geometry.footprint_px) * scale
        cx = (self.width - 1) / 2.0
        cy = (self.height - 1) / 2.0

        linear = torch.zeros((b, 3, 3), device=dev, dtype=poses.dtype)
        linear[:, 0, :2] = right[:, :2]
        linear[:, 0, 2] = -(right * centre).sum(-1)
        linear[:, 1, :2] = -up[:, :2]
        linear[:, 1, 2] = (up * centre).sum(-1)
        linear[:, 2, :2] = forward[:, :2]
        linear[:, 2, 2] = -(forward * centre).sum(-1)

        intrinsics = torch.zeros((b, 3, 3), device=dev, dtype=poses.dtype)
        intrinsics[:, 0, 0] = focal
        intrinsics[:, 1, 1] = focal
        intrinsics[:, 0, 2] = cx
        intrinsics[:, 1, 2] = cy
        intrinsics[:, 2, 2] = 1.0
        return intrinsics @ linear

    def _render(self, poses: torch.Tensor, scale: torch.Tensor) -> dict:
        b = poses.shape[0]
        h_map_to_view = self._homography(poses, scale)
        h_view_to_map = torch.linalg.inv(h_map_to_view)

        grid = self._base()[None].expand(b, -1, -1, -1)
        pts = torch.einsum("bhwi,bji->bhwj", grid, h_view_to_map)
        front = pts[..., 2] > 0.0
        xy = pts[..., :2] / pts[..., 2:3].clamp_min(1e-8)
        mx, my = xy[..., 0], xy[..., 1]
        inside = (
            (mx >= 0)
            & (mx <= self.map_w - 1)
            & (my >= 0)
            & (my <= self.map_h - 1)
            & front
        )
        gx = 2.0 * mx / max(self.map_w - 1, 1) - 1.0
        gy = 2.0 * my / max(self.map_h - 1, 1) - 1.0
        sample_grid = torch.stack([gx, gy], dim=-1)
        img = F.grid_sample(
            self.map[None].expand(b, -1, -1, -1),
            sample_grid,
            align_corners=True,
            padding_mode="zeros",
        )
        bg = self.background[:, None, None]
        img = torch.where(inside[:, None], img, bg[None])
        valid = inside.float()[:, None]

        # Exact map point under the image centre.
        centre = torch.tensor(
            [[(self.width - 1) / 2.0, (self.height - 1) / 2.0, 1.0]],
            device=self.device,
        )
        p = centre[None] @ h_view_to_map.transpose(1, 2)
        centre_px = p[:, 0, :2] / p[:, 0, 2:3].clamp_min(1e-8)
        # If the image centre points at/above the horizon there is no ground
        # point there; fall back to the camera position (finite target).
        centre_px = torch.where(
            (p[:, 0, 2:3] > 0.0).expand_as(centre_px),
            centre_px,
            poses[:, :2],
        )
        return {"image": img, "valid": valid, "center_xy": centre_px}

    # ------------------------------------------------------------ photometry
    def _photometric(self, img: torch.Tensor, valid: torch.Tensor, gen):
        cfg = self.photometric
        if not cfg.enabled:
            return img, valid
        b, _, h, w = img.shape
        dev = self.device

        # Uniform-sigma i.i.d. Gaussian noise.
        sigma = _uniform(cfg.gauss_sigma, gen, (b, 1, 1, 1), dev) / 255.0
        img = img + sigma * torch.randn(img.shape, device=dev, generator=gen)
        img = img.clamp(0.0, 1.0)

        # Randomly invalid pixels.
        frac = _uniform(cfg.random_invalid_frac, gen, (b, 1, 1, 1), dev)
        invalid = torch.rand((b, 1, h, w), device=dev, generator=gen) < frac
        valid = valid * (~invalid).float()
        return img, valid

    # ---------------------------------------------------------------- public
    @torch.no_grad()
    def sample(
        self,
        poses: torch.Tensor,
        generator: torch.Generator | None = None,
        *,
        geometry: bool = True,
        photometric: bool = True,
        measurement=None,
    ) -> dict:
        """Render ``B`` views for poses ``(B, 5)`` in map px / radians."""
        poses = poses.to(self.device).float()
        if poses.shape[-1] < 5:
            pad = torch.zeros(
                (poses.shape[0], 5 - poses.shape[-1]),
                device=self.device,
                dtype=poses.dtype,
            )
            poses = torch.cat([poses, pad], dim=-1)
        b = poses.shape[0]
        gen = generator

        scale, head_pitch, head_yaw, head_roll = self._augmentation(
            b, gen, geometry
        )

        eff = poses.clone()
        eff[:, 2] = _wrap_torch(eff[:, 2] + head_yaw)
        eff[:, 3] = eff[:, 3] + head_pitch
        eff[:, 4] = eff[:, 4] + head_roll

        out = self._render(eff, scale)
        if photometric:
            img, valid = self._photometric(out["image"], out["valid"], gen)
        else:
            img, valid = out["image"].clamp(0, 1), out["valid"]

        if measurement is None:
            measurement = self.measurement
        if measurement is not None:
            yaw_sigma = math.radians(measurement.yaw_sigma_deg)
            pitch_sigma = math.radians(measurement.pitch_sigma_deg)
        else:
            yaw_sigma = pitch_sigma = 0.0
        yaw_input = _wrap_torch(
            eff[:, 2]
            + yaw_sigma
            * torch.randn(b, device=self.device, generator=gen)
        )
        pitch_input = eff[:, 3] + pitch_sigma * torch.randn(
            b, device=self.device, generator=gen
        )

        image = (img.clamp(0, 1) - 0.5) / 0.5  # same normalisation as datasets
        return {
            "image": image,
            "valid_mask": valid[:, 0],
            "gt_xy": poses[:, :2],
            "center_xy": out["center_xy"],
            "yaw": eff[:, 2],
            "pitch": eff[:, 3],
            "roll": eff[:, 4],
            "yaw_input": yaw_input,
            "pitch_input": pitch_input,
            "pose": eff,
            "scale": scale,
        }


class TensorPoseSampler:
    """Uniform camera-pose sampler on a map region, entirely on the device."""

    def __init__(
        self,
        region,
        device="cpu",
        trajectory: TrajectoryConfig | None = None,
    ):
        self.region = region
        self.device = torch.device(device)
        self.trajectory = trajectory or TrajectoryConfig()

    def sample(
        self, batch_size: int, generator: torch.Generator | None = None
    ) -> torch.Tensor:
        dev = self.device
        x = self.region.x0 + (self.region.x1 - self.region.x0) * torch.rand(
            batch_size, device=dev, generator=generator
        )
        y = self.region.y0 + (self.region.y1 - self.region.y0) * torch.rand(
            batch_size, device=dev, generator=generator
        )
        yaw = (
            torch.rand(batch_size, device=dev, generator=generator) * 2 - 1
        ) * math.pi
        cfg = self.trajectory
        pitch = math.radians(cfg.pitch_min_deg) + (
            math.radians(cfg.pitch_max_deg) - math.radians(cfg.pitch_min_deg)
        ) * torch.rand(batch_size, device=dev, generator=generator)
        roll = torch.clamp(
            math.radians(cfg.roll_sigma_deg)
            * torch.randn(batch_size, device=dev, generator=generator),
            -math.radians(cfg.roll_limit_deg),
            math.radians(cfg.roll_limit_deg),
        )
        return torch.stack([x, y, yaw, pitch, roll], dim=1)


class TensorTrajectorySampler:
    """Batched smooth random-walk camera trajectories, on the device.

    ``x``, ``y`` and ``yaw`` follow a smooth turning random walk; ``pitch``
    follows a smooth oblique range and ``roll`` a small OU curve around zero.  Border handling
    reflects the heading, which keeps every pose inside the requested region.
    """

    def __init__(
        self,
        region,
        *,
        seq_len: int,
        step_stride: int = 6,
        step_px: float = 3.0,
        trajectory: TrajectoryConfig | None = None,
        device="cpu",
    ):
        self.region = region
        self.seq_len = int(seq_len)
        self.step_stride = int(step_stride)
        self.step_px = float(step_px)
        self.trajectory = trajectory or TrajectoryConfig()
        self.device = torch.device(device)

    def sample(
        self,
        batch_size: int,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        dev = self.device
        b, t = batch_size, self.seq_len
        cfg = self.trajectory
        x = self.region.x0 + (self.region.x1 - self.region.x0) * torch.rand(
            b, device=dev, generator=generator
        )
        y = self.region.y0 + (self.region.y1 - self.region.y0) * torch.rand(
            b, device=dev, generator=generator
        )
        yaw = (torch.rand(b, device=dev, generator=generator) * 2 - 1) * math.pi
        speed = self.step_px * self.step_stride * (
            0.75 + 0.5 * torch.rand(b, device=dev, generator=generator)
        )
        # Smooth per-sequence turn rate (rad / frame).
        turn = torch.zeros(b, device=dev)
        turn_rate = 0.10 + 0.10 * torch.rand(b, device=dev, generator=generator)

        pitch = torch.clamp(
            math.radians(cfg.pitch_mean_deg)
            + math.radians(cfg.pitch_sigma_deg)
            * torch.randn(b, device=dev, generator=generator),
            math.radians(cfg.pitch_min_deg),
            math.radians(cfg.pitch_max_deg),
        )
        roll = torch.clamp(
            math.radians(cfg.roll_sigma_deg)
            * torch.randn(b, device=dev, generator=generator),
            -math.radians(cfg.roll_limit_deg),
            math.radians(cfg.roll_limit_deg),
        )
        pitch_step = math.radians(cfg.pitch_sigma_deg) * math.sqrt(
            max(0.0, 1.0 - cfg.pitch_corr**2)
        )
        roll_step = math.radians(cfg.roll_sigma_deg) * math.sqrt(
            max(0.0, 1.0 - cfg.roll_corr**2)
        )
        pitch_mean = math.radians(cfg.pitch_mean_deg)
        pitch_min = math.radians(cfg.pitch_min_deg)
        pitch_max = math.radians(cfg.pitch_max_deg)
        roll_limit = math.radians(cfg.roll_limit_deg)
        pitch_max_step = math.radians(cfg.pitch_max_step_deg)
        roll_max_step = math.radians(cfg.roll_max_step_deg)

        poses = []
        margin = max(8.0, self.step_px * self.step_stride)
        x0, x1 = self.region.x0 + margin, self.region.x1 - margin
        y0, y1 = self.region.y0 + margin, self.region.y1 - margin
        if x1 <= x0:
            x0, x1 = self.region.x0, self.region.x1
        if y1 <= y0:
            y0, y1 = self.region.y0, self.region.y1
        for _ in range(t):
            poses.append(torch.stack([x, y, yaw, pitch, roll], dim=1))
            turn = 0.85 * turn + 0.15 * (
                (torch.rand(b, device=dev, generator=generator) * 2 - 1)
                * turn_rate
            )
            yaw = yaw + turn
            x = x + speed * torch.cos(yaw)
            y = y + speed * torch.sin(yaw)

            pitch_target = torch.clamp(
                pitch_mean
                + cfg.pitch_corr * (pitch - pitch_mean)
                + pitch_step
                * torch.randn(b, device=dev, generator=generator),
                pitch_min,
                pitch_max,
            )
            pitch = pitch + torch.clamp(
                pitch_target - pitch, -pitch_max_step, pitch_max_step
            )
            roll_target = torch.clamp(
                cfg.roll_corr * roll
                + roll_step
                * torch.randn(b, device=dev, generator=generator),
                -roll_limit,
                roll_limit,
            )
            roll = roll + torch.clamp(
                roll_target - roll, -roll_max_step, roll_max_step
            )

            # Reflect at the region boundary.
            over_x = (x < x0) | (x > x1)
            over_y = (y < y0) | (y > y1)
            yaw = torch.where(over_x, math.pi - yaw, yaw)
            yaw = torch.where(over_y, -yaw, yaw)
            x = x.clamp(x0, x1)
            y = y.clamp(y0, y1)
        return torch.stack(poses, dim=1)  # (B, T, 5)