"""Batched, GPU-friendly observation generator.

The numpy/cv2 generator in :mod:`map2loc.observation` is the reference
implementation (and the one used for evaluation).  This module renders a whole
batch of BEV observations with torch operations instead, which keeps online
training fed while the GPU is busy.  It implements the same geometric model and
the important photometric corruptions, and it returns per-pixel validity masks
so the observation encoder can learn to ignore scratches/occluders.

Usage::

    gen = TensorObservationGenerator(map_.image, geometry, photometric)
    poses = sample_poses(...)                    # (B, 3) map pixels / radians
    batch = gen.sample(poses)
    batch["image"]        # (B, 3, H, W), dataset-normalised
    batch["valid_mask"]   # (B, H, W), 1 = trustworthy
    batch["gt_xy"], batch["yaw"]
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F

from .config import GeometryConfig, PhotometricConfig


def _uniform(cfg, gen, size, device, dtype=torch.float32):
    lo, hi = float(cfg[0]), float(cfg[1])
    return lo + (hi - lo) * torch.rand(size, device=device, dtype=dtype, generator=gen)


def _randint(gen, lo: int, hi: int, size, device):
    if hi <= lo:
        return torch.full(size, int(lo), device=device, dtype=torch.long)
    return torch.randint(int(lo), int(hi) + 1, size, device=device, generator=gen)


class TensorObservationGenerator:
    def __init__(
        self,
        map_image,  # np.uint8 HWC or torch CHW in [0, 1] or [0, 255]
        geometry: GeometryConfig | None = None,
        photometric: PhotometricConfig | None = None,
        *,
        device: str | torch.device = "cpu",
    ):
        self.geometry = geometry or GeometryConfig()
        self.photometric = photometric or PhotometricConfig()
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

    def _geometry(self, b: int, gen):
        g = self.geometry
        dev = self.device
        zoom = torch.exp(_uniform((math.log(g.zoom[0]), math.log(g.zoom[1])), gen, (b,), dev))
        aspect = _uniform(g.aspect, gen, (b,), dev)
        sx = zoom * aspect
        sy = zoom / aspect
        shx = _uniform((-g.shear, g.shear), gen, (b,), dev)
        shy = _uniform((-g.shear, g.shear), gen, (b,), dev)
        rot = _uniform((-g.rot_deg, g.rot_deg), gen, (b,), dev) * math.pi / 180.0
        tx = _uniform((-g.translate_px, g.translate_px), gen, (b,), dev)
        ty = _uniform((-g.translate_px, g.translate_px), gen, (b,), dev)

        c, s = torch.cos(rot), torch.sin(rot)
        # K = R @ S @ shear
        k00 = c * sx
        k01 = (-s) * sy + c * sx * shx
        k10 = s * sx
        k11 = c * sy + s * sx * shx
        # shear_y acts on x before scale: K = R @ [[1,0],[shy,1]] @ S
        k00 = k00 + (-s) * sy * shy
        k10 = k10 + c * sy * shy

        cx = (self.width - 1) / 2.0
        cy = (self.height - 1) / 2.0
        a = torch.zeros((b, 3, 3), device=dev)
        a[:, 0, 0], a[:, 0, 1] = k00, k01
        a[:, 1, 0], a[:, 1, 1] = k10, k11
        a[:, 0, 2] = cx + tx - (k00 * cx + k01 * cy)
        a[:, 1, 2] = cy + ty - (k10 * cx + k11 * cy)
        a[:, 2, 2] = 1.0

        # Mild projective term.
        persp = torch.rand(b, device=dev, generator=gen) < g.persp_prob
        px = _uniform((-g.persp_strength, g.persp_strength), gen, (b,), dev)
        py = _uniform((-g.persp_strength, g.persp_strength), gen, (b,), dev)
        a[:, 2, 0] = torch.where(persp, px, torch.zeros_like(px))
        a[:, 2, 1] = torch.where(persp, py, torch.zeros_like(py))
        return a

    def _nominal(self, poses: torch.Tensor):
        b = poses.shape[0]
        x, y, yaw = poses[:, 0], poses[:, 1], poses[:, 2]
        s = self.width / float(self.geometry.fov_px)
        c, sn = torch.cos(yaw), torch.sin(yaw)
        # R = [[-sin, -cos], [cos, -sin]]
        r00, r01 = -sn, -c
        r10, r11 = c, -sn
        cx = (self.width - 1) / 2.0
        cy = (self.height - 1) / 2.0
        h = torch.zeros((b, 3, 3), device=self.device)
        h[:, 0, 0], h[:, 0, 1] = r00 / s, r01 / s
        h[:, 1, 0], h[:, 1, 1] = r10 / s, r11 / s
        h[:, 0, 2] = x - (r00 * cx + r01 * cy) / s
        h[:, 1, 2] = y - (r10 * cx + r11 * cy) / s
        h[:, 2, 2] = 1.0
        return h

    def _render(self, poses: torch.Tensor, geom: torch.Tensor) -> dict:
        b = poses.shape[0]
        h_total = self._nominal(poses) @ geom
        grid = self._base()[None].expand(b, -1, -1, -1)
        pts = torch.einsum("bhwi,bji->bhwj", grid, h_total)  # B,H,W,3
        xy = pts[..., :2] / pts[..., 2:3].clamp_min(1e-8)
        mx, my = xy[..., 0], xy[..., 1]
        inside = (
            (mx >= 0)
            & (mx <= self.map_w - 1)
            & (my >= 0)
            & (my <= self.map_h - 1)
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

        # Exact target = homography applied to the observation centre.
        centre = torch.tensor(
            [[(self.width - 1) / 2.0, (self.height - 1) / 2.0, 1.0]],
            device=self.device,
        )
        p = centre[None] @ h_total.transpose(1, 2)
        gt_xy = p[:, 0, :2] / p[:, 0, 2:3]
        return {"image": img, "valid": valid, "gt_xy": gt_xy}

    # ------------------------------------------------------------ photometry
    def _photometric(self, img: torch.Tensor, valid: torch.Tensor, gen):
        cfg = self.photometric
        if not cfg.enabled:
            return img, valid
        b, _, h, w = img.shape
        dev = self.device

        # illumination: gain / gamma / tint / vignette / gradient
        gain = _uniform(cfg.gain, gen, (b, 1, 1, 1), dev)
        gamma = _uniform(cfg.gamma, gen, (b, 1, 1, 1), dev)
        tint = 1.0 + _uniform((-cfg.tint, cfg.tint), gen, (b, 3, 1, 1), dev)
        img = img * gain * tint
        img = img.clamp(1e-4, 1.0) ** gamma

        yy, xx = torch.meshgrid(
            torch.linspace(-1, 1, h, device=dev),
            torch.linspace(-1, 1, w, device=dev),
            indexing="ij",
        )
        r2 = (xx * xx + yy * yy) / 2.0
        vig = _uniform(cfg.vignette, gen, (b, 1, 1, 1), dev)
        field = 1.0 - vig * r2
        grad = _uniform(cfg.gradient, gen, (b, 1, 1, 1), dev)
        ang = _uniform((0.0, 2 * math.pi), gen, (b, 1, 1, 1), dev)
        ramp = (torch.cos(ang) * xx + torch.sin(ang) * yy) / 2.0
        field = field * (1.0 + grad * ramp)
        img = img * field

        # Blur (fixed Gaussian; random per batch is enough for augmentation).
        if cfg.blur_max >= 3:
            k = torch.tensor(
                [1.0, 2.0, 1.0], device=dev, dtype=img.dtype
            )
            k = (k[:, None] * k[None, :]).view(1, 1, 3, 3)
            k = k / k.sum()
            blurred = F.conv2d(
                img.reshape(b * 3, 1, h, w), k, padding=1
            ).reshape(b, 3, h, w)
            do_blur = (
                torch.rand(b, 1, 1, 1, device=dev, generator=gen) < cfg.blur_prob
            )
            img = torch.where(do_blur, blurred, img)

        # Gaussian noise.
        sigma = _uniform(cfg.gauss_sigma, gen, (b, 1, 1, 1), dev) / 255.0
        img = img + sigma * torch.randn(img.shape, device=dev, generator=gen)
        img = img.clamp(0.0, 1.0)

        # Salt-and-pepper.
        sp = _uniform(cfg.salt_pepper, gen, (b, 1, 1, 1), dev)
        sp_mask = torch.rand((b, 1, h, w), device=dev, generator=gen) < sp
        salt = sp_mask & (torch.rand((b, 1, h, w), device=dev, generator=gen) < 0.5)
        pepper = sp_mask & ~salt
        img = torch.where(salt, torch.ones_like(img), img)
        img = torch.where(pepper, torch.zeros_like(img), img)
        valid = valid * (~sp_mask).float()

        # Occluders: filled rounded rectangles.
        if cfg.occlusion_prob > 0:
            valid = self._occlusions(img, valid, gen)
        # Scratches/dust: a few rasterised polylines and dots.
        if cfg.scratch_prob > 0:
            valid = self._scratches(img, valid, gen)
        return img, valid

    def _occlusions(self, img, valid, gen):
        cfg = self.photometric
        b, _, h, w = img.shape
        dev = self.device
        for i in range(b):
            if float(torch.rand((), device=dev, generator=gen)) >= cfg.occlusion_prob:
                continue
            n = int(
                torch.randint(
                    cfg.occlusion_count[0],
                    cfg.occlusion_count[1] + 1,
                    (),
                    device=dev,
                    generator=gen,
                )
            )
            for _ in range(n):
                f = float(_uniform(cfg.occlusion_frac, gen, (), dev))
                ow = int(max(4, min(w, w * math.sqrt(f) * float(_uniform((0.6, 1.4), gen, (), dev)))))
                oh = int(max(4, min(h, h * math.sqrt(f) * float(_uniform((0.6, 1.4), gen, (), dev)))))
                x0 = int(torch.randint(0, max(1, w - ow), (), device=dev, generator=gen))
                y0 = int(torch.randint(0, max(1, h - oh), (), device=dev, generator=gen))
                if float(torch.rand((), device=dev, generator=gen)) < 0.75:
                    colour = _uniform((0.02, 0.27), gen, (3,), dev)
                else:
                    colour = _uniform((0.67, 0.96), gen, (3,), dev)
                a = float(_uniform(cfg.occlusion_alpha, gen, (), dev))
                ys = torch.arange(y0, y0 + oh, device=dev)
                xs = torch.arange(x0, x0 + ow, device=dev)
                gy, gx = torch.meshgrid(ys, xs, indexing="ij")
                yy = (gy - (y0 + oh / 2)).abs() / (oh / 2 + 1e-6)
                xx = (gx - (x0 + ow / 2)).abs() / (ow / 2 + 1e-6)
                mask = ((xx**4 + yy**4) <= 1.0).float() * a
                region = img[i, :, y0 : y0 + oh, x0 : x0 + ow]
                img[i, :, y0 : y0 + oh, x0 : x0 + ow] = (
                    region * (1 - mask) + colour[:, None, None] * mask
                )
                vreg = valid[i, :, y0 : y0 + oh, x0 : x0 + ow]
                valid[i, :, y0 : y0 + oh, x0 : x0 + ow] = vreg * (1 - mask)
        return valid

    def _scratches(self, img, valid, gen):
        cfg = self.photometric
        b, _, h, w = img.shape
        dev = self.device
        diag = math.hypot(w, h)
        for i in range(b):
            if float(torch.rand((), device=dev, generator=gen)) >= cfg.scratch_prob:
                continue
            n = int(
                torch.randint(
                    cfg.scratch_count[0], cfg.scratch_count[1] + 1, (), device=dev, generator=gen
                )
            )
            mask = torch.zeros((h, w), device=dev)
            for _ in range(n):
                p = torch.tensor(
                    [
                        float(torch.rand((), device=dev, generator=gen)) * w,
                        float(torch.rand((), device=dev, generator=gen)) * h,
                    ],
                    device=dev,
                )
                ang = float(torch.rand((), device=dev, generator=gen)) * 2 * math.pi
                length = float(_uniform(cfg.scratch_length, gen, (), dev)) * diag
                nseg = int(torch.randint(1, 5, (), device=dev, generator=gen))
                for _seg in range(nseg):
                    ang += float(torch.randn((), device=dev, generator=gen)) * 0.18
                    step = length / nseg
                    p = p + step * torch.tensor([math.cos(ang), math.sin(ang)], device=dev)
                    npts = max(2, int(step))
                    ts = torch.linspace(0, 1, npts, device=dev)
                    prev = p - step * torch.tensor([math.cos(ang), math.sin(ang)], device=dev)
                    xs = (prev[0] + (p[0] - prev[0]) * ts).round().long().clamp(0, w - 1)
                    ys = (prev[1] + (p[1] - prev[1]) * ts).round().long().clamp(0, h - 1)
                    mask[ys, xs] = 1.0
            width = int(
                torch.randint(
                    cfg.scratch_width[0], cfg.scratch_width[1] + 1, (), device=dev, generator=gen
                )
            )
            if width > 1:
                mask = F.max_pool2d(
                    mask[None, None], width, stride=1, padding=width // 2
                )[0, 0, :h, :w]
            a = float(_uniform(cfg.scratch_alpha, gen, (), dev))
            if float(torch.rand((), device=dev, generator=gen)) < 0.65:
                colour = _uniform((0.75, 1.0), gen, (3,), dev)
            elif float(torch.rand((), device=dev, generator=gen)) < 0.6:
                colour = _uniform((0.0, 0.18), gen, (3,), dev)
            else:
                colour = self.background + _uniform((0.0, 0.16), gen, (3,), dev)
            m = (mask * a)[None]
            img[i] = img[i] * (1 - m) + colour[:, None, None] * m
            valid[i] = valid[i] * (1 - m)
        return valid

    # ---------------------------------------------------------------- public
    @torch.no_grad()
    def sample(
        self,
        poses: torch.Tensor,
        generator: torch.Generator | None = None,
        *,
        geometry: bool = True,
        photometric: bool = True,
    ) -> dict:
        """Render ``B`` observations for poses ``(B, 3)`` in map pixels/radians."""
        poses = poses.to(self.device).float()
        b = poses.shape[0]
        gen = generator
        if geometry:
            geom = self._geometry(b, gen)
        else:
            geom = torch.eye(3, device=self.device, dtype=torch.float32)[None].expand(b, -1, -1)
        out = self._render(poses, geom)
        if photometric:
            img, valid = self._photometric(out["image"], out["valid"], gen)
        else:
            img, valid = out["image"].clamp(0, 1), out["valid"]
        image = (img.clamp(0, 1) - 0.5) / 0.5  # same normalisation as datasets
        return {
            "image": image,
            "valid_mask": valid[:, 0],
            "gt_xy": out["gt_xy"],
            "yaw": poses[:, 2],
            "pose": poses,
        }


class TensorPoseSampler:
    """Uniform pose sampler on a map region, entirely on the device."""

    def __init__(self, region, device="cpu"):
        self.region = region
        self.device = torch.device(device)

    def sample(self, batch_size: int, generator: torch.Generator | None = None) -> torch.Tensor:
        x = self.region.x0 + (self.region.x1 - self.region.x0) * torch.rand(
            batch_size, device=self.device, generator=generator
        )
        y = self.region.y0 + (self.region.y1 - self.region.y0) * torch.rand(
            batch_size, device=self.device, generator=generator
        )
        yaw = (torch.rand(batch_size, device=self.device, generator=generator) * 2 - 1) * math.pi
        return torch.stack([x, y, yaw], dim=1)


class TensorTrajectorySampler:
    """Batched smooth random-walk trajectories, entirely on the device.

    Each sequence has a constant-ish forward speed and a smooth turn rate so
    that consecutive frames overlap like they do in the evaluation protocol.
    Border handling reflects the heading, which keeps every pose inside the
    requested region without needing a CPU trajectory generator.
    """

    def __init__(
        self,
        region,
        *,
        seq_len: int,
        step_stride: int = 6,
        step_px: float = 3.0,
        device="cpu",
    ):
        self.region = region
        self.seq_len = int(seq_len)
        self.step_stride = int(step_stride)
        self.step_px = float(step_px)
        self.device = torch.device(device)

    def sample(
        self,
        batch_size: int,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        dev = self.device
        b, t = batch_size, self.seq_len
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
        poses = []
        margin = max(8.0, self.step_px * self.step_stride)
        x0, x1 = self.region.x0 + margin, self.region.x1 - margin
        y0, y1 = self.region.y0 + margin, self.region.y1 - margin
        if x1 <= x0:
            x0, x1 = self.region.x0, self.region.x1
        if y1 <= y0:
            y0, y1 = self.region.y0, self.region.y1
        for _ in range(t):
            poses.append(torch.stack([x, y, yaw], dim=1))
            turn = 0.85 * turn + 0.15 * (
                (torch.rand(b, device=dev, generator=generator) * 2 - 1) * turn_rate
            )
            yaw = yaw + turn
            x = x + speed * torch.cos(yaw)
            y = y + speed * torch.sin(yaw)
            # Reflect at the region boundary.
            over_x = (x < x0) | (x > x1)
            over_y = (y < y0) | (y > y1)
            yaw = torch.where(over_x, math.pi - yaw, yaw)
            yaw = torch.where(over_y, -yaw, yaw)
            x = x.clamp(x0, x1)
            y = y.clamp(y0, y1)
        return torch.stack(poses, dim=1)  # (B, T, 3)