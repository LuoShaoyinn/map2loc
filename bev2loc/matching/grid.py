"""Shared BEV pose grid and likelihood-volume container.

A likelihood volume is a dense tensor over ``(yaw, y, x)``.  ``x`` and ``y``
index a regular lattice in map pixel coordinates; ``yaw`` indexes a set of
equally spaced headings.  Every matcher and every temporal model in this
package speaks this representation, which is what makes the comparison in
:mod:`bev2loc.evaluation` exactly like-for-like.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn.functional as F

from ..map import FixedMap
from ..types import Pose

if TYPE_CHECKING:  # pragma: no cover
    pass


def wrap_angle(a):
    return (a + np.pi) % (2.0 * np.pi) - np.pi


@dataclass(frozen=True)
class PoseGrid:
    """Regular ``(x, y, yaw)`` lattice over a map image.

    Parameters
    ----------
    map_width, map_height:
        Size of the map image in pixels.
    resolution:
        Map pixels per grid cell.  Cell ``ij`` is centred at
        ``((j + 0.5) * resolution, (i + 0.5) * resolution)``.
    n_yaw:
        Number of heading bins, evenly spaced on ``[-pi, pi)``.
    """

    map_width: int
    map_height: int
    resolution: float = 8.0
    n_yaw: int = 64

    @classmethod
    def from_map(
        cls, map_: FixedMap, *, resolution: float = 8.0, n_yaw: int = 64
    ) -> "PoseGrid":
        return cls(
            map_width=int(map_.width),
            map_height=int(map_.height),
            resolution=float(resolution),
            n_yaw=int(n_yaw),
        )

    @property
    def nx(self) -> int:
        return int(np.ceil(self.map_width / self.resolution))

    @property
    def ny(self) -> int:
        return int(np.ceil(self.map_height / self.resolution))

    @property
    def dyaw(self) -> float:
        return 2.0 * np.pi / self.n_yaw

    @property
    def xs(self) -> np.ndarray:
        return (np.arange(self.nx, dtype=np.float32) + 0.5) * self.resolution

    @property
    def ys(self) -> np.ndarray:
        return (np.arange(self.ny, dtype=np.float32) + 0.5) * self.resolution

    @property
    def yaws(self) -> np.ndarray:
        return np.linspace(-np.pi, np.pi, self.n_yaw, endpoint=False).astype(
            np.float32
        )

    @property
    def shape(self) -> tuple[int, int, int]:
        """Volume shape ``(n_yaw, ny, nx)``."""
        return (self.n_yaw, self.ny, self.nx)

    # ---------------------------------------------------------------- coords
    def xy_at(self, iy: int, ix: int) -> np.ndarray:
        return np.array(
            [
                (ix + 0.5) * self.resolution,
                (iy + 0.5) * self.resolution,
            ],
            dtype=np.float32,
        )

    def pose_at(self, iyaw: int, iy: int, ix: int) -> Pose:
        xy = self.xy_at(iy, ix)
        return Pose(float(xy[0]), float(xy[1]), float(self.yaws[iyaw]))

    def bins_of(self, xy, yaw: float) -> tuple[int, int, int]:
        """Nearest ``(iyaw, iy, ix)`` bin, with clamping at the map border."""
        ix = int(np.clip(round(float(xy[0]) / self.resolution - 0.5), 0, self.nx - 1))
        iy = int(np.clip(round(float(xy[1]) / self.resolution - 0.5), 0, self.ny - 1))
        iyaw = int(round((float(yaw) - (-np.pi)) / self.dyaw)) % self.n_yaw
        return iyaw, iy, ix

    def nearest_pose(self, xy, yaw: float) -> Pose:
        return self.pose_at(*self.bins_of(xy, yaw))

    def flat_index(self, xy, yaw: float) -> int:
        iyaw, iy, ix = self.bins_of(xy, yaw)
        return (iyaw * self.ny + iy) * self.nx + ix

    def volume_coords(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Meshgrid of world x, y and yaw for every volume element."""
        ys, xs = np.meshgrid(self.ys, self.xs, indexing="ij")
        yaws = np.broadcast_to(self.yaws[:, None, None], self.shape)
        return xs, ys, yaws

    def to(self, device=None, dtype=torch.float32) -> dict[str, torch.Tensor]:
        xs, ys, yaws = self.volume_coords()
        return {
            "x": torch.as_tensor(np.array(xs, copy=True), dtype=dtype, device=device),
            "y": torch.as_tensor(np.array(ys, copy=True), dtype=dtype, device=device),
            "yaw": torch.as_tensor(np.array(yaws, copy=True), dtype=dtype, device=device),
            "yaws": torch.as_tensor(np.array(self.yaws, copy=True), dtype=dtype, device=device),
        }


@dataclass
class LikelihoodVolume:
    """Dense per-frame likelihood over ``(yaw, y, x)``.

    ``logits`` has shape ``(n_yaw, ny, nx)`` or a leading batch dimension
    ``(B, n_yaw, ny, nx)``.  Log-space is used throughout so that temporal
    fusion is a simple sum and never underflows.
    """

    logits: torch.Tensor
    grid: PoseGrid

    def __post_init__(self) -> None:
        if self.logits.ndim not in (3, 4):
            raise ValueError(
                f"logits must be (Y,H,W) or (B,Y,H,W), got {tuple(self.logits.shape)}"
            )
        expected = self.grid.shape
        if self.logits.shape[-3:] != expected:
            raise ValueError(
                f"logits spatial shape {tuple(self.logits.shape[-3:])} != grid {expected}"
            )

    @property
    def batched(self) -> bool:
        return self.logits.ndim == 4

    def probs(self, temperature: float = 1.0) -> torch.Tensor:
        logits = self.logits / float(temperature)
        return torch.softmax(logits.flatten(-3), dim=-1).reshape(logits.shape)

    def log_probs(self, temperature: float = 1.0) -> torch.Tensor:
        logits = self.logits / float(temperature)
        return torch.log_softmax(logits.flatten(-3), dim=-1).reshape(logits.shape)

    def best_bins(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return ``(iyaw, iy, ix)`` of the maximum, supporting batches."""
        shape = self.logits.shape
        flat = self.logits.flatten(-3)
        idx = flat.argmax(dim=-1)
        nx, ny = self.grid.nx, self.grid.ny
        ix = idx % nx
        iy = (idx // nx) % ny
        iyaw = idx // (nx * ny)
        return iyaw, iy, ix

    def best_pose(self) -> Pose:
        if self.batched:
            raise ValueError("best_pose() is only defined for a single volume")
        iyaw, iy, ix = (int(v) for v in self.best_bins())
        return self.grid.pose_at(iyaw, iy, ix)

    def best_poses(self) -> list[Pose]:
        if not self.batched:
            return [self.best_pose()]
        iyaw, iy, ix = self.best_bins()
        return [
            self.grid.pose_at(int(a), int(b), int(c))
            for a, b, c in zip(iyaw, iy, ix)
        ]

    def marginal_xy(self) -> torch.Tensor:
        """Sum over heading, shape ``(ny, nx)`` (or ``(B, ny, nx)``)."""
        p = self.probs()
        return p.sum(dim=-3)

    def entropy(self) -> torch.Tensor:
        lp = self.log_probs()
        p = lp.exp()
        return -(p * lp).flatten(-3).sum(-1)

    def to_log_volume(self, temperature: float = 1.0) -> torch.Tensor:
        """Natural-log probabilities; convenient for temporal fusion."""
        return self.log_probs(temperature)


def map_raster(
    map_: FixedMap,
    grid: PoseGrid,
    *,
    grayscale: bool = False,
    normalize: bool = True,
    device=None,
) -> dict[str, torch.Tensor]:
    """Rasterise the fixed map onto ``grid``.

    The returned tensors are ``image`` (``1xCxHxW``), ``valid``
    (``1x1xHxW``) and, when ``grayscale`` is set, ``gray`` (``1x1xHxW``).
    """
    import cv2

    img = map_.image
    resized = cv2.resize(
        img, (grid.nx, grid.ny), interpolation=cv2.INTER_AREA
    )
    rgb = torch.from_numpy(resized).to(dtype=torch.float32).permute(2, 0, 1)
    rgb = rgb[None] / 255.0
    if normalize:
        rgb = (rgb - 0.5) / 0.5
    valid = torch.ones((1, 1, grid.ny, grid.nx), dtype=torch.float32)
    out = {"image": rgb, "valid": valid}
    if grayscale:
        gray = torch.from_numpy(
            cv2.cvtColor(resized, cv2.COLOR_RGB2GRAY)
        ).to(dtype=torch.float32)
        if normalize:
            gray = (gray - gray.mean()) / gray.std().clamp_min(1e-6)
        else:
            gray = gray / 255.0
        out["gray"] = gray[None, None]
    if device is not None:
        out = {k: v.to(device) for k, v in out.items()}
    return out


def resample_volume(
    volume: torch.Tensor,
    src_grid: PoseGrid,
    dst_grid: PoseGrid,
) -> torch.Tensor:
    """Resample a ``(Y,H,W)`` or ``(B,Y,H,W)`` volume onto another grid."""
    squeeze = volume.ndim == 3
    if squeeze:
        volume = volume[None]
    b, y, h, w = volume.shape
    flat = volume.reshape(b * y, 1, h, w)
    out = F.interpolate(
        flat, size=(dst_grid.ny, dst_grid.nx), mode="bilinear", align_corners=False
    )
    out = out.reshape(b, y, dst_grid.ny, dst_grid.nx)
    if src_grid.n_yaw != dst_grid.n_yaw:
        # Heading is circular; interpolate the angle axis explicitly.
        src_ang = np.linspace(-np.pi, np.pi, src_grid.n_yaw, endpoint=False)
        dst_ang = np.linspace(-np.pi, np.pi, dst_grid.n_yaw, endpoint=False)
        # Wrap-aware linear interpolation.
        out_np = out.detach().cpu().numpy()
        idx = (dst_ang - src_ang[0]) / src_grid.dyaw
        i0 = np.floor(idx).astype(int) % src_grid.n_yaw
        i1 = (i0 + 1) % src_grid.n_yaw
        t = (idx - np.floor(idx)).astype(np.float32)
        out_np = (
            out_np[:, i0] * (1.0 - t)[None, :, None, None]
            + out_np[:, i1] * t[None, :, None, None]
        )
        out = torch.from_numpy(out_np).to(volume.device)
    if squeeze:
        out = out[0]
    return out