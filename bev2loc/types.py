"""Small data containers shared across the package."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class Pose:
    """Camera pose in map pixel coordinates. yaw in radians, 0 = +x."""

    x: float
    y: float
    yaw: float

    @property
    def xy(self) -> np.ndarray:
        return np.array([self.x, self.y], dtype=np.float32)

    def as_array(self) -> np.ndarray:
        return np.array([self.x, self.y, self.yaw], dtype=np.float32)


@dataclass
class Augmentation:
    """Sampled geometric corruption parameters (kept for replay/analysis)."""

    scale_x: float = 1.0
    scale_y: float = 1.0
    shear_x: float = 0.0
    shear_y: float = 0.0
    rot_deg: float = 0.0
    tx: float = 0.0
    ty: float = 0.0
    persp_pts: np.ndarray | None = None  # (4, 2) offsets, augmented->view

    def as_dict(self) -> dict[str, Any]:
        return {
            "scale_x": self.scale_x,
            "scale_y": self.scale_y,
            "shear_x": self.shear_x,
            "shear_y": self.shear_y,
            "rot_deg": self.rot_deg,
            "tx": self.tx,
            "ty": self.ty,
            "persp": None if self.persp_pts is None else self.persp_pts.tolist(),
        }


@dataclass
class Observation:
    """One rendered observation."""

    image: np.ndarray  # uint8 RGB, (H, W, 3)
    pose: Pose  # camera pose used to cut the window
    gt_xy: np.ndarray  # (2,) map pixel the *centre* of the obs maps to
    corners: np.ndarray  # (4, 2) obs corners in map pixels
    warp: np.ndarray  # (3, 3) obs pixel -> map pixel homography
    aug: Augmentation = field(default_factory=Augmentation)
    photo: dict[str, Any] = field(default_factory=dict)
    # (H, W) float mask, 1 = trustworthy map content, 0 = unreliable
    # (out of map, scratch / smudge / occlusion / saturated outlier).
    valid_mask: np.ndarray | None = None

    @property
    def gt_position(self) -> np.ndarray:
        """Centre-of-observation target (= pose when no translation jitter)."""
        return self.gt_xy

    def as_dict(self) -> dict[str, Any]:
        return {
            "image": self.image,
            "pose": self.pose.as_array(),
            "gt_xy": self.gt_xy,
            "corners": self.corners,
            "aug": self.aug.as_dict(),
            "photo": self.photo,
            "valid_frac": None
            if self.valid_mask is None
            else float(np.mean(self.valid_mask)),
        }


@dataclass
class Trajectory:
    """A smooth sequence of camera poses."""

    poses: np.ndarray  # (N, 3): x, y, yaw
    region: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)

    def __len__(self) -> int:
        return int(self.poses.shape[0])

    def __getitem__(self, idx: int | slice) -> Pose | list[Pose]:
        if isinstance(idx, slice):
            return [self[i] for i in range(*idx.indices(len(self)))]
        x, y, yaw = self.poses[idx]
        return Pose(float(x), float(y), float(yaw))