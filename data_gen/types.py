"""Small data containers shared across the package."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class Pose:
    """Camera pose in map pixel coordinates.

    Attributes
    ----------
    x, y:
        Camera position on the ground plane, in map pixels.
    yaw:
        Heading in radians, measured from ``+x`` (positive turns are
        clockwise on screen because ``y`` points down).
    pitch:
        Tilt in radians measured from the **horizontal** plane.  ``0`` looks
        straight ahead along the heading, ``-pi/2`` looks straight down at the
        ground (nadir), and negative values tilt towards the ground.
    roll:
        Rotation about the optical axis, in radians.
    height:
        Camera height above the ground plane in map pixels.  ``0`` means
        "use the generator's fixed height".
    """

    x: float
    y: float
    yaw: float
    pitch: float = 0.0
    roll: float = 0.0
    height: float = 0.0

    @property
    def xy(self) -> np.ndarray:
        return np.array([self.x, self.y], dtype=np.float32)

    def as_array(self) -> np.ndarray:
        """``(x, y, yaw, pitch, roll)`` as float32."""
        return np.array(
            [self.x, self.y, self.yaw, self.pitch, self.roll],
            dtype=np.float32,
        )

    def as_array6(self) -> np.ndarray:
        """``(x, y, yaw, pitch, roll, height)`` as float32."""
        return np.array(
            [self.x, self.y, self.yaw, self.pitch, self.roll, self.height],
            dtype=np.float32,
        )


@dataclass
class Augmentation:
    """Per-frame camera augmentation (kept for replay / analysis)."""

    # Random zoom; observationally equivalent to a camera-height change.
    scale: float = 1.0
    # Small head/gimbal rotations added to the smooth camera pose (radians).
    head_pitch: float = 0.0
    head_yaw: float = 0.0
    head_roll: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "scale": self.scale,
            "head_pitch_deg": float(np.rad2deg(self.head_pitch)),
            "head_yaw_deg": float(np.rad2deg(self.head_yaw)),
            "head_roll_deg": float(np.rad2deg(self.head_roll)),
        }


@dataclass
class Observation:
    """One rendered camera view of the map."""

    image: np.ndarray  # uint8 RGB, (H, W, 3)
    pose: Pose  # effective camera pose (smooth pose + head jitter)
    gt_xy: np.ndarray  # (2,) camera ground position in map pixels
    center_xy: np.ndarray  # (2,) map pixel seen at the image centre
    corners: np.ndarray  # (4, 2) view corners in map pixels
    warp: np.ndarray  # (3, 3) view pixel -> map pixel homography
    # Noisy attitude inputs available to the model: ground truth + N(0, s).
    yaw_input: float = 0.0
    pitch_input: float = 0.0
    aug: Augmentation = field(default_factory=Augmentation)
    photo: dict[str, Any] = field(default_factory=dict)
    # (H, W) float mask, 1 = trustworthy map content, 0 = unreliable
    # (out of map, or randomly invalidated pixel).
    valid_mask: np.ndarray | None = None

    @property
    def gt_position(self) -> np.ndarray:
        """Localization target: the camera's ground position."""
        return self.gt_xy

    def as_dict(self) -> dict[str, Any]:
        return {
            "image": self.image,
            "pose": self.pose.as_array(),
            "gt_xy": self.gt_xy,
            "center_xy": self.center_xy,
            "corners": self.corners,
            "yaw_input": self.yaw_input,
            "pitch_input": self.pitch_input,
            "aug": self.aug.as_dict(),
            "photo": self.photo,
            "valid_frac": None
            if self.valid_mask is None
            else float(np.mean(self.valid_mask)),
        }


@dataclass
class Trajectory:
    """A smooth sequence of camera poses.

    ``poses`` has columns ``(x, y, yaw, pitch, roll)``.
    """

    poses: np.ndarray
    region: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)

    def __len__(self) -> int:
        return int(self.poses.shape[0])

    def __getitem__(self, idx: int | slice) -> Pose | list[Pose]:
        if isinstance(idx, slice):
            return [self[i] for i in range(*idx.indices(len(self)))]
        x, y, yaw, pitch, roll = self.poses[idx]
        return Pose(float(x), float(y), float(yaw), float(pitch), float(roll))