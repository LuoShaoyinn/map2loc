"""Fixed bird's-eye-view map and its coordinate helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .types import Pose


@dataclass(frozen=True)
class Region:
    """Axis-aligned rectangle in map pixel coordinates."""

    x0: float
    y0: float
    x1: float
    y1: float

    @property
    def width(self) -> float:
        return self.x1 - self.x0

    @property
    def height(self) -> float:
        return self.y1 - self.y0

    @property
    def center(self) -> np.ndarray:
        return np.array([(self.x0 + self.x1) / 2.0, (self.y0 + self.y1) / 2.0])

    def shrink(self, frac: float) -> "Region":
        dx = self.width * frac
        dy = self.height * frac
        return Region(self.x0 + dx, self.y0 + dy, self.x1 - dx, self.y1 - dy)

    def contains(self, xy: np.ndarray, radius: float = 0.0) -> bool:
        x, y = float(xy[0]), float(xy[1])
        return (
            self.x0 + radius <= x <= self.x1 - radius
            and self.y0 + radius <= y <= self.y1 - radius
        )

    def as_tuple(self) -> tuple[float, float, float, float]:
        return (self.x0, self.y0, self.x1, self.y1)


class FixedMap:
    """A known, static map image.

    The map is loaded once and never modified.  Everything downstream samples
    from it.  Background colour is estimated from the image corners and is
    used to fill regions outside the map when a corrupted window pokes out.
    """

    def __init__(self, path: str | Path, *, background: np.ndarray | None = None):
        self.path = Path(path)
        bgr = cv2.imread(str(self.path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(f"could not read map image: {self.path}")
        self.image = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        self.height, self.width = self.image.shape[:2]
        self.background = (
            np.asarray(background, dtype=np.float32)
            if background is not None
            else self._estimate_background()
        )
        self._content_bbox: Region | None = None

    # ------------------------------------------------------------------ setup
    def _estimate_background(self) -> np.ndarray:
        h, w = self.image.shape[:2]
        k = max(2, min(h, w) // 40)
        patches = [
            self.image[:k, :k],
            self.image[:k, -k:],
            self.image[-k:, :k],
            self.image[-k:, -k:],
        ]
        pixels = np.concatenate([p.reshape(-1, 3) for p in patches], axis=0)
        return np.median(pixels, axis=0).astype(np.float32)

    # --------------------------------------------------------------- geometry
    @property
    def content_bbox(self) -> Region:
        """Bounding box of everything that differs from the background."""
        if self._content_bbox is None:
            diff = np.linalg.norm(
                self.image.astype(np.float32) - self.background[None, None, :],
                axis=-1,
            )
            mask = diff > 25.0
            ys, xs = np.nonzero(mask)
            if xs.size == 0:
                self._content_bbox = Region(0, 0, self.width, self.height)
            else:
                self._content_bbox = Region(
                    float(xs.min()),
                    float(ys.min()),
                    float(xs.max() + 1),
                    float(ys.max() + 1),
                )
        return self._content_bbox

    def region(self, kind: str = "content", shrink: float = 0.0) -> Region:
        if kind == "full":
            box = Region(0.0, 0.0, float(self.width), float(self.height))
        elif kind == "content":
            box = self.content_bbox
        else:
            raise ValueError(f"unknown region kind: {kind!r}")
        return box.shrink(shrink) if shrink else box

    # ------------------------------------------------------------ coordinates
    def normalize_xy(self, xy: np.ndarray) -> np.ndarray:
        """Map pixels -> [0, 1] using the image extent."""
        denom = np.array([self.width - 1, self.height - 1], dtype=np.float32)
        return np.asarray(xy, dtype=np.float32) / denom

    def denormalize_xy(self, xy: np.ndarray) -> np.ndarray:
        denom = np.array([self.width - 1, self.height - 1], dtype=np.float32)
        return np.asarray(xy, dtype=np.float32) * denom

    def clamp_pose(self, pose: Pose, region: Region, radius: float = 0.0) -> Pose:
        x = float(np.clip(pose.x, region.x0 + radius, region.x1 - radius))
        y = float(np.clip(pose.y, region.y0 + radius, region.y1 - radius))
        return Pose(x, y, pose.yaw)

    def random_pose(
        self,
        rng: np.random.Generator,
        region: Region,
        *,
        yaw: float | None = None,
    ) -> Pose:
        x = rng.uniform(region.x0, region.x1)
        y = rng.uniform(region.y0, region.y1)
        if yaw is None:
            yaw = rng.uniform(-np.pi, np.pi)
        return Pose(float(x), float(y), float(yaw))

    def in_bounds(self, xy: np.ndarray, margin: float = 0.0) -> bool:
        return (
            margin <= xy[0] <= self.width - margin
            and margin <= xy[1] <= self.height - margin
        )