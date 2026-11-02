"""Pinhole camera model for the ground-plane map.

The map is a textured ground plane (``z = 0``).  A camera with a fixed
height looks down at it, optionally pitched/rolled a little.  Because the
scene is planar, the full projection reduces to a 3x3 homography, so the
"renderer" is an exact inverse warp (the same operation a perspective
renderer would perform for the plane).

Conventions
-----------
* World: ``x`` right, ``y`` down (map-image axes), ``z`` up.
* Camera pose: position ``(x, y, height)`` plus ``yaw``, ``pitch``, ``roll``.
* ``pitch`` is measured from the **horizontal** plane: ``0`` looks straight
  ahead, ``-90`` looks straight down (nadir), and negative values tilt the
  optical axis towards the ground.
* At ``yaw = 0`` the heading is ``+x``; positive yaw turns clockwise on
  screen.  Positive ``roll`` rotates the image about the optical axis.
"""

from __future__ import annotations

from typing import Iterable

import cv2
import numpy as np

from .config import GeometryConfig
from .types import Pose


def wrap_angle(angle: float) -> float:
    return float((angle + np.pi) % (2.0 * np.pi) - np.pi)


def apply_homography(h: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Apply a 3x3 homography to ``(N, 2)`` points."""
    pts = np.asarray(pts, dtype=np.float64)
    ones = np.ones((pts.shape[0], 1), dtype=np.float64)
    hom = np.concatenate([pts, ones], axis=1) @ np.asarray(h).T
    return (hom[:, :2] / hom[:, 2:3]).astype(np.float32)


# Backwards-compatible private alias used by older callers.
_apply_h = apply_homography


def camera_basis(
    yaw: float, pitch: float, roll: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return world-space ``(right, up, forward)`` camera axes.

    ``right`` points to increasing image columns, ``up`` to decreasing image
    rows, and ``forward`` along the optical axis.
    """
    cy, sy = np.cos(yaw), np.sin(yaw)
    cp, sp = np.cos(pitch), np.sin(pitch)

    # Pitch is measured from horizontal (0 = ahead, -90 deg = straight down).
    forward = np.array([cp * cy, cp * sy, sp], dtype=np.float64)
    up = np.array([-sp * cy, -sp * sy, cp], dtype=np.float64)
    right = np.cross(up, forward)

    if abs(roll) > 1e-12:
        cr, sr = np.cos(roll), np.sin(roll)
        right, up = cr * right + sr * up, cr * up - sr * right
    return right, up, forward


def focal_pixels(
    geometry: GeometryConfig,
    height: float,
    scale: float = 1.0,
) -> float:
    """Focal length that gives ``footprint_px`` ground width at nadir."""
    return float(height) * float(geometry.obs_width) / float(
        geometry.footprint_px
    ) * float(scale)


def map_to_view(
    pose: Pose,
    geometry: GeometryConfig,
    scale: float = 1.0,
) -> np.ndarray:
    """Homography mapping ground-plane map pixels to view pixels."""
    height = float(pose.height) if pose.height > 0 else float(geometry.height_px)
    right, up, forward = camera_basis(pose.yaw, pose.pitch, pose.roll)
    centre = np.array([pose.x, pose.y, height], dtype=np.float64)
    focal = focal_pixels(geometry, height, scale)

    # Camera coordinates of a ground point (X, Y, 0):
    #   x_c = right . (P - C)
    #   y_c = down . (P - C) = -up . (P - C)
    #   z_c = forward . (P - C)
    linear = np.array(
        [
            [right[0], right[1], -float(right @ centre)],
            [-up[0], -up[1], float(up @ centre)],
            [forward[0], forward[1], -float(forward @ centre)],
        ],
        dtype=np.float64,
    )
    cx = (geometry.obs_width - 1) / 2.0
    cy = (geometry.obs_height - 1) / 2.0
    intrinsics = np.array(
        [[focal, 0.0, cx], [0.0, focal, cy], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    return intrinsics @ linear


def view_to_map(
    pose: Pose,
    geometry: GeometryConfig,
    scale: float = 1.0,
) -> np.ndarray:
    """Homography mapping view pixels to ground-plane map pixels."""
    return np.linalg.inv(map_to_view(pose, geometry, scale))


def view_border_corners(geometry: GeometryConfig) -> np.ndarray:
    w, h = geometry.obs_width, geometry.obs_height
    return np.array(
        [[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32
    )


def _border_spec(geometry: GeometryConfig, background: Iterable[float]):
    if geometry.border == "background":
        return cv2.BORDER_CONSTANT, tuple(int(round(v)) for v in background)
    if geometry.border == "zero":
        return cv2.BORDER_CONSTANT, (0, 0, 0)
    if geometry.border == "reflect":
        return cv2.BORDER_REFLECT_101, tuple(int(round(v)) for v in background)
    raise ValueError(f"unknown border mode: {geometry.border!r}")


def front_mask(
    view_to_map_h: np.ndarray,
    geometry: GeometryConfig,
) -> np.ndarray:
    """1 where a view ray intersects the ground plane *in front* of the camera.

    With an oblique camera the horizon can enter the image.  Rays above the
    horizon do not hit the ground plane at all; the projective inverse maps
    them to mirrored / behind-camera points, which must not be rendered.
    """
    w, h = geometry.obs_width, geometry.obs_height
    us, vs = np.meshgrid(
        np.arange(w, dtype=np.float64), np.arange(h, dtype=np.float64)
    )
    pts = np.stack([us, vs, np.ones_like(us)], axis=-1).reshape(-1, 3)
    depth = (pts @ np.asarray(view_to_map_h, dtype=np.float64).T)[:, 2]
    return (depth > 0.0).astype(np.float32).reshape(h, w)


def render_view(
    map_image: np.ndarray,
    view_to_map_h: np.ndarray,
    geometry: GeometryConfig,
    background: Iterable[float],
) -> np.ndarray:
    """Warp the ground-plane map into the camera view."""
    mode, value = _border_spec(geometry, background)
    image = cv2.warpPerspective(
        map_image,
        view_to_map_h,
        (geometry.obs_width, geometry.obs_height),
        flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
        borderMode=mode,
        borderValue=value,
    )
    front = front_mask(view_to_map_h, geometry)
    if not bool(front.all()):
        image[front < 0.5] = tuple(int(round(v)) for v in background)
    return image


def render_validity(
    map_image: np.ndarray,
    view_to_map_h: np.ndarray,
    geometry: GeometryConfig,
) -> np.ndarray:
    """1 where a view pixel samples the known map, 0 outside it."""
    ones = np.ones(map_image.shape[:2], dtype=np.uint8)
    valid = cv2.warpPerspective(
        ones,
        view_to_map_h,
        (geometry.obs_width, geometry.obs_height),
        flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    front = front_mask(view_to_map_h, geometry)
    return valid.astype(np.float32) * front