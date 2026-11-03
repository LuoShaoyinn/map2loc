"""Inverse-perspective rectification of oblique camera views into BEV.

The observation model is a pinhole camera at a fixed height that looks
obliquely at the ground plane.  Its image is therefore a *perspective*
quadrilateral of map content, not a bird's-eye-view rectangle.  Every matcher
in this package compares a map-aligned BEV template against the global map, so
the observation has to be rectified first.

Because the scene is planar this is exact: for a known camera attitude
(``yaw, pitch, roll``), height and focal scale, the mapping from a BEV patch to
the camera image is a single 3x3 homography.  Camera ``x, y`` only translate
the patch, so the homography can be built for a camera at the origin and the
unknown position is recovered by template matching afterwards.

The patch is centred on the ground point seen at the image centre.  That point
is known relative to the camera from the attitude alone, so the rectified
template is independent of the unknown position.  After matching, the camera
position is ``patch_centre - centre_offset``.

The attitude inputs are noisy (default sigma 5 deg), so a small residual yaw
and a set of focal-scale hypotheses are searched at matching time.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from data_gen.camera import camera_basis
from data_gen.config import GeometryConfig


@dataclass(frozen=True)
class BEVSpec:
    """Geometry of the local rectified patch.

    ``patch_px`` is the template size in BEV pixels and ``map_px_per_px`` the
    number of map pixels covered by one BEV pixel.  The default 164 x 2 px
    spans 328 map pixels, i.e. 41 cells at the 8 px matching resolution.
    """

    patch_px: int = 164
    map_px_per_px: float = 2.0

    @property
    def span_px(self) -> float:
        return float(self.patch_px) * float(self.map_px_per_px)

    def to_dict(self) -> dict:
        return dict(self.__dict__)

    @classmethod
    def from_dict(cls, d: dict) -> "BEVSpec":
        return cls(**{k: d[k] for k in cls.__dataclass_fields__ if k in d})


# Fixed focal-scale hypotheses covering the generator's zoom range [0.70, 1.40].
SCALE_HYPOTHESES: tuple[float, ...] = tuple(
    float(v) for v in np.geomspace(0.71, 1.39, 7)
)

# Minimum |pitch| for the image-centre ray to hit the ground at a sane range.
_MIN_PITCH = np.deg2rad(3.0)


def centre_offset(yaw: float, pitch: float, height: float) -> np.ndarray:
    """Ground point seen at the image centre, relative to the camera.

    Returns ``(2,)`` map-pixel offset.  For a camera looking straight down the
    offset is zero; for an oblique camera it is along the heading.
    """
    if pitch > -_MIN_PITCH:
        return np.zeros(2, dtype=np.float32)
    d = -float(height) / np.tan(float(pitch))
    # Clamp shallow pitches to a finite but large range.
    d = float(np.clip(d, -8.0 * height, 8.0 * height))
    return np.array([d * np.cos(yaw), d * np.sin(yaw)], dtype=np.float32)


def patch_to_view_homography(
    yaw: float,
    pitch: float,
    roll: float,
    geometry: GeometryConfig,
    scale: float,
    spec: BEVSpec,
    *,
    height: float | None = None,
) -> np.ndarray:
    """Homography mapping BEV patch pixels to camera-view pixels.

    The camera is placed at the origin of the map frame; only the attitude,
    height and scale matter.  Use it with ``cv2.warpPerspective(...,
    WARP_INVERSE_MAP)`` to sample the view into a BEV patch.
    """
    h = float(height) if height is not None else float(geometry.height_px)
    right, up, forward = camera_basis(yaw, pitch, roll)
    focal = h * geometry.obs_width / geometry.footprint_px * float(scale)
    cx = (geometry.obs_width - 1) / 2.0
    cy = (geometry.obs_height - 1) / 2.0

    # Ground point (X, Y, 0) relative to a camera at (0, 0, h).
    linear = np.array(
        [
            [right[0], right[1], -right[2] * h],
            [-up[0], -up[1], up[2] * h],
            [forward[0], forward[1], -forward[2] * h],
        ],
        dtype=np.float64,
    )
    intrinsics = np.array(
        [[focal, 0.0, cx], [0.0, focal, cy], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    ground_to_view = intrinsics @ linear

    # Patch pixel -> ground offset (relative to the camera).
    off = centre_offset(yaw, pitch, h)
    centre = (spec.patch_px - 1) / 2.0
    patch_to_ground = np.array(
        [
            [spec.map_px_per_px, 0.0, off[0] - spec.map_px_per_px * centre],
            [0.0, spec.map_px_per_px, off[1] - spec.map_px_per_px * centre],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    return ground_to_view @ patch_to_ground


def rectify_numpy(
    image: np.ndarray,
    valid: np.ndarray | None,
    yaw: float,
    pitch: float,
    roll: float,
    geometry: GeometryConfig,
    scale: float,
    spec: BEVSpec,
) -> tuple[np.ndarray, np.ndarray]:
    """Rectify one camera view into a BEV patch (uint8 image, float mask)."""
    import cv2

    h_patch_to_view = patch_to_view_homography(
        yaw, pitch, roll, geometry, scale, spec
    )
    patch = cv2.warpPerspective(
        image,
        h_patch_to_view,
        (spec.patch_px, spec.patch_px),
        flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    ones = (
        np.ones(image.shape[:2], dtype=np.float32)
        if valid is None
        else np.asarray(valid, dtype=np.float32)
    )
    mask = cv2.warpPerspective(
        ones,
        h_patch_to_view,
        (spec.patch_px, spec.patch_px),
        flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    mask = mask * _front_mask_numpy(h_patch_to_view, spec)
    return patch, (mask > 0.5).astype(np.float32)


def _front_mask_numpy(h_patch_to_view: np.ndarray, spec: BEVSpec) -> np.ndarray:
    p = spec.patch_px
    us, vs = np.meshgrid(
        np.arange(p, dtype=np.float64), np.arange(p, dtype=np.float64)
    )
    pts = np.stack([us, vs, np.ones_like(us)], axis=-1).reshape(-1, 3)
    depth = (pts @ np.asarray(h_patch_to_view).T)[:, 2]
    return (depth > 0.0).astype(np.float32).reshape(p, p)


def rectify_torch(
    image: torch.Tensor,
    valid: torch.Tensor | None,
    yaw: torch.Tensor,
    pitch: torch.Tensor,
    roll: torch.Tensor,
    geometry: GeometryConfig,
    scale: torch.Tensor,
    spec: BEVSpec,
    *,
    height: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Batched BEV rectification.

    ``image`` is ``(B, 3, H, W)`` in ``[0, 1]``; ``valid`` is ``(B, H, W)``.
    ``yaw``/``pitch``/``roll``/``scale`` are ``(B,)``.  Returns
    ``(B, 3, P, P)`` patches and ``(B, P, P)`` validity masks.
    """
    if image.ndim != 4:
        raise ValueError("image must be (B,3,H,W)")
    b, _, obs_h, obs_w = image.shape
    dev = image.device
    dtype = image.dtype
    p = int(spec.patch_px)
    res = float(spec.map_px_per_px)
    h = float(height) if height is not None else float(geometry.height_px)

    cy, sy = torch.cos(yaw), torch.sin(yaw)
    cp, sp = torch.cos(pitch), torch.sin(pitch)
    forward = torch.stack([cp * cy, cp * sy, sp], dim=-1)
    up = torch.stack([-sp * cy, -sp * sy, cp], dim=-1)
    right = torch.cross(up, forward, dim=-1)
    cr, sr = torch.cos(roll), torch.sin(roll)
    right_r = cr[:, None] * right + sr[:, None] * up
    up_r = cr[:, None] * up - sr[:, None] * right
    right, up = right_r, up_r

    focal = h * obs_w / float(geometry.footprint_px) * scale
    cx = (obs_w - 1) / 2.0
    cy_px = (obs_h - 1) / 2.0

    linear = torch.zeros((b, 3, 3), device=dev, dtype=dtype)
    linear[:, 0, :2] = right[:, :2]
    linear[:, 0, 2] = -right[:, 2] * h
    linear[:, 1, :2] = -up[:, :2]
    linear[:, 1, 2] = up[:, 2] * h
    linear[:, 2, :2] = forward[:, :2]
    linear[:, 2, 2] = -forward[:, 2] * h
    intrinsics = torch.zeros((b, 3, 3), device=dev, dtype=dtype)
    intrinsics[:, 0, 0] = focal
    intrinsics[:, 1, 1] = focal
    intrinsics[:, 0, 2] = cx
    intrinsics[:, 1, 2] = cy_px
    intrinsics[:, 2, 2] = 1.0
    ground_to_view = intrinsics @ linear

    # Centre offset: -h / tan(pitch) along the heading.
    t = torch.clamp(
        -h / torch.tan(pitch.clamp(max=-_MIN_PITCH)),
        min=-8.0 * h,
        max=8.0 * h,
    )
    off = torch.stack([t * cy, t * sy], dim=-1)

    centre = (p - 1) / 2.0
    us = torch.arange(p, device=dev, dtype=dtype)
    gy, gx = torch.meshgrid(us, us, indexing="ij")
    q = torch.stack(
        [
            off[:, 0, None, None] + (gx[None] - centre) * res,
            off[:, 1, None, None] + (gy[None] - centre) * res,
            torch.ones((b, p, p), device=dev, dtype=dtype),
        ],
        dim=-1,
    )  # B,P,P,3
    uv = torch.einsum("bhwj,bij->bhwi", q, ground_to_view)
    depth = uv[..., 2]
    front = depth > 1e-6
    uv = uv[..., :2] / depth.clamp_min(1e-6)[..., None]

    gx_n = 2.0 * uv[..., 0] / max(obs_w - 1, 1) - 1.0
    gy_n = 2.0 * uv[..., 1] / max(obs_h - 1, 1) - 1.0
    sample = torch.stack([gx_n, gy_n], dim=-1)
    inside = (
        (uv[..., 0] >= 0)
        & (uv[..., 0] <= obs_w - 1)
        & (uv[..., 1] >= 0)
        & (uv[..., 1] <= obs_h - 1)
        & front
    )

    patch = F.grid_sample(
        image, sample, align_corners=True, padding_mode="zeros"
    )
    if valid is not None:
        v = valid[:, None].to(dtype)
        mask = F.grid_sample(
            v, sample, align_corners=True, padding_mode="zeros"
        )[:, 0]
    else:
        mask = torch.ones((b, p, p), device=dev, dtype=dtype)
    mask = mask * inside.to(dtype)
    return patch, mask


def camera_from_patch(
    xy: np.ndarray | torch.Tensor,
    yaw,
    pitch,
    height: float,
) -> np.ndarray | torch.Tensor:
    """Convert a patch-centre position to the camera ground position."""
    if isinstance(xy, torch.Tensor):
        t = torch.clamp(
            -height / torch.tan(pitch.clamp(max=-_MIN_PITCH)),
            min=-8.0 * height,
            max=8.0 * height,
        )
        off = torch.stack([t * torch.cos(yaw), t * torch.sin(yaw)], dim=-1)
        return xy - off
    off = centre_offset(float(yaw), float(pitch), float(height))
    return np.asarray(xy) - off