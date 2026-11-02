"""Camera augmentation and the sensor-noise model.

Geometry is a real pinhole camera (see :mod:`data_gen.camera`).  The
per-frame geometric augmentation is a random zoom (equivalent to a camera
height change) and random head/gimbal rotations.

The photometric model combines per-view Gaussian sensor noise, independent
per-pixel Gaussian noise, random brightness scaling, random scratches, and a
configurable fraction of pixels that are marked invalid in the validity mask.
"""

from __future__ import annotations

import cv2
import numpy as np

from .config import GeometryConfig, PhotometricConfig
from .types import Augmentation


def sample_augmentation(
    rng: np.random.Generator,
    cfg: GeometryConfig,
    *,
    enabled: bool = True,
) -> Augmentation:
    """Sample a random zoom and small head rotations around zero."""
    if not enabled:
        return Augmentation()

    def _symmetric(a: float) -> float:
        return float(np.deg2rad(rng.uniform(-a, a))) if a > 0 else 0.0

    return Augmentation(
        scale=float(rng.uniform(*cfg.scale)),
        head_pitch=_symmetric(cfg.head_pitch_deg),
        head_yaw=_symmetric(cfg.head_yaw_deg),
        head_roll=_symmetric(cfg.head_roll_deg),
    )


def _draw_scratches(
    image: np.ndarray,
    rng: np.random.Generator,
    cfg: PhotometricConfig,
) -> tuple[int, np.ndarray]:
    """Draw random thin scratches and return ``(count, mask)``.

    Each scratch is a straight line with a random orientation, length,
    width and brightness delta.  The mask is 1 on scratch pixels.
    """
    h, w = image.shape[:2]
    mask = np.zeros((h, w), dtype=np.float32)
    if cfg.scratch_prob <= 0.0 or rng.random() >= cfg.scratch_prob:
        return 0, mask

    lo, hi = (int(v) for v in cfg.scratch_count)
    count = int(rng.integers(lo, hi + 1))
    diag = float(np.hypot(h, w))
    for _ in range(count):
        centre = np.array([rng.uniform(0, w), rng.uniform(0, h)])
        angle = rng.uniform(0.0, np.pi)
        half = 0.5 * rng.uniform(0.5, 1.5) * diag
        direction = np.array([np.cos(angle), np.sin(angle)]) * half
        p0 = tuple(np.round(centre - direction).astype(int))
        p1 = tuple(np.round(centre + direction).astype(int))
        width = int(rng.integers(int(cfg.scratch_width[0]), int(cfg.scratch_width[1]) + 1))
        delta = float(rng.uniform(*cfg.scratch_delta)) * float(rng.choice((-1.0, 1.0)))
        cv2.line(
            image,
            p0,
            p1,
            (delta, delta, delta),
            width,
            cv2.LINE_AA,
        )
        cv2.line(mask, p0, p1, 1.0, width, cv2.LINE_AA)
    return count, mask


def corrupt(
    image: np.ndarray,
    rng: np.random.Generator,
    cfg: PhotometricConfig,
    *,
    background: np.ndarray | None = None,
    return_mask: bool = False,
):
    """Apply photometric corruption and the invalid-pixel mask.

    Parameters
    ----------
    image:
        RGB ``uint8`` image.
    rng:
        Random generator.
    cfg:
        Noise configuration.
    background:
        Unused; kept so callers can share the old signature.
    return_mask:
        When true, also return a ``(H, W)`` float32 mask where 1 marks
        trustworthy map content and 0 marks invalid pixels.

    Returns
    -------
    image, info[, valid]
    """
    del background  # only kept for API compatibility

    if not cfg.enabled:
        if return_mask:
            return (
                image.copy(),
                {"enabled": False},
                np.ones(image.shape[:2], dtype=np.float32),
            )
        return image.copy(), {"enabled": False}

    out = image.astype(np.float32)
    valid = np.ones(image.shape[:2], dtype=np.float32)

    # Random per-view brightness scaling.
    intensity_scale = float(rng.uniform(*cfg.intensity_scale))
    intensity_shift = float(rng.uniform(*cfg.intensity_shift))
    out = out * intensity_scale + intensity_shift

    # One Gaussian sigma per view, applied independently to every pixel.
    sigma = float(rng.uniform(*cfg.gauss_sigma))
    if sigma > 1e-6:
        out = out + rng.normal(0.0, sigma, size=out.shape).astype(np.float32)

    # Extra Gaussian noise with an independent sigma for every pixel/channel.
    pixel_sigma = rng.uniform(
        *cfg.pixel_gauss_sigma, size=out.shape
    ).astype(np.float32)
    if float(pixel_sigma.max()) > 1e-6:
        out = out + pixel_sigma * rng.normal(
            size=out.shape
        ).astype(np.float32)

    # Random scratches hide the map; optionally mark them invalid too.
    n_scratch, scratch_mask = _draw_scratches(out, rng, cfg)
    scratch_pixels = scratch_mask > 0.01
    if cfg.scratch_invalid:
        valid[scratch_pixels] = 0.0

    # Random invalid pixels.  Their rendered value is kept: the mask says
    # they cannot be trusted, even if they look plausible.
    invalid_frac = float(np.clip(cfg.invalid_pixel_rate, 0.0, 1.0))
    n_random = 0
    if invalid_frac > 1e-9:
        invalid = rng.random(image.shape[:2]) < invalid_frac
        n_random = int(invalid.sum())
        valid[invalid] = 0.0

    info = {
        "enabled": True,
        "gauss_sigma": sigma,
        "pixel_gauss_sigma": float(pixel_sigma.mean()),
        "invalid_frac": invalid_frac,
        "n_invalid": int((valid < 0.5).sum()),
        "n_random_invalid": n_random,
        "n_scratch": n_scratch,
        "n_scratch_invalid": int(scratch_pixels.sum()),
        "intensity_scale": intensity_scale,
        "intensity_shift": intensity_shift,
        "valid_frac": float(valid.mean()),
    }
    result = np.clip(out, 0.0, 255.0).astype(np.uint8)
    if return_mask:
        return result, info, valid
    return result, info


def overlay_invalid(image: np.ndarray, valid: np.ndarray, colour=(255, 0, 0)) -> np.ndarray:
    """Tint invalid pixels so they are visible in previews."""
    out = image.copy()
    invalid = np.asarray(valid) < 0.5
    if invalid.any():
        out[invalid] = (
            0.35 * out[invalid].astype(np.float32)
            + 0.65 * np.asarray(colour, dtype=np.float32)
        ).astype(np.uint8)
    return out


def draw_mask(mask: np.ndarray) -> np.ndarray:
    """Convert a ``[0, 1]`` validity mask into a viewable RGB image."""
    m = np.clip(np.asarray(mask, dtype=np.float32), 0.0, 1.0)
    g = (m * 255.0).astype(np.uint8)
    return cv2.cvtColor(g, cv2.COLOR_GRAY2RGB)