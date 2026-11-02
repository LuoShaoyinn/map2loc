"""Camera augmentation and the sensor-noise model.

Geometry is now a real pinhole camera (see :mod:`data_gen.camera`).  The only
per-frame geometric augmentation is a small zoom (equivalent to a camera
height change) and tiny head/gimbal rotations around zero.

The noise model is intentionally simple and matches the task statement:
independent Gaussian noise over the whole view, plus a random subset of pixels
that are marked invalid in the validity mask.
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


def corrupt(
    image: np.ndarray,
    rng: np.random.Generator,
    cfg: PhotometricConfig,
    *,
    background: np.ndarray | None = None,
    return_mask: bool = False,
):
    """Add Gaussian noise and randomly invalidate pixels.

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
        trustworthy map content and 0 marks randomly invalid pixels.

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

    # One sigma per view, applied independently to every pixel/channel.
    sigma = float(rng.uniform(*cfg.gauss_sigma))
    if sigma > 1e-6:
        out = out + rng.normal(0.0, sigma, size=out.shape).astype(np.float32)

    # Random invalid pixels.  Their rendered value is kept: the mask says
    # they cannot be trusted, even if they look plausible.
    invalid_frac = float(rng.uniform(*cfg.random_invalid_frac))
    n_invalid = 0
    if invalid_frac > 1e-9:
        invalid = rng.random(image.shape[:2]) < invalid_frac
        n_invalid = int(invalid.sum())
        valid[invalid] = 0.0

    info = {
        "enabled": True,
        "gauss_sigma": sigma,
        "invalid_frac": invalid_frac,
        "n_invalid": n_invalid,
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