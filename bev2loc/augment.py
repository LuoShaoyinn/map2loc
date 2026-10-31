"""Geometric and photometric corruptions for observation windows.

Geometry is expressed as a single 3x3 homography mapping augmented
observation pixels back to the un-corrupted observation frame.  All
geometric operations (scale, aspect, shear, rotation, translation,
perspective) are applied around the observation centre so the centre target
stays stable; a small random translation is the only thing that shifts it and
the exact ground truth is always recomputed from the homography.
"""

from __future__ import annotations

import cv2
import numpy as np

from .config import GeometryConfig, PhotometricConfig
from .types import Augmentation


# --------------------------------------------------------------------------
# geometric sampling
# --------------------------------------------------------------------------
def sample_augmentation(
    rng: np.random.Generator,
    cfg: GeometryConfig,
    width: int,
    height: int,
) -> tuple[Augmentation, np.ndarray]:
    """Sample geometry parameters and their augmented->view homography."""
    zmin, zmax = cfg.zoom
    zoom = float(np.exp(rng.uniform(np.log(zmin), np.log(zmax))))
    amin, amax = cfg.aspect
    aspect = float(rng.uniform(amin, amax))

    aug = Augmentation(
        scale_x=zoom * aspect,
        scale_y=zoom / aspect,
        shear_x=float(rng.uniform(-cfg.shear, cfg.shear)),
        shear_y=float(rng.uniform(-cfg.shear, cfg.shear)),
        rot_deg=float(rng.uniform(-cfg.rot_deg, cfg.rot_deg)),
        tx=float(rng.uniform(-cfg.translate_px, cfg.translate_px)),
        ty=float(rng.uniform(-cfg.translate_px, cfg.translate_px)),
    )

    if rng.random() < cfg.persp_prob:
        s = cfg.persp_strength * min(width, height)
        aug.persp_pts = rng.uniform(-s, s, size=(4, 2)).astype(np.float32)

    return aug, build_geometry_matrix(aug, width, height)


def _translate(tx: float, ty: float) -> np.ndarray:
    m = np.eye(3, dtype=np.float64)
    m[0, 2] = tx
    m[1, 2] = ty
    return m


def _scale(sx: float, sy: float) -> np.ndarray:
    m = np.eye(3, dtype=np.float64)
    m[0, 0] = sx
    m[1, 1] = sy
    return m


def _shear(shx: float, shy: float) -> np.ndarray:
    m = np.eye(3, dtype=np.float64)
    m[0, 1] = shx
    m[1, 0] = shy
    return m


def _rotate(deg: float) -> np.ndarray:
    a = np.deg2rad(deg)
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)


def build_geometry_matrix(
    aug: Augmentation, width: int, height: int
) -> np.ndarray:
    """Return H mapping augmented observation pixels -> view pixels.

    The view frame is the nominal crop for the camera pose.  The centre of the
    augmented observation maps to ``centre + (tx, ty)`` in the view frame.
    """
    cx = (width - 1) / 2.0
    cy = (height - 1) / 2.0
    # scale + shear + rotation, all about the centre
    k = _rotate(aug.rot_deg) @ _scale(aug.scale_x, aug.scale_y) @ _shear(
        aug.shear_x, aug.shear_y
    )
    a = _translate(cx + aug.tx, cy + aug.ty) @ k @ _translate(-cx, -cy)

    if aug.persp_pts is None:
        return a

    dst = np.array(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
        dtype=np.float32,
    )
    src = dst + aug.persp_pts.astype(np.float32)
    p = cv2.getPerspectiveTransform(dst, src).astype(np.float64)
    return a @ p


# --------------------------------------------------------------------------
# photometric corruption
# --------------------------------------------------------------------------
def _gaussian_blur(img: np.ndarray, rng: np.random.Generator, cfg: PhotometricConfig) -> tuple[np.ndarray, int]:
    k = int(rng.integers(3, max(4, cfg.blur_max + 1)))
    k = k if k % 2 == 1 else k + 1
    if rng.random() < 0.4:  # motion blur
        kernel = np.zeros((k, k), dtype=np.float32)
        kernel[k // 2, :] = 1.0
        ang = float(rng.uniform(0, 180))
        rot = cv2.getRotationMatrix2D((k / 2 - 0.5, k / 2 - 0.5), ang, 1.0)
        kernel = cv2.warpAffine(kernel, rot, (k, k))
        s = kernel.sum()
        if s > 1e-6:
            kernel /= s
        return cv2.filter2D(img, -1, kernel), k
    return cv2.GaussianBlur(img, (k, k), 0), k


def _illumination(img: np.ndarray, rng: np.random.Generator, cfg: PhotometricConfig) -> tuple[np.ndarray, dict]:
    h, w = img.shape[:2]
    gain = float(rng.uniform(*cfg.gain))
    gamma = float(rng.uniform(*cfg.gamma))
    tint = 1.0 + rng.uniform(-cfg.tint, cfg.tint, size=3).astype(np.float32)
    vig = float(rng.uniform(*cfg.vignette))
    grad = float(rng.uniform(*cfg.gradient))

    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    xx = (xx - w / 2) / (w / 2)
    yy = (yy - h / 2) / (h / 2)
    r2 = (xx * xx + yy * yy) / 2.0
    field = 1.0 - vig * r2

    if grad > 1e-6:
        ang = float(rng.uniform(0, 2 * np.pi))
        gx, gy = np.cos(ang), np.sin(ang)
        ramp = (gx * xx + gy * yy) / 2.0  # roughly [-1, 1]
        field = field * (1.0 + grad * ramp)

    out = img.astype(np.float32) * gain
    out *= tint[None, None, :]
    out *= field[..., None]
    out = np.clip(out, 0.0, 255.0)
    if abs(gamma - 1.0) > 1e-3:
        out = 255.0 * np.power(out / 255.0, gamma)
    return np.clip(out, 0.0, 255.0), {
        "gain": gain,
        "gamma": gamma,
        "vignette": vig,
        "gradient": grad,
    }


def _noise(
    img: np.ndarray, rng: np.random.Generator, cfg: PhotometricConfig
) -> tuple[np.ndarray, dict, np.ndarray]:
    sigma = float(rng.uniform(*cfg.gauss_sigma))
    out = img
    if sigma > 1e-3:
        out = out + rng.normal(0.0, sigma, size=out.shape).astype(np.float32)

    valid = np.ones(out.shape[:2], dtype=np.float32)
    sp = float(rng.uniform(*cfg.salt_pepper))
    if sp > 1e-6:
        mask = rng.random(out.shape[:2]) < sp
        # half pepper, half salt
        salt = mask & (rng.random(out.shape[:2]) < 0.5)
        pepper = mask & ~salt
        out[salt] = 255.0
        out[pepper] = 0.0
        valid[mask] = 0.0

    return np.clip(out, 0.0, 255.0), {"gauss_sigma": sigma, "salt_pepper": sp}, valid


def _scratches(
    img: np.ndarray,
    rng: np.random.Generator,
    cfg: PhotometricConfig,
    background: np.ndarray | None,
) -> tuple[np.ndarray, dict, np.ndarray]:
    h, w = img.shape[:2]
    diag = float(np.hypot(w, h))
    # OpenCV's drawing routines work on uint8, so paint there and blend later.
    overlay = np.zeros((h, w, 3), dtype=np.uint8)
    alpha = np.zeros((h, w), dtype=np.uint8)
    valid = np.ones((h, w), dtype=np.float32)

    n_scratch = int(rng.integers(cfg.scratch_count[0], cfg.scratch_count[1] + 1))
    bg = background if background is not None else np.array([60, 120, 60], dtype=np.float32)

    for _ in range(n_scratch):
        # colour: bright lens scratch, dark scratch, or pale background-ish
        u = rng.random()
        if u < 0.65:
            color = rng.uniform(190, 255, size=3)
        elif u < 0.85:
            color = rng.uniform(0, 45, size=3)
        else:
            color = np.clip(bg * rng.uniform(1.0, 1.4) + rng.uniform(0, 40), 0, 255)
        color_u8 = tuple(int(v) for v in color)

        # start anywhere, travel in a random (mostly long) direction
        p = np.array([rng.uniform(0, w), rng.uniform(0, h)], dtype=np.float32)
        ang = float(rng.uniform(0, 2 * np.pi))
        length = float(rng.uniform(*cfg.scratch_length)) * diag
        n_seg = int(rng.integers(1, 5))
        pts = [p.copy()]
        for _seg in range(n_seg):
            ang += float(rng.normal(0.0, 0.18))
            step = length / n_seg
            p = p + step * np.array([np.cos(ang), np.sin(ang)], dtype=np.float32)
            pts.append(p.copy())
        pts_arr = np.round(np.array(pts)).astype(np.int32).reshape(-1, 1, 2)

        width = int(rng.integers(cfg.scratch_width[0], cfg.scratch_width[1] + 1))
        a = int(round(float(rng.uniform(*cfg.scratch_alpha)) * 255))
        cv2.polylines(overlay, [pts_arr], False, color_u8, width, cv2.LINE_AA)
        cv2.polylines(alpha, [pts_arr], False, a, width, cv2.LINE_AA)

    n_dust = int(rng.integers(cfg.dust_count[0], cfg.dust_count[1] + 1))
    for _ in range(n_dust):
        p = (int(rng.integers(0, w)), int(rng.integers(0, h)))
        r = int(rng.integers(1, 3))
        color = rng.uniform(0, 80, size=3) if rng.random() < 0.5 else rng.uniform(180, 255, size=3)
        a = int(round(float(rng.uniform(0.4, 1.0)) * 255))
        cv2.circle(overlay, p, r, tuple(int(v) for v in color), -1, cv2.LINE_AA)
        cv2.circle(alpha, p, r, a, -1, cv2.LINE_AA)

    out = img * (1.0 - alpha[..., None] / 255.0) + overlay.astype(np.float32) * (
        alpha[..., None] / 255.0
    )
    valid[alpha > 20] = 0.0

    # smudges: soft dark/light blobs, e.g. fingerprints on the lens
    n_smudge = 0
    if rng.random() < cfg.smudge_prob:
        n_smudge = int(rng.integers(1, 3))
        sm_overlay = np.zeros_like(overlay)
        sm_alpha = np.zeros_like(alpha)
        for _ in range(n_smudge):
            center = (int(rng.integers(0, w)), int(rng.integers(0, h)))
            axes = (int(rng.integers(w // 12, w // 4)), int(rng.integers(h // 16, h // 4)))
            ang = float(rng.uniform(0, 180))
            color = rng.uniform(0, 90, size=3) if rng.random() < 0.6 else rng.uniform(160, 240, size=3)
            a = int(round(float(rng.uniform(0.15, 0.45)) * 255))
            cv2.ellipse(sm_overlay, center, axes, ang, 0, 360, tuple(int(v) for v in color), -1)
            cv2.ellipse(sm_alpha, center, axes, ang, 0, 360, a, -1)
        sigma = max(w, h) / 40.0
        sm_overlay_f = cv2.GaussianBlur(sm_overlay, (0, 0), sigmaX=sigma).astype(np.float32) / 255.0
        sm_alpha_f = cv2.GaussianBlur(sm_alpha, (0, 0), sigmaX=sigma).astype(np.float32)[..., None] / 255.0
        out = out * (1.0 - sm_alpha_f) + sm_overlay_f * 255.0 * sm_alpha_f
        valid[sm_alpha_f[..., 0] > 0.05] = 0.0

    return np.clip(out, 0.0, 255.0), {
        "n_scratch": n_scratch,
        "n_dust": n_dust,
        "n_smudge": n_smudge,
    }, valid


def _occlusions(
    img: np.ndarray,
    rng: np.random.Generator,
    cfg: PhotometricConfig,
) -> tuple[np.ndarray, dict, np.ndarray]:
    """Paint opaque low-texture blobs and mark them unreliable."""
    if rng.random() >= cfg.occlusion_prob:
        return img, {"n_occlusion": 0}, np.ones(img.shape[:2], dtype=np.float32)

    h, w = img.shape[:2]
    out = img.astype(np.float32)
    valid = np.ones((h, w), dtype=np.float32)
    n = int(rng.integers(cfg.occlusion_count[0], cfg.occlusion_count[1] + 1))
    for _ in range(n):
        f = float(rng.uniform(*cfg.occlusion_frac))
        ow = max(4, int(round(w * np.sqrt(f) * rng.uniform(0.6, 1.4))))
        oh = max(4, int(round(h * np.sqrt(f) * rng.uniform(0.6, 1.4))))
        ow, oh = min(ow, w), min(oh, h)
        x0 = int(rng.integers(0, max(1, w - ow)))
        y0 = int(rng.integers(0, max(1, h - oh)))
        # Most occluders are dark/neutral; a few are bright.
        if rng.random() < 0.75:
            colour = rng.uniform(5, 70, size=3)
        else:
            colour = rng.uniform(170, 245, size=3)
        a = float(rng.uniform(*cfg.occlusion_alpha))
        # Rounded rectangle gives a more natural object silhouette.
        patch = np.zeros((oh, ow, 3), dtype=np.float32) + colour
        mask = np.zeros((oh, ow), dtype=np.float32)
        cv2.rectangle(mask, (0, 0), (ow - 1, oh - 1), 1.0, -1)
        k = max(1, min(ow, oh) // 5)
        if k > 1:
            mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=k)
        mask = np.clip(mask * a, 0.0, 1.0)
        region = out[y0 : y0 + oh, x0 : x0 + ow]
        out[y0 : y0 + oh, x0 : x0 + ow] = region * (1.0 - mask[..., None]) + patch * mask[..., None]
        vpatch = valid[y0 : y0 + oh, x0 : x0 + ow]
        valid[y0 : y0 + oh, x0 : x0 + ow] = vpatch * (1.0 - mask)

    return np.clip(out, 0.0, 255.0), {"n_occlusion": n}, valid


def _jpeg(
    img: np.ndarray, rng: np.random.Generator, cfg: PhotometricConfig
) -> tuple[np.ndarray, int | None]:
    if rng.random() >= cfg.jpeg_prob:
        return img, None
    q = int(rng.integers(cfg.jpeg_quality[0], cfg.jpeg_quality[1] + 1))
    u8 = np.clip(img, 0, 255).astype(np.uint8)
    rgb = cv2.cvtColor(u8, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".jpg", rgb, [int(cv2.IMWRITE_JPEG_QUALITY), q])
    if not ok:
        return img, None
    dec = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    return cv2.cvtColor(dec, cv2.COLOR_BGR2RGB).astype(np.float32), q


def corrupt(
    image: np.ndarray,
    rng: np.random.Generator,
    cfg: PhotometricConfig,
    *,
    background: np.ndarray | None = None,
    return_mask: bool = False,
):
    """Apply a random photometric degradation chain to an RGB uint8 image.

    When ``return_mask`` is true, also returns a ``(H, W)`` float32 mask where
    1 marks trustworthy map content and 0 marks pixels corrupted by
    salt-and-pepper noise, scratches/dust/smudges or occluders.
    """
    if not cfg.enabled:
        if return_mask:
            return image.copy(), {"enabled": False}, np.ones(image.shape[:2], dtype=np.float32)
        return image.copy(), {"enabled": False}

    info: dict = {"enabled": True}
    out = image.astype(np.float32)
    valid = np.ones(image.shape[:2], dtype=np.float32)

    if rng.random() < cfg.blur_prob:
        out, k = _gaussian_blur(out, rng, cfg)
        info["blur_kernel"] = k

    out, illum = _illumination(out, rng, cfg)
    info.update(illum)

    out, noise, vmask = _noise(out, rng, cfg)
    info.update(noise)
    valid *= vmask

    if rng.random() < cfg.scratch_prob:
        out, scr, vmask = _scratches(out, rng, cfg, background)
        info.update(scr)
        valid *= vmask
    else:
        info.update({"n_scratch": 0, "n_dust": 0, "n_smudge": 0})

    out, occ, vmask = _occlusions(out, rng, cfg)
    info.update(occ)
    valid *= vmask

    out, q = _jpeg(out, rng, cfg)
    if q is not None:
        info["jpeg_quality"] = q

    info["valid_frac"] = float(valid.mean())
    result = np.clip(out, 0, 255).astype(np.uint8)
    if return_mask:
        return result, info, valid
    return result, info