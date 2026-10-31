"""Evaluation protocols: corruption levels, map splits and sequences.

All methods in a comparison see exactly the same rendered observations,
trajectories and ego-motion.  Observations are rendered once and then passed
to each matcher, so corruption cannot accidentally differ between methods.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from ..config import InfraConfig, GeometryConfig, PhotometricConfig, TrajectoryConfig
from ..map import FixedMap, Region
from ..observation import ObservationGenerator
from ..trajectory import generate_trajectory
from ..types import Observation, Pose


# --------------------------------------------------------------------------
# corruption levels
# --------------------------------------------------------------------------
_CORRUPTION_NAMES = (
    "clean",
    "mild",
    "moderate",
    "severe",
    "scratches",
    "occlusion",
    "geometric",
)


@dataclass
class CorruptionLevel:
    name: str
    cfg: InfraConfig
    geometry: bool = True
    photometric: bool = True


def _base(cfg: InfraConfig | None = None) -> InfraConfig:
    return copy.deepcopy(cfg) if cfg is not None else InfraConfig()


def corruption_level(name: str, base: InfraConfig | None = None) -> CorruptionLevel:
    """Return a preset ``InfraConfig`` plus on/off flags for rendering."""
    if name not in _CORRUPTION_NAMES:
        raise ValueError(f"unknown corruption level: {name!r}")
    cfg = _base(base)

    if name == "clean":
        return CorruptionLevel(name, cfg, geometry=False, photometric=False)

    if name == "mild":
        cfg.geometry = GeometryConfig(
            zoom=(0.92, 1.08),
            aspect=(0.98, 1.02),
            shear=0.06,
            rot_deg=5.0,
            translate_px=3.0,
            persp_prob=0.2,
            persp_strength=0.01,
        )
        cfg.photometric = PhotometricConfig(
            gauss_sigma=(0.0, 6.0),
            salt_pepper=(0.0, 0.0005),
            gain=(0.85, 1.15),
            gamma=(0.9, 1.1),
            tint=0.03,
            vignette=(0.0, 0.15),
            gradient=(0.0, 0.08),
            blur_prob=0.2,
            scratch_prob=0.25,
            scratch_count=(0, 2),
            dust_count=(0, 8),
            smudge_prob=0.1,
            occlusion_prob=0.03,
            jpeg_prob=0.15,
            jpeg_quality=(60, 90),
        )
        return CorruptionLevel(name, cfg)

    if name == "moderate":
        return CorruptionLevel(name, cfg)

    if name == "severe":
        cfg.geometry = GeometryConfig(
            zoom=(0.72, 1.35),
            aspect=(0.92, 1.08),
            shear=0.24,
            rot_deg=18.0,
            translate_px=12.0,
            persp_prob=0.7,
            persp_strength=0.03,
        )
        cfg.photometric = PhotometricConfig(
            gauss_sigma=(6.0, 24.0),
            salt_pepper=(0.0, 0.006),
            gain=(0.5, 1.6),
            gamma=(0.6, 1.6),
            tint=0.10,
            vignette=(0.1, 0.5),
            gradient=(0.05, 0.35),
            blur_prob=0.8,
            blur_max=7,
            scratch_prob=1.0,
            scratch_count=(3, 9),
            scratch_width=(1, 4),
            scratch_length=(0.4, 1.6),
            scratch_alpha=(0.5, 1.0),
            dust_count=(5, 35),
            smudge_prob=0.7,
            occlusion_prob=0.35,
            occlusion_count=(1, 4),
            occlusion_frac=(0.08, 0.30),
            jpeg_prob=0.8,
            jpeg_quality=(20, 60),
        )
        return CorruptionLevel(name, cfg)

    if name == "scratches":
        cfg.photometric = PhotometricConfig(
            gauss_sigma=(2.0, 10.0),
            gain=(0.8, 1.2),
            gamma=(0.85, 1.15),
            vignette=(0.0, 0.2),
            gradient=(0.0, 0.1),
            blur_prob=0.3,
            scratch_prob=1.0,
            scratch_count=(5, 12),
            scratch_width=(2, 5),
            scratch_length=(0.5, 1.8),
            scratch_alpha=(0.6, 1.0),
            dust_count=(10, 40),
            smudge_prob=0.6,
            occlusion_prob=0.0,
            jpeg_prob=0.4,
        )
        return CorruptionLevel(name, cfg)

    if name == "occlusion":
        cfg.geometry = GeometryConfig(
            zoom=(0.85, 1.15),
            shear=0.10,
            rot_deg=8.0,
            translate_px=5.0,
            persp_prob=0.3,
        )
        cfg.photometric = PhotometricConfig(
            gauss_sigma=(2.0, 12.0),
            gain=(0.75, 1.25),
            gamma=(0.85, 1.15),
            vignette=(0.0, 0.2),
            blur_prob=0.3,
            scratch_prob=0.2,
            scratch_count=(0, 2),
            dust_count=(0, 10),
            smudge_prob=0.2,
            occlusion_prob=1.0,
            occlusion_count=(2, 5),
            occlusion_frac=(0.10, 0.30),
            occlusion_alpha=(0.85, 1.0),
            jpeg_prob=0.3,
        )
        return CorruptionLevel(name, cfg)

    if name == "geometric":
        cfg.geometry = GeometryConfig(
            zoom=(0.60, 1.55),
            aspect=(0.80, 1.20),
            shear=0.35,
            rot_deg=30.0,
            translate_px=18.0,
            persp_prob=1.0,
            persp_strength=0.055,
        )
        cfg.photometric = PhotometricConfig(
            gauss_sigma=(0.0, 8.0),
            gain=(0.8, 1.2),
            gamma=(0.9, 1.1),
            vignette=(0.0, 0.15),
            blur_prob=0.2,
            scratch_prob=0.1,
            scratch_count=(0, 1),
            dust_count=(0, 5),
            smudge_prob=0.0,
            occlusion_prob=0.0,
            jpeg_prob=0.1,
        )
        return CorruptionLevel(name, cfg)

    raise AssertionError("unreachable")


def all_corruption_levels(base: InfraConfig | None = None) -> list[CorruptionLevel]:
    return [corruption_level(n, base) for n in _CORRUPTION_NAMES]


# --------------------------------------------------------------------------
# map splits
# --------------------------------------------------------------------------
@dataclass
class MapSplit:
    """Train map(s), held-out map variants, and spatial regions."""

    name: str
    train_map: FixedMap
    eval_maps: dict[str, FixedMap]
    train_region: Region
    seen_region: Region
    unseen_region: Region
    metadata: dict = field(default_factory=dict)

    def eval_map(self, name: str | None = None) -> FixedMap:
        if name is None or name in ("train", "seen"):
            return self.train_map
        if name not in self.eval_maps:
            raise KeyError(
                f"unknown eval map {name!r}; available: {sorted(self.eval_maps)}"
            )
        return self.eval_maps[name]


def _derive_variant(map_: FixedMap, kind: str) -> np.ndarray:
    img = map_.image.copy()
    if kind == "rot90":
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    elif kind == "mirror":
        img = cv2.flip(img, 1)
    elif kind == "rot180":
        img = cv2.rotate(img, cv2.ROTATE_180)
    elif kind == "recolor":
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        hsv[..., 0] = (hsv[..., 0].astype(np.int16) + 40) % 180
        hsv[..., 1] = np.clip(hsv[..., 1].astype(np.float32) * 1.25, 0, 255).astype(np.uint8)
        img = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    else:
        raise ValueError(f"unknown map variant: {kind!r}")
    return img


def build_map_split(
    map_path: str | Path,
    *,
    out_dir: str | Path = "outputs/maps",
    variants: tuple[str, ...] = ("rot90", "mirror", "recolor"),
    spatial_split: float = 0.6,
) -> MapSplit:
    """Build a deterministic train/eval split from one or more maps.

    With a single supplied map, held-out *maps* are structural variants of the
    source map (rotation / mirror / recolouring) written to ``out_dir``.  A
    real unseen-map evaluation simply supplies additional map paths to
    :func:`build_map_split_from_paths`.
    """
    map_ = FixedMap(map_path)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    eval_maps: dict[str, FixedMap] = {}
    stem = Path(map_path).stem
    for kind in variants:
        p = out / f"{stem}_{kind}.png"
        cv2.imwrite(str(p), cv2.cvtColor(_derive_variant(map_, kind), cv2.COLOR_RGB2BGR))
        eval_maps[kind] = FixedMap(p)

    bbox = map_.region("content", 0.02)
    x_split = bbox.x0 + spatial_split * bbox.width
    seen = Region(bbox.x0, bbox.y0, x_split, bbox.y1)
    unseen = Region(x_split, bbox.y0, bbox.x1, bbox.y1)
    return MapSplit(
        name=stem,
        train_map=map_,
        eval_maps=eval_maps,
        train_region=seen,
        seen_region=seen,
        unseen_region=unseen,
        metadata={
            "source": str(map_path),
            "variants": list(variants),
            "spatial_split": float(spatial_split),
        },
    )


def build_map_split_from_paths(
    train_path: str | Path,
    eval_paths: list[str | Path],
    *,
    spatial_split: float = 0.6,
) -> MapSplit:
    """True unseen-map split from explicit map files."""
    map_ = FixedMap(train_path)
    bbox = map_.region("content", 0.02)
    x_split = bbox.x0 + spatial_split * bbox.width
    seen = Region(bbox.x0, bbox.y0, x_split, bbox.y1)
    unseen = Region(x_split, bbox.y0, bbox.x1, bbox.y1)
    return MapSplit(
        name=Path(train_path).stem,
        train_map=map_,
        eval_maps={Path(p).stem: FixedMap(p) for p in eval_paths},
        train_region=seen,
        seen_region=seen,
        unseen_region=unseen,
        metadata={"train": str(train_path), "eval": [str(p) for p in eval_paths]},
    )


# --------------------------------------------------------------------------
# sequence rendering
# --------------------------------------------------------------------------
@dataclass
class RenderedSequence:
    map: FixedMap
    poses: np.ndarray  # (T, 3) ground-truth x, y, yaw
    observations: list[Observation]
    gt_xy: np.ndarray  # (T, 2) centre-of-observation target
    gt_yaw: np.ndarray  # (T,) heading of the observation
    region: tuple[float, float, float, float]
    corruption: str


def render_sequence(
    map_: FixedMap,
    corruption: CorruptionLevel,
    *,
    seed: int,
    seq_len: int = 40,
    step_stride: int = 6,
    region: Region | None = None,
    start: object | None = None,
    distractor_frames: int = 0,
    distractor_map: FixedMap | None = None,
) -> RenderedSequence:
    """Render a deterministic observation sequence.

    ``distractor_frames`` replaces the first *k* observations with views from a
    different, wrong location on the map.  This creates an identically-defined
    "incorrect initial localization" event for every method and tests recovery.
    """
    rng = np.random.default_rng(seed)
    n_steps = (seq_len - 1) * step_stride + 1
    traj = generate_trajectory(
        map_,
        corruption.cfg.trajectory,
        n_steps=n_steps,
        rng=rng,
        region=region,
        start=start,
    )
    pose_idx = np.arange(seq_len) * step_stride
    poses = traj.poses[pose_idx]

    gen = ObservationGenerator(
        map_, corruption.cfg.geometry, corruption.cfg.photometric
    )
    obs: list[Observation] = []
    gt_xy = []
    wrong_map = distractor_map or map_
    wrong_rng = np.random.default_rng(seed + 777)
    wrong_gen = ObservationGenerator(
        wrong_map, corruption.cfg.geometry, corruption.cfg.photometric
    )
    wrong_region = wrong_map.region("content", 0.1)
    for i, p in enumerate(poses):
        # Independent seed per frame: every method gets the same image.
        frame_rng = np.random.default_rng(seed + 101 * (i + 1))
        if i < distractor_frames:
            wrong_pose = wrong_map.random_pose(wrong_rng, wrong_region)
            o = wrong_gen.generate(
                wrong_pose,
                frame_rng,
                geometry=corruption.geometry,
                photometric=corruption.photometric,
            )
        else:
            pose = Pose(float(p[0]), float(p[1]), float(p[2]))
            o = gen.generate(
                pose,
                frame_rng,
                geometry=corruption.geometry,
                photometric=corruption.photometric,
            )
        obs.append(o)
        gt_xy.append(o.gt_xy)
    return RenderedSequence(
        map=map_,
        poses=poses,
        observations=obs,
        gt_xy=np.stack(gt_xy).astype(np.float32),
        gt_yaw=np.array([o.pose.yaw for o in obs], dtype=np.float32),
        region=traj.region,
        corruption=corruption.name,
    )


def ego_motion_from_poses(
    poses: np.ndarray,
    t: int,
    *,
    noise=None,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Relative motion from pose ``t-1`` to ``t`` in the previous body frame.

    Returns ``(forward, right, dyaw)``.  This is the "available ego-motion"
    used by every filter comparison; add ``noise=(sigma_fwd, sigma_right,
    sigma_yaw)`` for a noisy-odometry condition.
    """
    if t <= 0:
        return np.zeros(3, dtype=np.float32)
    x0, y0, yaw0 = poses[t - 1]
    x1, y1, yaw1 = poses[t]
    dx, dy = x1 - x0, y1 - y0
    c, s = np.cos(yaw0), np.sin(yaw0)
    forward = c * dx + s * dy
    right = -s * dx + c * dy
    dyaw = (yaw1 - yaw0 + np.pi) % (2 * np.pi) - np.pi
    out = np.array([forward, right, dyaw], dtype=np.float32)
    if noise is not None and rng is not None:
        out = out + rng.normal(0.0, 1.0, size=3).astype(np.float32) * np.asarray(
            noise, dtype=np.float32
        )
    return out