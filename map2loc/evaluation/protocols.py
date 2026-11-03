"""Evaluation protocols for the updated perspective-camera task.

All methods see exactly the same rendered observations, attitude measurements,
trajectories and ego-motion.  Each observation is a perspective camera view at
a fixed height; the matchers rectify it to BEV with the measured attitude.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from data_gen.config import (
    GeometryConfig,
    InfraConfig,
    MeasurementConfig,
    PhotometricConfig,
    TrajectoryConfig,
)
from data_gen.map import FixedMap, Region
from data_gen.observation import ObservationGenerator
from data_gen.trajectory import generate_trajectory, sample_pose
from data_gen.types import Observation, Pose


_CORRUPTION_NAMES = ("clean", "mild", "moderate", "severe", "scratches", "pixel")


@dataclass
class CorruptionLevel:
    name: str
    cfg: InfraConfig
    geometry: bool = True
    photometric: bool = True


def _base(cfg: InfraConfig | None = None) -> InfraConfig:
    return copy.deepcopy(cfg) if cfg is not None else InfraConfig()


def corruption_level(name: str, base: InfraConfig | None = None) -> CorruptionLevel:
    if name not in _CORRUPTION_NAMES:
        raise ValueError(f"unknown corruption level: {name!r}")
    cfg = _base(base)
    # All attitude channels are measured with the same 5 deg uncertainty.
    cfg.measurement = MeasurementConfig(
        yaw_sigma_deg=5.0, pitch_sigma_deg=5.0, roll_sigma_deg=5.0
    )

    if name == "clean":
        return CorruptionLevel(name, cfg, geometry=False, photometric=False)

    if name == "mild":
        cfg.photometric = PhotometricConfig(
            gauss_sigma=(0.0, 4.0),
            pixel_gauss_sigma=(0.0, 1.5),
            invalid_pixel_rate=0.10,
            intensity_scale=(0.95, 1.05),
            intensity_shift=(-3.0, 3.0),
            scratch_prob=0.15,
            scratch_count=(0, 1),
        )
        return CorruptionLevel(name, cfg)

    if name == "moderate":
        return CorruptionLevel(name, cfg)

    if name == "severe":
        cfg.photometric = PhotometricConfig(
            gauss_sigma=(8.0, 22.0),
            pixel_gauss_sigma=(2.0, 8.0),
            invalid_pixel_rate=0.75,
            intensity_scale=(0.70, 1.30),
            intensity_shift=(-15.0, 15.0),
            scratch_prob=0.95,
            scratch_count=(2, 5),
            scratch_width=(1, 3),
            scratch_delta=(40.0, 90.0),
        )
        return CorruptionLevel(name, cfg)

    if name == "scratches":
        cfg.photometric = PhotometricConfig(
            gauss_sigma=(2.0, 8.0),
            pixel_gauss_sigma=(0.5, 3.0),
            invalid_pixel_rate=0.20,
            scratch_prob=1.0,
            scratch_count=(3, 8),
            scratch_width=(2, 4),
            scratch_delta=(50.0, 110.0),
        )
        return CorruptionLevel(name, cfg)

    if name == "pixel":
        cfg.photometric = PhotometricConfig(
            gauss_sigma=(0.0, 3.0),
            pixel_gauss_sigma=(0.0, 1.0),
            invalid_pixel_rate=0.80,
            intensity_scale=(0.9, 1.1),
            intensity_shift=(-5.0, 5.0),
            scratch_prob=0.0,
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
        hsv[..., 1] = np.clip(
            hsv[..., 1].astype(np.float32) * 1.25, 0, 255
        ).astype(np.uint8)
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
    map_ = FixedMap(map_path)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    eval_maps: dict[str, FixedMap] = {}
    stem = Path(map_path).stem
    for kind in variants:
        p = out / f"{stem}_{kind}.png"
        cv2.imwrite(
            str(p), cv2.cvtColor(_derive_variant(map_, kind), cv2.COLOR_RGB2BGR)
        )
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
    poses: np.ndarray  # (T, 5) effective camera x, y, yaw, pitch, roll
    observations: list[Observation]
    gt_xy: np.ndarray  # (T, 2) camera ground position
    gt_center_xy: np.ndarray  # (T, 2) ground point at the image centre
    gt_yaw: np.ndarray  # (T,) effective heading
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
        map_,
        corruption.cfg.geometry,
        corruption.cfg.photometric,
        corruption.cfg.measurement,
    )
    obs: list[Observation] = []
    wrong_map = distractor_map or map_
    wrong_rng = np.random.default_rng(seed + 777)
    wrong_gen = ObservationGenerator(
        wrong_map,
        corruption.cfg.geometry,
        corruption.cfg.photometric,
        corruption.cfg.measurement,
    )
    wrong_region = wrong_map.region("content", 0.1)
    for i, p in enumerate(poses):
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
            pose = Pose(
                float(p[0]),
                float(p[1]),
                float(p[2]),
                float(p[3]),
                float(p[4]),
            )
            o = gen.generate(
                pose,
                frame_rng,
                geometry=corruption.geometry,
                photometric=corruption.photometric,
            )
        obs.append(o)

    eff = np.stack([o.pose.as_array() for o in obs]).astype(np.float32)
    return RenderedSequence(
        map=map_,
        poses=eff,
        observations=obs,
        gt_xy=np.stack([o.gt_xy for o in obs]).astype(np.float32),
        gt_center_xy=np.stack([o.center_xy for o in obs]).astype(np.float32),
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
    """Relative motion from pose ``t-1`` to ``t`` in the previous body frame."""
    if t <= 0:
        return np.zeros(3, dtype=np.float32)
    x0, y0, yaw0 = poses[t - 1, :3]
    x1, y1, yaw1 = poses[t, :3]
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