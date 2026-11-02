"""Smooth camera trajectories over the map.

Two position/heading generators are available (``TrajectoryConfig.mode``):

``waypoint`` (default)
    Pick a random far-away waypoint, steer towards it with a bounded turn
    rate, then pick the next one.  Gives smooth arcs that cover the whole
    map, which is what a roaming agent / drone actually does.

``random_walk``
    Curvature is an Ornstein-Uhlenbeck process.  Produces meandering paths
    that stay local; useful for stress-testing but poor coverage.

In both modes ``pitch`` and ``roll`` are independent smooth OU processes.
``pitch`` stays in a configurable oblique range (the camera looks forward and
down, not straight down); ``roll`` is a small curve around zero.  The
returned pose array has columns ``(x, y, yaw, pitch, roll)``.
"""

from __future__ import annotations

import numpy as np

from .config import TrajectoryConfig
from .map import FixedMap, Region
from .types import Pose, Trajectory


def _wrap(angle: float) -> float:
    return float((angle + np.pi) % (2.0 * np.pi) - np.pi)


def _resolve_region(
    map_: FixedMap, cfg: TrajectoryConfig, region: Region | None
) -> Region:
    region = region or map_.region(cfg.region, cfg.region_shrink)
    margin = cfg.boundary_lookahead_px
    if region.width <= 2 * margin or region.height <= 2 * margin:
        margin = 0.0
    return region


def _reflect(pose: Pose, region: Region, margin: float) -> Pose:
    """Hard fallback: keep the pose inside and mirror the heading."""
    x, y, yaw = pose.x, pose.y, pose.yaw
    if x < region.x0 + margin:
        x = 2 * (region.x0 + margin) - x
        yaw = _wrap(np.pi - yaw)
    elif x > region.x1 - margin:
        x = 2 * (region.x1 - margin) - x
        yaw = _wrap(np.pi - yaw)
    if y < region.y0 + margin:
        y = 2 * (region.y0 + margin) - y
        yaw = _wrap(-yaw)
    elif y > region.y1 - margin:
        y = 2 * (region.y1 - margin) - y
        yaw = _wrap(-yaw)
    return Pose(float(x), float(y), _wrap(yaw), pose.pitch, pose.roll, pose.height)


def _sample_target(
    rng: np.random.Generator,
    region: Region,
    pose: Pose,
    min_dist: float,
) -> np.ndarray:
    for _ in range(32):
        target = np.array(
            [rng.uniform(region.x0, region.x1), rng.uniform(region.y0, region.y1)]
        )
        if np.linalg.norm(target - pose.xy) >= min_dist:
            return target
    return region.center


def _initial_angle(
    rng: np.random.Generator,
    sigma_deg: float,
    min_deg: float,
    max_deg: float,
    mean_deg: float = 0.0,
) -> float:
    angle = mean_deg + rng.normal(0.0, sigma_deg)
    return float(np.deg2rad(np.clip(angle, min_deg, max_deg)))


def _ou_step(
    value: float,
    rng: np.random.Generator,
    *,
    sigma_deg: float,
    corr: float,
    min_deg: float,
    max_deg: float,
    max_step_deg: float,
    mean_deg: float = 0.0,
) -> float:
    """One smooth mean-reverting OU step with bounds and a slew limit."""
    sigma = np.deg2rad(sigma_deg)
    mean = np.deg2rad(mean_deg)
    lo = np.deg2rad(min_deg)
    hi = np.deg2rad(max_deg)
    max_step = np.deg2rad(max_step_deg)
    target = mean + corr * (value - mean) + np.sqrt(
        max(0.0, 1.0 - corr**2)
    ) * rng.normal(0.0, sigma)
    target = float(np.clip(target, lo, hi))
    return value + float(np.clip(target - value, -max_step, max_step))


def _waypoint_path(
    rng: np.random.Generator,
    cfg: TrajectoryConfig,
    region: Region,
    n_steps: int,
    start: Pose,
    margin: float,
) -> Trajectory:
    pose = Pose(
        float(np.clip(start.x, region.x0 + margin, region.x1 - margin)),
        float(np.clip(start.y, region.y0 + margin, region.y1 - margin)),
        float(start.yaw),
        float(start.pitch),
        float(start.roll),
        float(start.height),
    )
    min_dist = cfg.waypoint_min_dist_frac * min(region.width, region.height)
    # A bounded turn rate implies a minimum turning radius 1/turn_rate, so a
    # waypoint can never be hit exactly.  Treat it as reached once we are
    # inside that radius, otherwise the path orbits the target forever.
    reach = max(cfg.waypoint_reach_px, 1.0 / cfg.turn_rate + 2 * cfg.step_px)
    target = _sample_target(rng, region, pose, min_dist)
    poses = np.empty((n_steps, 5), dtype=np.float32)
    yaw_jitter = np.deg2rad(cfg.yaw_jitter_deg)

    for i in range(n_steps):
        poses[i] = (pose.x, pose.y, pose.yaw, pose.pitch, pose.roll)
        if i == n_steps - 1:
            break

        to_target = target - pose.xy
        dist = float(np.linalg.norm(to_target))
        if dist < reach:
            target = _sample_target(rng, region, pose, min_dist)
            to_target = target - pose.xy

        ds = cfg.step_px * rng.uniform(1.0 - cfg.speed_jitter, 1.0 + cfg.speed_jitter)
        desired = float(np.arctan2(to_target[1], to_target[0]))
        err = _wrap(desired - pose.yaw)

        # bounded turn rate => smooth arcs; noise adds a little wander
        turn = cfg.steer_gain * err + rng.normal(0.0, yaw_jitter)
        max_turn = cfg.turn_rate * ds
        turn = float(np.clip(turn, -max_turn, max_turn))
        pose.yaw = _wrap(pose.yaw + turn)
        pose.x += ds * np.cos(pose.yaw)
        pose.y += ds * np.sin(pose.yaw)

        # Smooth oblique pitch and small roll curves.
        pose.pitch = _ou_step(
            pose.pitch,
            rng,
            sigma_deg=cfg.pitch_sigma_deg,
            corr=cfg.pitch_corr,
            min_deg=cfg.pitch_min_deg,
            max_deg=cfg.pitch_max_deg,
            max_step_deg=cfg.pitch_max_step_deg,
            mean_deg=cfg.pitch_mean_deg,
        )
        pose.roll = _ou_step(
            pose.roll,
            rng,
            sigma_deg=cfg.roll_sigma_deg,
            corr=cfg.roll_corr,
            min_deg=-cfg.roll_limit_deg,
            max_deg=cfg.roll_limit_deg,
            max_step_deg=cfg.roll_max_step_deg,
        )

        # soft border steering if the look-ahead leaves the region
        look = pose.xy + cfg.boundary_lookahead_px * np.array(
            [np.cos(pose.yaw), np.sin(pose.yaw)]
        )
        if not region.contains(look):
            desired_center = float(
                np.arctan2(region.center[1] - pose.y, region.center[0] - pose.x)
            )
            pose.yaw = _wrap(
                pose.yaw + cfg.boundary_gain * _wrap(desired_center - pose.yaw)
            )

        if not region.contains(pose.xy, margin):
            pose = _reflect(pose, region, margin)

    return Trajectory(poses=poses, region=region.as_tuple())


def _random_walk_path(
    rng: np.random.Generator,
    cfg: TrajectoryConfig,
    region: Region,
    n_steps: int,
    start: Pose,
    margin: float,
) -> Trajectory:
    pose = Pose(
        float(np.clip(start.x, region.x0 + margin, region.x1 - margin)),
        float(np.clip(start.y, region.y0 + margin, region.y1 - margin)),
        float(start.yaw),
        float(start.pitch),
        float(start.roll),
        float(start.height),
    )
    center = region.center
    poses = np.empty((n_steps, 5), dtype=np.float32)
    kappa = rng.normal(0.0, cfg.curvature_sigma)
    yaw_jitter = np.deg2rad(cfg.yaw_jitter_deg)

    for i in range(n_steps):
        poses[i] = (pose.x, pose.y, pose.yaw, pose.pitch, pose.roll)
        if i == n_steps - 1:
            break

        kappa = (
            cfg.curvature_corr * kappa
            + np.sqrt(1.0 - cfg.curvature_corr**2)
            * rng.normal(0.0, cfg.curvature_sigma)
        )

        look = pose.xy + cfg.boundary_lookahead_px * np.array(
            [np.cos(pose.yaw), np.sin(pose.yaw)]
        )
        if not region.contains(look):
            desired = float(np.arctan2(center[1] - pose.y, center[0] - pose.x))
            err = _wrap(desired - pose.yaw)
            over = max(
                0.0,
                max(
                    region.x0 - look[0],
                    look[0] - region.x1,
                    region.y0 - look[1],
                    look[1] - region.y1,
                ),
            )
            gain = min(
                1.0,
                cfg.boundary_gain + over / max(cfg.boundary_lookahead_px, 1e-6),
            )
            kappa = 0.0
            pose.yaw = _wrap(pose.yaw + gain * err)

        ds = cfg.step_px * rng.uniform(1.0 - cfg.speed_jitter, 1.0 + cfg.speed_jitter)
        pose.yaw = _wrap(pose.yaw + kappa * ds + rng.normal(0.0, yaw_jitter))
        pose.x += ds * np.cos(pose.yaw)
        pose.y += ds * np.sin(pose.yaw)

        pose.pitch = _ou_step(
            pose.pitch,
            rng,
            sigma_deg=cfg.pitch_sigma_deg,
            corr=cfg.pitch_corr,
            min_deg=cfg.pitch_min_deg,
            max_deg=cfg.pitch_max_deg,
            max_step_deg=cfg.pitch_max_step_deg,
            mean_deg=cfg.pitch_mean_deg,
        )
        pose.roll = _ou_step(
            pose.roll,
            rng,
            sigma_deg=cfg.roll_sigma_deg,
            corr=cfg.roll_corr,
            min_deg=-cfg.roll_limit_deg,
            max_deg=cfg.roll_limit_deg,
            max_step_deg=cfg.roll_max_step_deg,
        )

        if not region.contains(pose.xy, margin):
            pose = _reflect(pose, region, margin)

    return Trajectory(poses=poses, region=region.as_tuple())


def generate_trajectory(
    map_: FixedMap,
    cfg: TrajectoryConfig | None = None,
    *,
    n_steps: int = 256,
    rng: np.random.Generator | None = None,
    region: Region | None = None,
    start: Pose | None = None,
) -> Trajectory:
    """Generate a smooth pose sequence inside ``region``.

    Parameters
    ----------
    n_steps:
        Number of poses.
    start:
        Optional initial pose.  If omitted a random pose and heading are used,
        with pitch drawn in the oblique range and roll near zero.
    region:
        Allowed area.  Defaults to the map's content bbox (the pitch).
    """
    cfg = cfg or TrajectoryConfig()
    rng = rng or np.random.default_rng()
    if n_steps < 1:
        raise ValueError("n_steps must be >= 1")

    region = _resolve_region(map_, cfg, region)
    margin = cfg.boundary_lookahead_px
    if region.width <= 2 * margin or region.height <= 2 * margin:
        margin = 0.0

    if start is None:
        start = map_.random_pose(rng, region)
        start.pitch = _initial_angle(
            rng,
            cfg.pitch_sigma_deg,
            cfg.pitch_min_deg,
            cfg.pitch_max_deg,
            cfg.pitch_mean_deg,
        )
        start.roll = _initial_angle(
            rng,
            cfg.roll_sigma_deg,
            -cfg.roll_limit_deg,
            cfg.roll_limit_deg,
        )

    if cfg.mode == "waypoint":
        return _waypoint_path(rng, cfg, region, n_steps, start, margin)
    if cfg.mode == "random_walk":
        return _random_walk_path(rng, cfg, region, n_steps, start, margin)
    raise ValueError(f"unknown trajectory mode: {cfg.mode!r}")


def sample_pose(
    map_: FixedMap,
    rng: np.random.Generator,
    *,
    region: Region | None = None,
    region_kind: str = "content",
    region_shrink: float = 0.02,
    yaw: float | None = None,
    pitch: float | None = None,
    roll: float | None = None,
    cfg: TrajectoryConfig | None = None,
) -> Pose:
    """Uniformly sample a camera pose inside the allowed region."""
    cfg = cfg or TrajectoryConfig()
    region = region or map_.region(region_kind, region_shrink)
    pose = map_.random_pose(rng, region, yaw=yaw)
    if pitch is None:
        pitch = float(
            rng.uniform(
                np.deg2rad(cfg.pitch_min_deg), np.deg2rad(cfg.pitch_max_deg)
            )
        )
    if roll is None:
        roll = _initial_angle(
            rng,
            cfg.roll_sigma_deg,
            -cfg.roll_limit_deg,
            cfg.roll_limit_deg,
        )
    pose.pitch = float(pitch)
    pose.roll = float(roll)
    return pose