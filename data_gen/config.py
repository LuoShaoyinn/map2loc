"""Configuration objects for the fixed-map localization infra.

All coordinates are expressed in map-image pixels unless stated otherwise.
x grows to the right, y grows downwards, yaw is the angle of the camera
heading measured from +x (clockwise on screen because y points down).

The observation model is a real pinhole camera looking down at the ground
plane.  The map is the ground plane (``z = 0``), the camera sits at a fixed
height above it, and the rendered view is clipped to a fixed image size.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class GeometryConfig:
    """Pinhole camera used to render an observation window."""

    obs_width: int = 256
    obs_height: int = 256

    # Fixed camera height above the ground plane, in map pixels.
    height_px: float = 180.0
    # Ground-plane width covered by the view at nadir (pitch = -90 deg).
    # The focal length is derived from this and ``height_px``; smaller
    # footprint => narrower field of view.
    footprint_px: float = 120.0

    # Random zoom applied to the rendered view.  With a fixed camera height
    # this is exactly equivalent to changing the height, so it is the only
    # scale augmentation needed.  Deliberately wide.
    scale: tuple[float, float] = (0.70, 1.40)

    # Small per-frame head / gimbal rotations around zero, in degrees.
    # Sampled uniformly in ``[-head_*_deg, +head_*_deg]`` and added to the
    # smooth camera attitude before rendering.  Kept small relative to the
    # oblique base pose.
    head_pitch_deg: float = 3.0
    head_yaw_deg: float = 5.0
    head_roll_deg: float = 1.5

    # 'background' fills unknown area with the map background colour,
    # 'reflect' mirrors the map, 'zero' fills black.
    border: str = "background"


@dataclass
class PhotometricConfig:
    """Sensor noise and image-corruption model.

    The model intentionally combines several independent effects:

    * a per-view Gaussian sensor sigma applied to every pixel/channel;
    * an extra independent Gaussian noise whose sigma is drawn **per pixel**;
    * per-view multiplicative/additive brightness scaling;
    * random scratches (thin lines) that can also invalidate pixels;
    * a configurable fraction of pixels marked invalid in the mask.

    Invalid pixels keep their rendered value; they are *untrustworthy*, not
    necessarily visually obvious.
    """

    enabled: bool = True

    # Additive Gaussian sensor noise, sigma in 8-bit intensity units, drawn
    # once per view and applied independently to every pixel/channel.
    gauss_sigma: tuple[float, float] = (1.0, 10.0)

    # Extra zero-mean Gaussian noise with an independent sigma per pixel
    # (8-bit intensity units).  This makes the noise heteroscedastic.
    pixel_gauss_sigma: tuple[float, float] = (0.5, 4.0)

    # Fraction of pixels randomly marked invalid in the validity mask.
    invalid_pixel_rate: float = 0.5

    # Random per-view brightness scaling: out = out * scale + shift.
    intensity_scale: tuple[float, float] = (0.85, 1.15)
    intensity_shift: tuple[float, float] = (-8.0, 8.0)

    # Random scratches: thin lines drawn over the view.  They hide the map,
    # so they are also marked invalid by default.
    scratch_prob: float = 0.5
    scratch_count: tuple[int, int] = (1, 3)
    scratch_width: tuple[int, int] = (1, 2)
    # Brightness delta of a scratch, in 8-bit units (sign is random).
    scratch_delta: tuple[float, float] = (25.0, 70.0)
    scratch_invalid: bool = True


@dataclass
class MeasurementConfig:
    """Noisy attitude measurements supplied to the model.

    The camera pitch and yaw are *inputs*: ground truth plus a small
    independent Gaussian error.  Position is not given.
    """

    yaw_sigma_deg: float = 3.5
    pitch_sigma_deg: float = 2.5


@dataclass
class TrajectoryConfig:
    """Smooth camera trajectory.

    ``x``, ``y``, ``yaw``, ``pitch`` and ``roll`` all follow smooth curves.
    Pitch is a mean-reverting oblique tilt (the camera looks forward/down, not
    straight down) and roll is a small mean-reverting curve around zero, while
    x/y/yaw use waypoint navigation (broad map coverage) or a local
    OU-curvature random walk.
    """

    # 'waypoint': steer between random far-away targets (broad coverage).
    # 'random_walk': OU curvature meander (local coverage).
    mode: str = "waypoint"
    step_px: float = 3.0
    speed_jitter: float = 0.35
    # Waypoint mode: arrival radius, minimum distance of a new target, and
    # steering law.  turn_rate is in rad per map pixel (bounded curvature).
    waypoint_reach_px: float = 22.0
    waypoint_min_dist_frac: float = 0.40
    steer_gain: float = 0.20
    turn_rate: float = 0.014
    # Random-walk mode: curvature is an OU process in rad/px.
    curvature_sigma: float = 0.0045
    curvature_corr: float = 0.94
    # Soft steering back towards the region when approaching its border.
    boundary_lookahead_px: float = 40.0
    boundary_gain: float = 0.30
    # 'content' = bbox of non-background pixels (the pitch), 'full' = image.
    region: str = "content"
    region_shrink: float = 0.02
    # Small random yaw jitter added every step (hand tremor / heading noise).
    yaw_jitter_deg: float = 0.8

    # Smooth pitch curve (radians at runtime, degrees in the config).
    # Pitch is measured from horizontal: 0 looks ahead along the heading,
    # -90 looks straight down.  The camera therefore looks obliquely at the
    # ground, with a smooth tilt inside [-80, 0] degrees.
    pitch_mean_deg: float = -45.0
    pitch_sigma_deg: float = 25.0
    pitch_corr: float = 0.985
    pitch_min_deg: float = -80.0
    pitch_max_deg: float = 0.0
    pitch_max_step_deg: float = 0.30
    # Smooth roll curve around zero.
    roll_sigma_deg: float = 1.0
    roll_corr: float = 0.98
    roll_limit_deg: float = 3.0
    roll_max_step_deg: float = 0.15


@dataclass
class InfraConfig:
    geometry: GeometryConfig = field(default_factory=GeometryConfig)
    photometric: PhotometricConfig = field(default_factory=PhotometricConfig)
    measurement: MeasurementConfig = field(default_factory=MeasurementConfig)
    trajectory: TrajectoryConfig = field(default_factory=TrajectoryConfig)

    # Per-channel normalisation used by the torch datasets.
    norm_mean: tuple[float, float, float] = (0.5, 0.5, 0.5)
    norm_std: tuple[float, float, float] = (0.5, 0.5, 0.5)