"""Configuration objects for the fixed-map localization infra.

All coordinates are expressed in map-image pixels unless stated otherwise.
x grows to the right, y grows downwards, yaw is the angle of the camera
heading measured from +x (clockwise on screen because y points down).
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class GeometryConfig:
    """How an observation window is cut out and geometrically corrupted."""

    obs_width: int = 256
    obs_height: int = 256
    # Map pixels covered by the observation width before any augmentation.
    # Larger fov_px => wider context / smaller objects in the crop.
    fov_px: float = 320.0

    # Multiplicative zoom applied to the crop content (1.0 = no change).
    zoom: tuple[float, float] = (0.80, 1.25)
    # Independent x/y aspect scaling applied after zoom.
    aspect: tuple[float, float] = (0.95, 1.05)
    # Affine shear coefficients, sampled uniformly in [-shear, shear].
    shear: float = 0.18
    # Extra in-plane rotation on top of the camera yaw, in degrees.
    rot_deg: float = 12.0
    # Translation of the crop content, in observation pixels.
    translate_px: float = 8.0
    # Random mild perspective (oblique viewing) applied with this probability.
    persp_prob: float = 0.5
    # Corner jitter as a fraction of the observation size.
    persp_strength: float = 0.02

    # 'background' fills unknown area with the map background colour,
    # 'reflect' mirrors the map, 'zero' fills black.
    border: str = "background"


@dataclass
class PhotometricConfig:
    """Sensor / capture degradations, sampled independently per frame."""

    enabled: bool = True

    # Additive Gaussian sensor noise.
    gauss_sigma: tuple[float, float] = (0.0, 14.0)
    # Salt-and-pepper outlier fraction.
    salt_pepper: tuple[float, float] = (0.0, 0.0025)
    # Global brightness gain / gamma / per-channel tint.
    gain: tuple[float, float] = (0.65, 1.35)
    gamma: tuple[float, float] = (0.75, 1.35)
    tint: float = 0.06
    # Vignetting and a linear illumination gradient.
    vignette: tuple[float, float] = (0.0, 0.35)
    gradient: tuple[float, float] = (0.0, 0.20)
    # Defocus / motion blur.
    blur_prob: float = 0.45
    blur_max: int = 5

    # Scratches, dust and smudges drawn *after* noise (on the lens/sensor).
    scratch_prob: float = 0.85
    scratch_count: tuple[int, int] = (0, 7)
    scratch_width: tuple[int, int] = (1, 3)
    scratch_length: tuple[float, float] = (0.25, 1.40)  # x image diagonal
    scratch_alpha: tuple[float, float] = (0.35, 0.95)
    dust_count: tuple[int, int] = (0, 25)
    smudge_prob: float = 0.35

    # Large occluders (e.g. a hand, a pole, a passing object).  These are
    # explicitly marked invalid so an observation model can learn to ignore
    # them instead of hallucinating map structure.
    occlusion_prob: float = 0.15
    occlusion_count: tuple[int, int] = (1, 3)
    occlusion_frac: tuple[float, float] = (0.05, 0.22)  # fraction of the image
    occlusion_alpha: tuple[float, float] = (0.65, 1.0)

    # Final image compression artefacts.
    jpeg_prob: float = 0.4
    jpeg_quality: tuple[int, int] = (30, 90)


@dataclass
class TrajectoryConfig:
    """Smooth trajectory generator (waypoint navigation by default)."""

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
    yaw_jitter_deg: float = 0.15


@dataclass
class InfraConfig:
    geometry: GeometryConfig = field(default_factory=GeometryConfig)
    photometric: PhotometricConfig = field(default_factory=PhotometricConfig)
    trajectory: TrajectoryConfig = field(default_factory=TrajectoryConfig)

    # Per-channel normalisation used by the torch datasets.
    norm_mean: tuple[float, float, float] = (0.5, 0.5, 0.5)
    norm_std: tuple[float, float, float] = (0.5, 0.5, 0.5)