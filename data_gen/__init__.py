"""data_gen: fixed-map localization observation infrastructure.

See ``README.md`` for the camera model, coordinate conventions and usage
examples.
"""

from .camera import (
    apply_homography,
    camera_basis,
    focal_pixels,
    front_mask,
    map_to_view,
    render_validity,
    render_view,
    view_to_map,
)
from .config import (
    GeometryConfig,
    InfraConfig,
    MeasurementConfig,
    PhotometricConfig,
    TrajectoryConfig,
)
from .dataset import (
    PoseDataset,
    TrajectoryDataset,
    localization_collate,
)
from .map import FixedMap, Region
from .observation import ObservationGenerator
from .trajectory import generate_trajectory, sample_pose
from .types import Augmentation, Observation, Pose, Trajectory

__all__ = [
    "GeometryConfig",
    "InfraConfig",
    "MeasurementConfig",
    "PhotometricConfig",
    "TrajectoryConfig",
    "PoseDataset",
    "TrajectoryDataset",
    "localization_collate",
    "FixedMap",
    "Region",
    "ObservationGenerator",
    "generate_trajectory",
    "sample_pose",
    "Augmentation",
    "Observation",
    "Pose",
    "Trajectory",
    "apply_homography",
    "camera_basis",
    "focal_pixels",
    "front_mask",
    "map_to_view",
    "render_validity",
    "render_view",
    "view_to_map",
]