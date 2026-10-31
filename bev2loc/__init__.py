"""bev2loc: fixed-map localization observation infrastructure.

See ``README.md`` for the coordinate conventions and usage examples.
"""

from .config import (
    GeometryConfig,
    InfraConfig,
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
]