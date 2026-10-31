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
from .evaluation import (
    BenchmarkConfig,
    BenchmarkRunner,
    Scenario,
    build_map_split,
    build_map_split_from_paths,
    corruption_level,
    default_scenarios,
    ego_motion_from_poses,
    render_sequence,
    save_results,
)
from .map import FixedMap, Region
from .matching import (
    ClassicalNCCMatcher,
    LearnedMatcher,
    LearnedMatcherConfig,
    LikelihoodVolume,
    MapEncoder,
    ObservationEncoder,
    PoseGrid,
    map_raster,
)
from .observation import ObservationGenerator
from .tensor_generator import (
    TensorObservationGenerator,
    TensorPoseSampler,
    TensorTrajectorySampler,
)
from .temporal import ConvGRULocalizer, FilterConfig, SE2HistogramFilter
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
    "TensorObservationGenerator",
    "TensorPoseSampler",
    "TensorTrajectorySampler",
    "generate_trajectory",
    "sample_pose",
    "Augmentation",
    "Observation",
    "Pose",
    "Trajectory",
    # matching
    "ClassicalNCCMatcher",
    "LearnedMatcher",
    "LearnedMatcherConfig",
    "LikelihoodVolume",
    "MapEncoder",
    "ObservationEncoder",
    "PoseGrid",
    "map_raster",
    # temporal
    "FilterConfig",
    "SE2HistogramFilter",
    "ConvGRULocalizer",
    # evaluation
    "BenchmarkConfig",
    "BenchmarkRunner",
    "Scenario",
    "build_map_split",
    "build_map_split_from_paths",
    "corruption_level",
    "default_scenarios",
    "ego_motion_from_poses",
    "render_sequence",
    "save_results",
]