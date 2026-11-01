"""Evaluation protocols and benchmark runner."""

from .metrics import pose_errors, sequence_errors, summarize
from .protocols import (
    CorruptionLevel,
    MapSplit,
    RenderedSequence,
    all_corruption_levels,
    build_map_split,
    build_map_split_from_paths,
    corruption_level,
    ego_motion_from_poses,
    render_sequence,
)
from .runner import (
    BenchmarkConfig,
    BenchmarkRunner,
    Scenario,
    default_scenarios,
    save_results,
    summary_table,
)

__all__ = [
    "pose_errors",
    "sequence_errors",
    "summarize",
    "CorruptionLevel",
    "MapSplit",
    "RenderedSequence",
    "all_corruption_levels",
    "build_map_split",
    "build_map_split_from_paths",
    "corruption_level",
    "ego_motion_from_poses",
    "render_sequence",
    "BenchmarkConfig",
    "BenchmarkRunner",
    "Scenario",
    "default_scenarios",
    "save_results",
    "summary_table",
]