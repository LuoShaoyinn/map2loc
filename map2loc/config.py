"""Compatibility re-exports of the observation configuration."""

from data_gen.config import (
    GeometryConfig,
    InfraConfig,
    MeasurementConfig,
    PhotometricConfig,
    TrajectoryConfig,
)

__all__ = [
    "GeometryConfig",
    "PhotometricConfig",
    "MeasurementConfig",
    "TrajectoryConfig",
    "InfraConfig",
]