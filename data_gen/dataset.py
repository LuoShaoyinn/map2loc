"""Torch datasets for fixed-map localization.

Two datasets are provided:

* :class:`PoseDataset`   -- independent single-frame samples, uniformly
  distributed over the map region (good for balanced coverage).
* :class:`TrajectoryDataset` -- short smooth sequences, one per item
  (good for temporal models / sequence training).

Both are fully procedural: every sample is rendered on the fly from a seed,
so no disk storage is needed and the effective dataset is unbounded.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from .config import InfraConfig
from .map import FixedMap, Region
from .observation import ObservationGenerator
from .trajectory import generate_trajectory, sample_pose
from .types import Observation


def _seed(base: int, offset: int, index: int) -> int:
    ss = np.random.SeedSequence([int(base), int(offset), int(index)])
    return int(ss.generate_state(1, dtype=np.uint32)[0])


def _to_tensor_image(image: np.ndarray, cfg: InfraConfig) -> torch.Tensor:
    x = torch.from_numpy(np.ascontiguousarray(image)).float().div_(255.0)
    x = x.permute(2, 0, 1)  # HWC -> CHW
    mean = torch.tensor(cfg.norm_mean, dtype=torch.float32).view(3, 1, 1)
    std = torch.tensor(cfg.norm_std, dtype=torch.float32).view(3, 1, 1)
    return (x - mean) / std


def _targets(obs: Observation, map_: FixedMap) -> dict[str, torch.Tensor]:
    pos = map_.normalize_xy(obs.gt_xy)
    yaw = float(obs.pose.yaw)
    pose = obs.pose.as_array()
    return {
        "pos": torch.from_numpy(pos.astype(np.float32)),
        "pos_px": torch.from_numpy(obs.gt_xy.astype(np.float32)),
        "yaw": torch.tensor([yaw], dtype=torch.float32),
        "yaw_sincos": torch.tensor(
            [np.sin(yaw), np.cos(yaw)], dtype=torch.float32
        ),
        "pitch": torch.tensor([obs.pose.pitch], dtype=torch.float32),
        "roll": torch.tensor([obs.pose.roll], dtype=torch.float32),
        # Noisy attitude inputs available to the model.
        "yaw_input": torch.tensor([obs.yaw_input], dtype=torch.float32),
        "pitch_input": torch.tensor([obs.pitch_input], dtype=torch.float32),
        "yaw_input_sincos": torch.tensor(
            [np.sin(obs.yaw_input), np.cos(obs.yaw_input)], dtype=torch.float32
        ),
        "pose_px": torch.from_numpy(pose.astype(np.float32)),
        "center_px": torch.from_numpy(obs.center_xy.astype(np.float32)),
        "corners": torch.from_numpy(obs.corners.astype(np.float32)),
    }


class _BaseDataset(Dataset):
    def __init__(
        self,
        map_path: str | Path | FixedMap,
        cfg: InfraConfig | None = None,
        *,
        length: int,
        seed: int,
        split: str,
        region: Region | None,
        geometry: bool,
        photometric: bool,
    ):
        self.cfg = cfg or InfraConfig()
        self.geometry = bool(geometry)
        self.photometric = bool(photometric)
        self.map = (
            map_path if isinstance(map_path, FixedMap) else FixedMap(map_path)
        )
        self.length = int(length)
        self.seed = int(seed)
        self.split = split
        self.split_offset = {"train": 0, "val": 1_000_003, "test": 2_000_003}.get(
            split, 0
        )
        self.region = region or self.map.region(
            self.cfg.trajectory.region, self.cfg.trajectory.region_shrink
        )
        self.generator = ObservationGenerator(
            self.map,
            self.cfg.geometry,
            self.cfg.photometric,
            self.cfg.measurement,
        )

    def __len__(self) -> int:
        return self.length


class PoseDataset(_BaseDataset):
    """Single-frame localization samples."""

    def __init__(
        self,
        map_path: str | Path | FixedMap,
        cfg: InfraConfig | None = None,
        *,
        length: int = 10_000,
        seed: int = 0,
        split: str = "train",
        region: Region | None = None,
        geometry: bool = True,
        photometric: bool = True,
    ):
        super().__init__(
            map_path,
            cfg,
            length=length,
            seed=seed,
            split=split,
            region=region,
            geometry=geometry,
            photometric=photometric,
        )

    def sample(self, index: int) -> Observation:
        rng = np.random.default_rng(
            _seed(self.seed, self.split_offset, index)
        )
        pose = sample_pose(
            self.map,
            rng,
            region=self.region,
            cfg=self.cfg.trajectory,
        )
        return self.generator.generate(
            pose, rng, geometry=self.geometry, photometric=self.photometric
        )

    def __getitem__(self, index: int) -> dict[str, Any]:
        obs = self.sample(index)
        item: dict[str, Any] = {
            "image": _to_tensor_image(obs.image, self.cfg),
            "valid_mask": torch.from_numpy(obs.valid_mask.astype(np.float32)),
            "index": torch.tensor(index, dtype=torch.long),
        }
        item.update(_targets(obs, self.map))
        item["meta"] = {
            "split": self.split,
            "aug": obs.aug.as_dict(),
            "photo": obs.photo,
            "camera": {
                "height_px": self.cfg.geometry.height_px,
                "footprint_px": self.cfg.geometry.footprint_px,
                "obs_size": (self.generator.width, self.generator.height),
            },
            "map_size": (self.map.width, self.map.height),
        }
        return item


class TrajectoryDataset(_BaseDataset):
    """Short smooth sequences of observations."""

    def __init__(
        self,
        map_path: str | Path | FixedMap,
        cfg: InfraConfig | None = None,
        *,
        length: int = 2_000,
        seq_len: int = 8,
        step_stride: int = 8,
        seed: int = 0,
        split: str = "train",
        region: Region | None = None,
        geometry: bool = True,
        photometric: bool = True,
    ):
        super().__init__(
            map_path,
            cfg,
            length=length,
            seed=seed,
            split=split,
            region=region,
            geometry=geometry,
            photometric=photometric,
        )
        self.seq_len = int(seq_len)
        self.step_stride = int(step_stride)

    def __getitem__(self, index: int) -> dict[str, Any]:
        rng = np.random.default_rng(
            _seed(self.seed, self.split_offset, index)
        )
        n_steps = (self.seq_len - 1) * self.step_stride + 1
        traj = generate_trajectory(
            self.map,
            self.cfg.trajectory,
            n_steps=n_steps,
            rng=rng,
            region=self.region,
        )
        pose_list = traj[:: self.step_stride][: self.seq_len]
        obs_list = [
            self.generator.generate(
                p, rng, geometry=self.geometry, photometric=self.photometric
            )
            for p in pose_list
        ]
        images = torch.stack(
            [_to_tensor_image(o.image, self.cfg) for o in obs_list], dim=0
        )
        valid_masks = torch.stack(
            [torch.from_numpy(o.valid_mask.astype(np.float32)) for o in obs_list],
            dim=0,
        )
        targets = [_targets(o, self.map) for o in obs_list]

        return {
            "images": images,  # (T, 3, H, W)
            "valid_masks": valid_masks,  # (T, H, W)
            "pos": torch.stack([t["pos"] for t in targets]),  # (T, 2)
            "pos_px": torch.stack([t["pos_px"] for t in targets]),
            "yaw": torch.stack([t["yaw"] for t in targets]),
            "yaw_sincos": torch.stack([t["yaw_sincos"] for t in targets]),
            "pitch": torch.stack([t["pitch"] for t in targets]),
            "roll": torch.stack([t["roll"] for t in targets]),
            "yaw_input": torch.stack([t["yaw_input"] for t in targets]),
            "pitch_input": torch.stack([t["pitch_input"] for t in targets]),
            "yaw_input_sincos": torch.stack(
                [t["yaw_input_sincos"] for t in targets]
            ),
            "pose_px": torch.stack([t["pose_px"] for t in targets]),
            "center_px": torch.stack([t["center_px"] for t in targets]),
            "corners": torch.stack([t["corners"] for t in targets]),
            "index": torch.tensor(index, dtype=torch.long),
            "meta": {
                "split": self.split,
                "region": traj.region,
                "aug": [o.aug.as_dict() for o in obs_list],
                "photo": [o.photo for o in obs_list],
                "camera": {
                    "height_px": self.cfg.geometry.height_px,
                    "footprint_px": self.cfg.geometry.footprint_px,
                    "obs_size": (self.generator.width, self.generator.height),
                },
                "map_size": (self.map.width, self.map.height),
            },
        }


def localization_collate(batch: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Collate dicts while keeping the python ``meta`` entries as a list."""
    out: dict[str, Any] = {}
    keys = batch[0].keys()
    for key in keys:
        values = [b[key] for b in batch]
        if isinstance(values[0], torch.Tensor):
            out[key] = torch.stack(values, dim=0)
        else:
            out[key] = values
    return out