"""Run every method on identical trajectories / corruptions / map splits.

Methods
-------
``classical``        rotation-aware masked NCC, per frame
``learned``          learned encoders + dense correlation, per frame
``classical+filter`` F3Loc-style SE(2) histogram filter over the NCC volume
``learned+filter``   the same filter over the frozen learned volume
``rnn``              ConvGRU over the frozen learned features / volume

The observation model used by ``learned``, ``learned+filter`` and ``rnn`` is
the same frozen checkpoint, so differences between the last three are purely
temporal.
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ..config import InfraConfig
from ..dataset import _to_tensor_image
from ..matching.classical import ClassicalNCCMatcher
from ..matching.grid import PoseGrid, map_raster
from ..matching.learned import LearnedMatcher, LearnedMatcherConfig
from ..temporal.histogram_filter import FilterConfig, SE2HistogramFilter
from ..temporal.rnn import ConvGRULocalizer
from ..types import Pose
from .metrics import sequence_errors, summarize
from .protocols import (
    CorruptionLevel,
    MapSplit,
    RenderedSequence,
    corruption_level,
    ego_motion_from_poses,
    render_sequence,
)


@dataclass
class Scenario:
    name: str
    map_key: str = "train"  # "train" or a key in MapSplit.eval_maps
    region_key: str = "seen"  # "seen" / "unseen"
    corruption: str = "moderate"
    init: str = "global"  # "global" / "wrong" / "oracle" (temporal only)
    distractor_frames: int = 0
    seed: int = 0
    seq_len: int = 40
    step_stride: int = 6


def default_scenarios(*, seq_len: int = 40) -> list[Scenario]:
    return [
        Scenario("seen_clean", corruption="clean", seed=11, seq_len=seq_len),
        Scenario("seen_mild", corruption="mild", seed=12, seq_len=seq_len),
        Scenario("seen_moderate", corruption="moderate", seed=13, seq_len=seq_len),
        Scenario("seen_severe", corruption="severe", seed=14, seq_len=seq_len),
        Scenario("scratches", corruption="scratches", seed=15, seq_len=seq_len),
        Scenario("occlusion", corruption="occlusion", seed=16, seq_len=seq_len),
        Scenario("geometric", corruption="geometric", seed=17, seq_len=seq_len),
        Scenario(
            "unseen_map_rot90",
            map_key="rot90",
            corruption="moderate",
            seed=18,
            seq_len=seq_len,
        ),
        Scenario(
            "unseen_region",
            region_key="unseen",
            corruption="moderate",
            seed=19,
            seq_len=seq_len,
        ),
        Scenario(
            "wrong_global_init",
            corruption="moderate",
            init="wrong",
            seed=20,
            seq_len=seq_len,
        ),
        Scenario(
            "recovery_distractor",
            corruption="moderate",
            distractor_frames=5,
            seed=21,
            seq_len=seq_len,
        ),
    ]


@dataclass
class BenchmarkConfig:
    grid_resolution: float = 8.0
    n_yaw: int = 64
    fov_px: float = 320.0
    obs_width: int = 256
    obs_height: int = 256
    ego_motion_noise: tuple[float, float, float] = (0.0, 0.0, 0.0)
    use_ego_motion: bool = True
    device: str = "cpu"
    learned_chunk: int = 4


class BenchmarkRunner:
    def __init__(
        self,
        split: MapSplit,
        cfg: BenchmarkConfig | None = None,
        *,
        learned_checkpoint: str | Path | None = None,
        rnn_checkpoint: str | Path | None = None,
    ):
        self.split = split
        self.cfg = cfg or BenchmarkConfig()
        self.device = torch.device(self.cfg.device)
        self.learned_checkpoint = (
            Path(learned_checkpoint) if learned_checkpoint else None
        )
        self.rnn_checkpoint = Path(rnn_checkpoint) if rnn_checkpoint else None
        self._grids: dict[str, PoseGrid] = {}
        self._classical: dict[str, ClassicalNCCMatcher] = {}
        self._learned: dict[str, LearnedMatcher] = {}
        self._map_feats: dict[str, torch.Tensor] = {}
        self._rnn: dict[str, ConvGRULocalizer] = {}

    # ------------------------------------------------------------- helpers
    def grid_for(self, map_) -> PoseGrid:
        key = str(map_.path)
        if key not in self._grids:
            self._grids[key] = PoseGrid.from_map(
                map_,
                resolution=self.cfg.grid_resolution,
                n_yaw=self.cfg.n_yaw,
            )
        return self._grids[key]

    def _map_features(self, map_) -> tuple[PoseGrid, torch.Tensor]:
        grid = self.grid_for(map_)
        key = f"{map_.path}:{self.cfg.grid_resolution}:{self.cfg.n_yaw}"
        if key not in self._map_feats:
            r = map_raster(map_, grid, grayscale=False, normalize=True, device=self.device)
            matcher = self._learned_matcher(map_)
            with torch.no_grad():
                self._map_feats[key] = matcher.encode_map(r["image"])
        return grid, self._map_feats[key]

    def _classical_matcher(self, map_) -> ClassicalNCCMatcher:
        key = str(map_.path)
        if key not in self._classical:
            m = ClassicalNCCMatcher(
                self.grid_for(map_),
                fov_px=self.cfg.fov_px,
                obs_width=self.cfg.obs_width,
                obs_height=self.cfg.obs_height,
                device=self.device,
            )
            m.set_map(map_)
            self._classical[key] = m
        return self._classical[key]

    def _learned_matcher(self, map_) -> LearnedMatcher:
        if self.learned_checkpoint is None:
            raise RuntimeError("no learned checkpoint supplied")
        key = str(map_.path)
        if key not in self._learned:
            ckpt = torch.load(self.learned_checkpoint, map_location="cpu", weights_only=False)
            lcfg = LearnedMatcherConfig.from_dict(ckpt["config"])
            model = LearnedMatcher(lcfg, self.grid_for(map_))
            model.load_state_dict(ckpt["state_dict"])
            model.to(self.device).eval()
            for p in model.parameters():
                p.requires_grad_(False)
            self._learned[key] = model
        return self._learned[key]

    def _rnn_model(self, map_) -> ConvGRULocalizer:
        if self.rnn_checkpoint is None:
            raise RuntimeError("no RNN checkpoint supplied")
        key = str(map_.path)
        if key not in self._rnn:
            ckpt = torch.load(self.rnn_checkpoint, map_location="cpu", weights_only=False)
            rcfg = ckpt["config"]
            model = ConvGRULocalizer(
                self.grid_for(map_),
                feature_dim=rcfg.get("feature_dim", 32),
                hidden_ch=rcfg.get("hidden_ch", 32),
                context_ch=rcfg.get("context_ch", 4),
                latent_yaw=rcfg.get("latent_yaw", 8),
                latent_xy=rcfg.get("latent_xy", 24),
                num_layers=rcfg.get("num_layers", 1),
            )
            model.load_state_dict(ckpt["state_dict"])
            model.to(self.device).eval()
            self._rnn[key] = model
        return self._rnn[key]

    def _obs_batch(self, seq: RenderedSequence, idx: list[int], cfg: InfraConfig):
        imgs = torch.stack(
            [_to_tensor_image(seq.observations[i].image, cfg) for i in idx]
        ).to(self.device)
        masks = torch.stack(
            [
                torch.from_numpy(seq.observations[i].valid_mask.astype(np.float32))
                for i in idx
            ]
        ).to(self.device)
        return imgs, masks

    # ------------------------------------------------------------ inference
    def classical_logits(self, seq: RenderedSequence) -> torch.Tensor:
        matcher = self._classical_matcher(seq.map)
        out = []
        with torch.no_grad():
            for obs in seq.observations:
                vol = matcher.match(obs.image, obs.valid_mask)
                out.append(vol.logits.detach().cpu())
        return torch.stack(out)

    def learned_logits(
        self, seq: RenderedSequence
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return per-frame logits and observation feature maps."""
        matcher = self._learned_matcher(seq.map)
        _, map_feats = self._map_features(seq.map)
        cfg = InfraConfig()
        logits_all: list[torch.Tensor] = []
        feat_all: list[torch.Tensor] = []
        with torch.no_grad():
            for start in range(0, len(seq.observations), self.cfg.learned_chunk):
                idx = list(
                    range(
                        start,
                        min(start + self.cfg.learned_chunk, len(seq.observations)),
                    )
                )
                imgs, masks = self._obs_batch(seq, idx, cfg)
                out = matcher(imgs, map_feats, valid_mask=masks)
                logits_all.append(out["logits"].detach().cpu())
                feat_all.append(out["features"].detach().cpu())
        return torch.cat(logits_all), torch.cat(feat_all)

    # -------------------------------------------------------------- methods
    def _wrong_pose(self, map_, gt: Pose) -> Pose:
        region = map_.region("content", 0.05)
        x = region.x1 - (gt.x - region.x0)
        y = region.y1 - (gt.y - region.y0)
        return Pose(
            float(np.clip(x, region.x0, region.x1)),
            float(np.clip(y, region.y0, region.y1)),
            float((gt.yaw + np.pi + np.pi) % (2 * np.pi) - np.pi),
        )

    def _filter_sequence(
        self,
        grid: PoseGrid,
        seq: RenderedSequence,
        logits: torch.Tensor,
        *,
        init: str,
        seed: int,
    ) -> list[Pose]:
        free = None
        filt = SE2HistogramFilter(
            grid,
            FilterConfig(),
            free_mask=free,
            device=self.device,
        )
        gt0 = Pose(*[float(v) for v in seq.poses[0]])
        if init == "wrong":
            filt.set_prior_pose(self._wrong_pose(seq.map, gt0))
        elif init == "oracle":
            filt.set_prior_pose(Pose(float(seq.gt_xy[0, 0]), float(seq.gt_xy[0, 1]), gt0.yaw))
        rng = np.random.default_rng(seed + 9001)
        estimates: list[Pose] = []
        for t in range(len(seq.observations)):
            if t > 0 and self.cfg.use_ego_motion:
                e = ego_motion_from_poses(
                    seq.poses,
                    t,
                    noise=self.cfg.ego_motion_noise or None,
                    rng=rng,
                )
                filt.predict(e)
            filt.update(logits[t])
            estimates.append(filt.best())
        return estimates

    def _rnn_sequence(
        self,
        seq: RenderedSequence,
        logits: torch.Tensor,
        features: torch.Tensor,
        *,
        init: str,
    ) -> list[Pose]:
        model = self._rnn_model(seq.map)
        device = self.device
        b = 1
        hidden = model.init_hidden(b, device)
        prev = None
        if init in ("wrong", "oracle"):
            gt0 = Pose(*[float(v) for v in seq.poses[0]])
            if init == "wrong":
                p0 = self._wrong_pose(seq.map, gt0)
            else:
                p0 = Pose(float(seq.gt_xy[0, 0]), float(seq.gt_xy[0, 1]), gt0.yaw)
            prior = self._pose_logits(model.grid, p0, device)
            _, hidden = model.step(prior, None, hidden)
        estimates = []
        with torch.no_grad():
            for t in range(logits.shape[0]):
                x = logits[t : t + 1].to(device)
                f = features[t : t + 1].to(device)
                refined, hidden = model.step(x, f, hidden)
                prev = refined
                idx = int(refined.flatten().argmax())
                nx, ny = model.grid.nx, model.grid.ny
                ix = idx % nx
                iy = (idx // nx) % ny
                iyaw = idx // (nx * ny)
                estimates.append(model.grid.pose_at(iyaw, iy, ix))
        return estimates

    def _pose_logits(self, grid: PoseGrid, pose: Pose, device) -> torch.Tensor:
        coords = grid.to(device)
        d2 = (coords["x"] - pose.x) ** 2 + (coords["y"] - pose.y) ** 2
        dyaw = torch.atan2(
            torch.sin(coords["yaw"] - pose.yaw), torch.cos(coords["yaw"] - pose.yaw)
        )
        lp = -0.5 * d2 / (3.0**2) - 0.5 * dyaw**2 / (np.deg2rad(6.0) ** 2)
        lp = torch.log_softmax(lp.flatten(), dim=0).reshape(lp.shape)
        return lp[None]

    # ------------------------------------------------------------- scenario
    def _prepare_sequence(self, sc: Scenario) -> RenderedSequence:
        map_ = self.split.eval_map(sc.map_key)
        region = (
            self.split.unseen_region
            if sc.region_key == "unseen"
            else self.split.seen_region
        )
        corruption = corruption_level(sc.corruption)
        distractor_map = None
        return render_sequence(
            map_,
            corruption,
            seed=sc.seed,
            seq_len=sc.seq_len,
            step_stride=sc.step_stride,
            region=region,
            distractor_frames=sc.distractor_frames,
            distractor_map=distractor_map,
        )

    def _poses_from_logits(self, logits: torch.Tensor, grid: PoseGrid) -> list[Pose]:
        poses = []
        for t in range(logits.shape[0]):
            vol = logits[t]
            idx = int(vol.flatten().argmax())
            nx, ny = grid.nx, grid.ny
            ix = idx % nx
            iy = (idx // nx) % ny
            iyaw = idx // (nx * ny)
            poses.append(grid.pose_at(iyaw, iy, ix))
        return poses

    def run_scenario(self, sc: Scenario) -> dict[str, Any]:
        seq = self._prepare_sequence(sc)
        grid = self.grid_for(seq.map)
        results: dict[str, Any] = {
            "scenario": sc.__dict__,
            "map": str(seq.map.path),
            "corruption": sc.corruption,
            "n_frames": len(seq.observations),
            "methods": {},
        }
        gt_yaw = seq.gt_yaw
        classical = self.classical_logits(seq)
        learned = features = None
        if self.learned_checkpoint is not None:
            learned, features = self.learned_logits(seq)

        def add(name: str, estimates: list[Pose], extra: dict | None = None):
            errs = sequence_errors(estimates, seq.gt_xy, gt_yaw)
            summary = summarize(
                errs, recovery_start=int(sc.distractor_frames)
            )
            results["methods"][name] = {
                **summary,
                "per_frame_pos_px": errs["pos_px"].tolist(),
                "per_frame_yaw_deg": np.rad2deg(errs["yaw_rad"]).tolist(),
                "estimates_xy_px": np.array([e.xy for e in estimates]).tolist(),
                "estimates_yaw": np.array([e.yaw for e in estimates]).tolist(),
            }
            if extra:
                results["methods"][name].update(extra)

        add("classical", self._poses_from_logits(classical, grid))
        add(
            "classical+filter",
            self._filter_sequence(grid, seq, classical, init=sc.init, seed=sc.seed),
        )
        if learned is not None:
            add("learned", self._poses_from_logits(learned, grid))
            add(
                "learned+filter",
                self._filter_sequence(grid, seq, learned, init=sc.init, seed=sc.seed),
            )
        if self.rnn_checkpoint is not None and learned is not None:
            add(
                "rnn",
                self._rnn_sequence(seq, learned, features, init=sc.init),
            )
        return results

    def run_all(self, scenarios: list[Scenario]) -> dict[str, Any]:
        out = {
            "config": {
                "grid_resolution": self.cfg.grid_resolution,
                "n_yaw": self.cfg.n_yaw,
                "fov_px": self.cfg.fov_px,
                "obs_width": self.cfg.obs_width,
                "obs_height": self.cfg.obs_height,
                "ego_motion_noise": list(self.cfg.ego_motion_noise),
                "use_ego_motion": self.cfg.use_ego_motion,
            },
            "split": {
                "name": self.split.name,
                "metadata": self.split.metadata,
                "eval_maps": list(self.split.eval_maps),
            },
            "scenarios": [],
        }
        for sc in scenarios:
            result = self.run_scenario(sc)
            out["scenarios"].append(result)
            print(f"[{sc.name}] done", flush=True)
        return out


def save_results(results: dict[str, Any], path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(results, indent=2))
    return p


def summary_table(results: dict[str, Any]) -> str:
    lines = [
        f"{'scenario':<22}{'method':<20}{'pos_med':>9}{'pos_p90':>9}"
        f"{'yaw_med':>9}{'succ@16/10':>12}{'recovery':>10}"
    ]
    lines.append("-" * len(lines[0]))
    for sc in results["scenarios"]:
        for name, m in sc["methods"].items():
            succ = m["success"].get("pos<16px_yaw<10deg", float("nan"))
            rec = m.get("recovery_frame")
            rec_s = "-" if rec is None else str(rec)
            lines.append(
                f"{sc['scenario']['name']:<22}{name:<20}"
                f"{m['pos_px_median']:>9.1f}{m['pos_px_p90']:>9.1f}"
                f"{m['yaw_deg_median']:>9.2f}{succ:>12.3f}{rec_s:>10}"
            )
    return "\n".join(lines)