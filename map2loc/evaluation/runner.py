"""Run every method on identical trajectories / corruptions / map splits.

Methods
-------
``classical``        masked NCC on attitude-rectified BEV templates
``learned``          learned BEV encoder + dense correlation
``classical+filter`` position histogram filter over the NCC likelihood
``learned+filter``   the same filter over the frozen learned likelihood

All per-frame methods consume the same rectified BEV patches, derived from the
same perspective observation, measured attitude and validity mask.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ..bev import SCALE_HYPOTHESES, BEVSpec, centre_offset, rectify_numpy
from ..matching.classical import ClassicalNCCMatcher
from ..matching.grid import PoseGrid, map_raster
from ..matching.learned import LearnedMatcher, LearnedMatcherConfig, merge_residual_yaw
from ..temporal.histogram_filter import FilterConfig, XYHistogramFilter
from ..types import Pose
from .metrics import sequence_errors, summarize
from .protocols import (
    MapSplit,
    RenderedSequence,
    corruption_level,
    ego_motion_from_poses,
    render_sequence,
)


@dataclass
class Scenario:
    name: str
    map_key: str = "train"
    region_key: str = "seen"
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
        Scenario("pixel_dropout", corruption="pixel", seed=16, seq_len=seq_len),
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
    patch_px: int = 164
    map_px_per_px: float = 2.0
    height_px: float = 180.0
    ego_motion_noise: tuple[float, float, float] = (0.0, 0.0, 0.0)
    use_ego_motion: bool = True
    device: str = "cpu"
    learned_chunk: int = 2
    scales: tuple[float, ...] = SCALE_HYPOTHESES
    # Measured-attitude uncertainty is 5 deg.  Pitch has the largest effect
    # through the image-centre lever arm, so search a few residual-pitch
    # hypotheses and marginalise them.  Roll is largely absorbed by the
    # residual-yaw template rotation.
    pitch_residuals_deg: tuple[float, ...] = (-5.0, 0.0, 5.0)
    roll_residuals_deg: tuple[float, ...] = (0.0,)
    scale_coverage_power: float = 0.5
    filter_sigma_xy_px: float = 3.0
    filter_global_mix: float = 0.05


class BenchmarkRunner:
    def __init__(
        self,
        split: MapSplit,
        cfg: BenchmarkConfig | None = None,
        *,
        learned_checkpoint: str | Path | None = None,
    ):
        self.split = split
        self.cfg = cfg or BenchmarkConfig()
        self.device = torch.device(self.cfg.device)
        self.learned_checkpoint = (
            Path(learned_checkpoint) if learned_checkpoint else None
        )
        self.bev = BEVSpec(self.cfg.patch_px, self.cfg.map_px_per_px)
        self._grids: dict[str, PoseGrid] = {}
        self._classical: dict[str, ClassicalNCCMatcher] = {}
        self._learned: dict[str, LearnedMatcher] = {}
        self._map_feats: dict[str, torch.Tensor] = {}

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
            r = map_raster(
                map_, grid, grayscale=False, normalize=True, device=self.device
            )
            matcher = self._learned_matcher(map_)
            with torch.no_grad():
                self._map_feats[key] = matcher.encode_map(r["image"])
        return grid, self._map_feats[key]

    def _classical_matcher(self, map_) -> ClassicalNCCMatcher:
        key = str(map_.path)
        if key not in self._classical:
            m = ClassicalNCCMatcher(
                self.grid_for(map_),
                patch_px=self.cfg.patch_px,
                map_px_per_px=self.cfg.map_px_per_px,
                scale_coverage_power=self.cfg.scale_coverage_power,
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
            ckpt = torch.load(
                self.learned_checkpoint, map_location="cpu", weights_only=False
            )
            lcfg = LearnedMatcherConfig.from_dict(ckpt["config"])
            if (
                lcfg.patch_px != self.cfg.patch_px
                or abs(lcfg.map_px_per_px - self.cfg.map_px_per_px) > 1e-6
                or abs(lcfg.grid_resolution - self.cfg.grid_resolution) > 1e-6
                or lcfg.n_yaw != self.cfg.n_yaw
            ):
                raise ValueError(
                    "evaluation geometry must match the learned checkpoint: "
                    f"checkpoint patch/res/yaw="
                    f"{lcfg.patch_px}/{lcfg.grid_resolution}/{lcfg.n_yaw}, "
                    f"benchmark={self.cfg.patch_px}/"
                    f"{self.cfg.grid_resolution}/{self.cfg.n_yaw}"
                )
            model = LearnedMatcher(lcfg, self.grid_for(map_))
            model.load_state_dict(ckpt["state_dict"])
            model.to(self.device).eval()
            for p in model.parameters():
                p.requires_grad_(False)
            self._learned[key] = model
        return self._learned[key]

    # ---------------------------------------------------------- observation
    def bev_hypotheses(
        self,
        seq: RenderedSequence,
        idx: int,
        *,
        pitch_offset: float = 0.0,
        roll_offset: float = 0.0,
    ):
        """Rectify observation ``idx`` at every focal-scale hypothesis."""
        o = seq.observations[idx]
        patches, masks = [], []
        for s in self.cfg.scales:
            p, m = rectify_numpy(
                o.image,
                o.valid_mask,
                o.yaw_input,
                o.pitch_input + pitch_offset,
                o.roll_input + roll_offset,
                seq_geometry(o),
                s,
                self.bev,
            )
            patches.append(p)
            masks.append(m)
        return np.stack(patches), np.stack(masks)

    def _shift_to_camera(
        self,
        vol,
        obs,
        *,
        pitch_offset: float = 0.0,
        roll_offset: float = 0.0,
    ) -> torch.Tensor:
        off = centre_offset(
            obs.yaw_input, obs.pitch_input + pitch_offset, self.cfg.height_px
        )
        xy = vol.xy_logits().detach().cpu()
        if xy.ndim == 3:
            xy = xy[0]
        from ..matching.grid import shift_map

        return shift_map(xy[None], vol.grid, -float(off[0]), -float(off[1]))[0]

    def attitude_hypotheses(self):
        """``(dpitch_rad, droll_rad, log_prior)`` residual hypotheses."""
        import math

        out = []
        for dp in self.cfg.pitch_residuals_deg:
            for dr in self.cfg.roll_residuals_deg:
                # Gaussian prior with the same 5 deg sigma as the measurement.
                lp = -0.5 * ((dp / 5.0) ** 2 + (dr / 5.0) ** 2)
                out.append((math.radians(dp), math.radians(dr), lp))
        return out

    # ------------------------------------------------------------ inference
    def classical_logits(self, seq: RenderedSequence) -> torch.Tensor:
        matcher = self._classical_matcher(seq.map)
        out = []
        hyps = self.attitude_hypotheses()
        with torch.no_grad():
            for i in range(len(seq.observations)):
                o = seq.observations[i]
                per_hyp = []
                for dp, dr, lp in hyps:
                    patches, masks = self.bev_hypotheses(
                        seq, i, pitch_offset=dp, roll_offset=dr
                    )
                    vol = matcher.match(patches, masks, yaw=o.yaw_input)
                    per_hyp.append(
                        self._shift_to_camera(
                            vol, o, pitch_offset=dp, roll_offset=dr
                        )
                        + lp
                    )
                out.append(torch.logsumexp(torch.stack(per_hyp), dim=0))
        return torch.stack(out)

    def learned_logits(
        self, seq: RenderedSequence
    ) -> tuple[torch.Tensor, torch.Tensor]:
        matcher = self._learned_matcher(seq.map)
        _, map_feats = self._map_features(seq.map)
        grid = self.grid_for(seq.map)
        logits: list[torch.Tensor] = []
        feats: list[torch.Tensor] = []
        hyps = self.attitude_hypotheses()
        with torch.no_grad():
            for i in range(len(seq.observations)):
                o = seq.observations[i]
                per_hyp = []
                for h, (dp, dr, lp) in enumerate(hyps):
                    patches, masks = self.bev_hypotheses(
                        seq, i, pitch_offset=dp, roll_offset=dr
                    )
                    bev = (
                        torch.from_numpy(patches)
                        .permute(0, 3, 1, 2)
                        .float()
                        .to(self.device)
                        / 127.5
                        - 1.0
                    )[None]  # (1,S,3,P,P)
                    vmask = (
                        torch.from_numpy(masks).float().to(self.device)[None]
                    )
                    out = matcher(bev, map_feats, valid_mask=vmask)
                    yawl = torch.tensor(
                        [o.yaw_input], device=self.device
                    )
                    vol = merge_residual_yaw(
                        out["scores"],
                        grid,
                        yawl,
                        grid.dyaw,
                    )
                    per_hyp.append(
                        self._shift_to_camera(
                            vol, o, pitch_offset=dp, roll_offset=dr
                        )
                        + lp
                    )
                    if h == 0:
                        feats.append(out["features"].detach().cpu())
                logits.append(torch.logsumexp(torch.stack(per_hyp), dim=0))
        return torch.stack(logits), torch.cat(feats)

    # -------------------------------------------------------------- methods
    def _wrong_pose(self, map_, gt: Pose) -> Pose:
        region = map_.region("content", 0.05)
        x = region.x1 - (gt.x - region.x0)
        y = region.y1 - (gt.y - region.y0)
        return Pose(
            float(np.clip(x, region.x0, region.x1)),
            float(np.clip(y, region.y0, region.y1)),
            float(gt.yaw + np.pi),
        )

    def filter_volumes(
        self,
        grid: PoseGrid,
        seq: RenderedSequence,
        logits: torch.Tensor,
        *,
        init: str,
        seed: int,
        return_posteriors: bool = False,
    ) -> list[Pose] | tuple[list[Pose], list[torch.Tensor]]:
        """Logits are already position marginals over camera xy.

        With ``return_posteriors`` the per-frame ``(ny, nx)`` posterior
        probabilities are returned as well, which is what the replay renderer
        draws on the map.
        """
        filt = XYHistogramFilter(
            grid,
            FilterConfig(
                sigma_xy_px=self.cfg.filter_sigma_xy_px,
                global_mix=self.cfg.filter_global_mix,
            ),
            device=self.device,
        )
        gt0 = Pose(*[float(v) for v in seq.poses[0]])
        if init == "wrong":
            filt.set_prior_pose(
                self._wrong_pose(seq.map, gt0), sigma_xy=200.0
            )
        elif init == "oracle":
            filt.set_prior_pose(
                Pose(float(seq.gt_xy[0, 0]), float(seq.gt_xy[0, 1]), gt0.yaw)
            )
        rng = np.random.default_rng(seed + 9001)
        estimates: list[Pose] = []
        posteriors: list[torch.Tensor] = []
        for t in range(len(seq.observations)):
            if t > 0 and self.cfg.use_ego_motion:
                e = ego_motion_from_poses(
                    seq.poses,
                    t,
                    noise=self.cfg.ego_motion_noise or None,
                    rng=rng,
                )
                filt.predict(e, yaw=float(seq.observations[t].yaw_input))
            else:
                filt._yaw = float(seq.observations[t].yaw_input)
            filt.update(logits[t])
            estimates.append(filt.best())
            if return_posteriors:
                posteriors.append(filt.marginal_xy().detach().cpu())
        if return_posteriors:
            return estimates, posteriors
        return estimates

    def _poses_from_logits(
        self, logits: torch.Tensor, grid: PoseGrid, yaws=None
    ) -> list[Pose]:
        poses = []
        for t in range(logits.shape[0]):
            xy = logits[t]
            idx = int(xy.flatten().argmax())
            ix = idx % grid.nx
            iy = idx // grid.nx
            xy_px = grid.xy_at(iy, ix)
            yaw = 0.0 if yaws is None else float(yaws[t])
            poses.append(Pose(float(xy_px[0]), float(xy_px[1]), yaw))
        return poses

    # ------------------------------------------------------------- scenario
    def _prepare_sequence(self, sc: Scenario) -> RenderedSequence:
        map_ = self.split.eval_map(sc.map_key)
        if sc.map_key in ("train", "seen") and sc.region_key == "unseen":
            region = self.split.unseen_region
        elif sc.map_key in ("train", "seen"):
            region = self.split.seen_region
        else:
            region = None
        corruption = corruption_level(sc.corruption)
        return render_sequence(
            map_,
            corruption,
            seed=sc.seed,
            seq_len=sc.seq_len,
            step_stride=sc.step_stride,
            region=region,
            distractor_frames=sc.distractor_frames,
        )

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
        classical = self.classical_logits(seq)
        learned = None
        if self.learned_checkpoint is not None:
            learned, _ = self.learned_logits(seq)

        def add(name: str, estimates: list[Pose], extra: dict | None = None):
            gt_yaw = np.asarray(seq.poses[:, 2], dtype=np.float32)
            errs = sequence_errors(estimates, seq.gt_xy, gt_yaw)
            summary = summarize(errs, recovery_start=int(sc.distractor_frames))
            results["methods"][name] = {
                **summary,
                "per_frame_pos_px": errs["pos_px"].tolist(),
                "per_frame_yaw_deg": np.rad2deg(errs["yaw_rad"]).tolist(),
                "estimates_xy_px": np.array([e.xy for e in estimates]).tolist(),
            }
            if extra:
                results["methods"][name].update(extra)

        yaw_inputs = np.array(
            [o.yaw_input for o in seq.observations], dtype=np.float32
        )
        add("classical", self._poses_from_logits(classical, grid, yaw_inputs))
        add(
            "classical+filter",
            self.filter_volumes(grid, seq, classical, init=sc.init, seed=sc.seed),
        )
        if learned is not None:
            add("learned", self._poses_from_logits(learned, grid, yaw_inputs))
            add(
                "learned+filter",
                self.filter_volumes(grid, seq, learned, init=sc.init, seed=sc.seed),
            )
        return results

    def run_all(self, scenarios: list[Scenario]) -> dict[str, Any]:
        out = {
            "config": {
                "grid_resolution": self.cfg.grid_resolution,
                "n_yaw": self.cfg.n_yaw,
                "patch_px": self.cfg.patch_px,
                "map_px_per_px": self.cfg.map_px_per_px,
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


def seq_geometry(obs) -> Any:
    """Geometry matching the observation's rendered size."""
    from data_gen.config import GeometryConfig

    return GeometryConfig(
        obs_width=int(obs.image.shape[1]),
        obs_height=int(obs.image.shape[0]),
    )


def save_results(results: dict[str, Any], path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(results, indent=2))
    return p


def summary_table(results: dict[str, Any]) -> str:
    lines = [
        f"{'scenario':<22}{'method':<20}{'pos_med':>9}{'pos_p90':>9}"
        f"{'succ@16/10':>12}{'recovery':>10}"
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
                f"{succ:>12.3f}{rec_s:>10}"
            )
    return "\n".join(lines)