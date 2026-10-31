"""Benchmark all BEV-to-map localization methods on identical data.

Example::

    python -m bev2loc.evaluate --learned-checkpoint checkpoints/obs.pt \\
        --rnn-checkpoint checkpoints/rnn.pt --out outputs/eval/results.json
"""

from __future__ import annotations

import argparse
import copy
from pathlib import Path

from .evaluation.protocols import build_map_split
from .evaluation.runner import (
    BenchmarkConfig,
    BenchmarkRunner,
    default_scenarios,
    save_results,
    summary_table,
)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--map", default="res/example.png")
    p.add_argument("--map-variants-dir", default="outputs/maps")
    p.add_argument("--learned-checkpoint", default=None)
    p.add_argument("--rnn-checkpoint", default=None)
    p.add_argument("--out", default="outputs/eval/results.json")
    p.add_argument("--scenarios", nargs="*", default=None)
    p.add_argument("--repeats", type=int, default=1,
                   help="independent trajectory seeds per scenario")
    p.add_argument("--seq-len", type=int, default=40)
    p.add_argument("--resolution", type=float, default=8.0)
    p.add_argument("--n-yaw", type=int, default=64)
    p.add_argument("--fov-px", type=float, default=320.0)
    p.add_argument("--obs-size", type=int, default=256)
    p.add_argument("--device", default="auto")
    p.add_argument(
        "--ego-noise",
        nargs=3,
        type=float,
        default=(0.0, 0.0, 0.0),
        metavar=("FWD", "RIGHT", "YAW"),
    )
    p.add_argument("--no-ego-motion", action="store_true")
    p.add_argument("--chunk", type=int, default=4)
    return p


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    device = args.device
    if device == "auto":
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
    split = build_map_split(args.map, out_dir=args.map_variants_dir)
    cfg = BenchmarkConfig(
        grid_resolution=args.resolution,
        n_yaw=args.n_yaw,
        fov_px=args.fov_px,
        obs_width=args.obs_size,
        obs_height=args.obs_size,
        ego_motion_noise=tuple(args.ego_noise),
        use_ego_motion=not args.no_ego_motion,
        device=device,
        learned_chunk=args.chunk,
    )
    runner = BenchmarkRunner(
        split,
        cfg,
        learned_checkpoint=args.learned_checkpoint,
        rnn_checkpoint=args.rnn_checkpoint,
    )
    scenarios = default_scenarios(seq_len=args.seq_len)
    if args.scenarios:
        wanted = set(args.scenarios)
        scenarios = [s for s in scenarios if s.name in wanted]
    if args.repeats > 1:
        expanded = []
        for sc in scenarios:
            for r in range(args.repeats):
                rep = copy.deepcopy(sc)
                rep.seed = sc.seed + 1000 * r
                expanded.append(rep)
        scenarios = expanded
    results = runner.run_all(scenarios)
    path = save_results(results, args.out)
    print(f"\nsaved {path}\n")
    print(summary_table(results))


if __name__ == "__main__":
    main()