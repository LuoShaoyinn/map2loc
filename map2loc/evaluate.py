"""Benchmark the updated BEV algorithms on identical scenarios."""

from __future__ import annotations

import argparse
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
    p.add_argument("--out", default="outputs/eval/results_bev.json")
    p.add_argument("--learned-checkpoint", default=None)
    p.add_argument("--device", default="auto")
    p.add_argument("--resolution", type=float, default=8.0)
    p.add_argument("--n-yaw", type=int, default=64)
    p.add_argument("--patch-px", type=int, default=164)
    p.add_argument("--map-px-per-px", type=float, default=2.0)
    p.add_argument("--seq-len", type=int, default=40)
    p.add_argument("--repeats", type=int, default=1)
    p.add_argument("--scenarios", default=None)
    p.add_argument("--no-ego-motion", action="store_true")
    p.add_argument("--map-variants-dir", default="outputs/maps")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    device = args.device
    if device == "auto":
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
    split = build_map_split(args.map, out_dir=args.map_variants_dir)
    cfg = BenchmarkConfig(
        grid_resolution=args.resolution,
        n_yaw=args.n_yaw,
        patch_px=args.patch_px,
        map_px_per_px=args.map_px_per_px,
        device=device,
        use_ego_motion=not args.no_ego_motion,
    )
    runner = BenchmarkRunner(
        split, cfg, learned_checkpoint=args.learned_checkpoint
    )
    scenarios = default_scenarios(seq_len=args.seq_len)
    if args.scenarios:
        keep = set(args.scenarios.split(","))
        scenarios = [s for s in scenarios if s.name in keep]
    all_results = []
    for r in range(args.repeats):
        for s in scenarios:
            s.seed = s.seed + 1000 * r
        res = runner.run_all(scenarios)
        all_results.append(res)
    # Report the last repeat; keep all for inspection.
    results = all_results[-1]
    results["repeats"] = args.repeats
    path = save_results(results, args.out)
    print(f"\nwrote {path}\n")
    print(summary_table(results))
    return path


if __name__ == "__main__":
    main()