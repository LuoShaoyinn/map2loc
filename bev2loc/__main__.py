from __future__ import annotations

import argparse


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="bev2loc", description="bev2loc observation infra")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("preview", help="render trajectory / observation previews")
    p.add_argument("--map", default="res/example.png")
    p.add_argument("--out-dir", default="outputs/previews")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--traj-steps", type=int, default=700)
    p.add_argument("--n-windows", type=int, default=12)
    p.add_argument("--obs-size", type=int, default=256)
    p.add_argument("--fov-px", type=float, default=320.0)

    args = parser.parse_args(argv)
    if args.command == "preview":
        from .visualize import save_previews
        from .config import GeometryConfig, InfraConfig

        cfg = InfraConfig(
            geometry=GeometryConfig(
                obs_width=args.obs_size,
                obs_height=args.obs_size,
                fov_px=args.fov_px,
            )
        )
        paths = save_previews(
            args.map,
            args.out_dir,
            cfg=cfg,
            seed=args.seed,
            traj_steps=args.traj_steps,
            n_windows=args.n_windows,
        )
        for name, path in paths.items():
            print(f"{name:>14}: {path}")


if __name__ == "__main__":
    main()