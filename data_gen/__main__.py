from __future__ import annotations

import argparse


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="data_gen", description="data_gen camera observation infra"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("preview", help="render camera / validity previews")
    p.add_argument("--map", default="res/example.png")
    p.add_argument("--out-dir", default="outputs/previews")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--traj-steps", type=int, default=700)
    p.add_argument("--n-windows", type=int, default=12)
    p.add_argument("--obs-size", type=int, default=256)
    p.add_argument("--height", type=float, default=180.0)
    p.add_argument("--footprint", type=float, default=120.0)

    p = sub.add_parser("video", help="render an example observation video")
    p.add_argument("--map", default="res/example.png")
    p.add_argument("--out", default="outputs/videos/data_gen_example.mp4")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--traj-steps", type=int, default=900)
    p.add_argument("--n-frames", type=int, default=300)
    p.add_argument("--fps", type=float, default=25.0)
    p.add_argument("--cell", type=int, default=256)
    p.add_argument("--obs-size", type=int, default=256)
    p.add_argument("--height", type=float, default=180.0)
    p.add_argument("--footprint", type=float, default=120.0)
    p.add_argument("--no-poster", action="store_true")

    args = parser.parse_args(argv)
    if args.command == "preview":
        from .visualize import save_previews
        from .config import GeometryConfig, InfraConfig

        cfg = InfraConfig(
            geometry=GeometryConfig(
                obs_width=args.obs_size,
                obs_height=args.obs_size,
                height_px=args.height,
                footprint_px=args.footprint,
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
            print(f"{name:>16}: {path}")
    elif args.command == "video":
        from .video import render_video
        from .config import GeometryConfig, InfraConfig

        cfg = InfraConfig(
            geometry=GeometryConfig(
                obs_width=args.obs_size,
                obs_height=args.obs_size,
                height_px=args.height,
                footprint_px=args.footprint,
            )
        )
        path = render_video(
            args.map,
            args.out,
            cfg=cfg,
            seed=args.seed,
            traj_steps=args.traj_steps,
            n_frames=args.n_frames,
            fps=args.fps,
            cell=args.cell,
            poster=not args.no_poster,
        )
        print(path)


if __name__ == "__main__":
    main()