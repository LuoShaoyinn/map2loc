"""Visual sanity checks for the observation / trajectory infra.

Usage:
    python -m map2loc.visualize --map res/example.png --out-dir outputs/previews
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from .config import GeometryConfig, InfraConfig, PhotometricConfig, TrajectoryConfig
from .map import FixedMap
from .observation import ObservationGenerator
from .trajectory import generate_trajectory
from .types import Observation, Pose

# RGB palette
_PALETTE = [
    (255, 170, 0),
    (0, 200, 255),
    (255, 80, 80),
    (200, 120, 255),
    (120, 255, 120),
    (255, 220, 0),
]


def _label_panel(
    image: np.ndarray,
    title: str,
    subtitle: str = "",
    *,
    bg: int = 255,
    fg: int = 20,
) -> np.ndarray:
    """Put an image on a light panel with a caption below it."""
    h, w = image.shape[:2]
    bar = 34 if subtitle else 22
    panel = np.full((h + bar, w, 3), bg, dtype=np.uint8)
    panel[:h] = image
    cv2.putText(
        panel,
        title,
        (6, h + 16),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (fg, fg, fg),
        1,
        cv2.LINE_AA,
    )
    if subtitle:
        cv2.putText(
            panel,
            subtitle,
            (6, h + 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.36,
            (90, 90, 90),
            1,
            cv2.LINE_AA,
        )
    return panel


def montage(
    images: list[np.ndarray],
    titles: list[str],
    subtitles: list[str] | None = None,
    *,
    cols: int = 4,
    cell: int = 256,
    gap: int = 8,
) -> np.ndarray:
    subtitles = subtitles or [""] * len(images)
    panels = []
    for img, title, sub in zip(images, titles, subtitles):
        img = cv2.resize(img, (cell, cell), interpolation=cv2.INTER_AREA)
        panels.append(_label_panel(img, title, sub))
    ph, pw = panels[0].shape[:2]
    rows = int(np.ceil(len(panels) / cols))
    out = np.full(
        (rows * ph + (rows + 1) * gap, cols * pw + (cols + 1) * gap, 3),
        245,
        dtype=np.uint8,
    )
    for i, panel in enumerate(panels):
        r, c = divmod(i, cols)
        y = gap + r * (ph + gap)
        x = gap + c * (pw + gap)
        out[y : y + ph, x : x + pw] = panel
    return out


def draw_trajectory(
    map_: FixedMap,
    traj,
    observations: list[Observation] | None = None,
) -> np.ndarray:
    img = map_.image.copy()
    pts = traj.poses[:, :2].astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(img, [pts], False, (255, 120, 0), 3, cv2.LINE_AA)

    # direction markers
    step = max(1, len(traj) // 20)
    for i in range(0, len(traj), step):
        x, y, yaw = traj.poses[i]
        p0 = (int(x), int(y))
        p1 = (int(x + 18 * np.cos(yaw)), int(y + 18 * np.sin(yaw)))
        cv2.arrowedLine(img, p0, p1, (255, 255, 0), 1, cv2.LINE_AA, tipLength=0.35)

    sx, sy, _ = traj.poses[0]
    cv2.circle(img, (int(sx), int(sy)), 7, (0, 255, 0), -1, cv2.LINE_AA)
    ex, ey, _ = traj.poses[-1]
    cv2.circle(img, (int(ex), int(ey)), 7, (0, 0, 255), -1, cv2.LINE_AA)

    if observations:
        for i, obs in enumerate(observations):
            color = _PALETTE[i % len(_PALETTE)]
            poly = np.round(obs.corners).astype(np.int32).reshape(-1, 1, 2)
            cv2.polylines(img, [poly], True, color, 2, cv2.LINE_AA)
            gx, gy = obs.gt_xy
            cv2.circle(img, (int(gx), int(gy)), 5, color, -1, cv2.LINE_AA)
            hx = gx + 22 * np.cos(obs.pose.yaw)
            hy = gy + 22 * np.sin(obs.pose.yaw)
            cv2.line(
                img,
                (int(gx), int(gy)),
                (int(hx), int(hy)),
                color,
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                img,
                str(i),
                (int(gx) + 7, int(gy) - 7),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2,
                cv2.LINE_AA,
            )
    return img


def save_previews(
    map_path: str | Path,
    out_dir: str | Path,
    *,
    cfg: InfraConfig | None = None,
    seed: int = 0,
    traj_steps: int = 300,
    n_windows: int = 16,
) -> dict[str, Path]:
    cfg = cfg or InfraConfig()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    map_ = FixedMap(map_path)
    gen = ObservationGenerator(map_, cfg.geometry, cfg.photometric)
    rng = np.random.default_rng(seed)

    traj = generate_trajectory(map_, cfg.trajectory, n_steps=traj_steps, rng=rng)
    idx = np.linspace(0, traj_steps - 1, n_windows).round().astype(int)
    obs_list = [gen.generate(traj[int(i)], rng) for i in idx]

    paths: dict[str, Path] = {}

    overlay = draw_trajectory(map_, traj, obs_list)
    p = out_dir / "preview_trajectory.png"
    cv2.imwrite(str(p), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    paths["trajectory"] = p

    # observation grid (full corruption)
    images, titles, subs = [], [], []
    for i, obs in zip(idx, obs_list):
        disp = obs.image.copy()
        c = disp.shape[1] // 2
        cv2.drawMarker(
            disp,
            (c, c),
            (255, 60, 60),
            markerType=cv2.MARKER_CROSS,
            markerSize=10,
            thickness=1,
        )
        images.append(disp)
        titles.append(f"t={int(i)}")
        yaw_deg = np.rad2deg((obs.pose.yaw + np.pi) % (2 * np.pi) - np.pi)
        subs.append(
            f"gt=({obs.gt_xy[0]:.0f},{obs.gt_xy[1]:.0f}) yaw={yaw_deg:.0f}deg"
        )
    grid = montage(images, titles, subs, cols=4, cell=cfg.geometry.obs_width)
    p = out_dir / "preview_observations.png"
    cv2.imwrite(str(p), cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))
    paths["observations"] = p

    # augmentation stages for one frame
    pose = traj[int(idx[len(idx) // 2])]
    clean = gen.generate(
        pose, np.random.default_rng(seed + 1), geometry=False, photometric=False
    )
    geom = gen.generate(
        pose, np.random.default_rng(seed + 2), geometry=True, photometric=False
    )
    # prefer a frame that actually contains a visible scratch for the demo
    full = None
    for k in range(8):
        cand = gen.generate(
            pose,
            np.random.default_rng(seed + 4 + k),
            aug=geom.aug,
            geometry=True,
            photometric=True,
        )
        if full is None or cand.photo.get("n_scratch", 0) > full.photo.get(
            "n_scratch", 0
        ):
            full = cand
    stage_imgs = [clean.image, geom.image, full.image]
    stage_titles = ["1. nominal crop", "2. + scale/shear/perspective", "3. + noise/scratch/JPEG"]
    stage_subs = [
        f"pose=({pose.x:.0f},{pose.y:.0f})",
        f"zoom=({geom.aug.scale_x:.2f},{geom.aug.scale_y:.2f}) "
        f"shear=({geom.aug.shear_x:.2f},{geom.aug.shear_y:.2f})",
        f"sigma={full.photo.get('gauss_sigma', 0):.1f} "
        f"scratches={full.photo.get('n_scratch', 0)}",
    ]
    for arr in stage_imgs:
        cv2.drawMarker(
            arr,
            (arr.shape[1] // 2, arr.shape[0] // 2),
            (255, 60, 60),
            markerType=cv2.MARKER_CROSS,
            markerSize=12,
            thickness=1,
        )
    p = out_dir / "preview_augmentation_stages.png"
    cv2.imwrite(str(p), cv2.cvtColor(montage(stage_imgs, stage_titles, stage_subs, cols=3), cv2.COLOR_RGB2BGR))
    paths["augmentation"] = p

    # clean geometry-only grid to inspect crop coverage
    g_imgs, g_titles, g_subs = [], [], []
    for i, obs in zip(idx, obs_list):
        geom_only = gen.generate(
            traj[int(i)],
            np.random.default_rng(int(i) + seed + 100),
            geometry=True,
            photometric=False,
        )
        g_imgs.append(geom_only.image)
        g_titles.append(f"t={int(i)}")
        g_subs.append(
            f"zoom=({geom_only.aug.scale_x:.2f},{geom_only.aug.scale_y:.2f}) "
            f"shear=({geom_only.aug.shear_x:+.2f},{geom_only.aug.shear_y:+.2f})"
        )
    p = out_dir / "preview_geometry.png"
    cv2.imwrite(
        str(p),
        cv2.cvtColor(
            montage(g_imgs, g_titles, g_subs, cols=4, cell=cfg.geometry.obs_width),
            cv2.COLOR_RGB2BGR,
        ),
    )
    paths["geometry"] = p

    return paths


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--map", default="res/example.png")
    parser.add_argument("--out-dir", default="outputs/previews")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--traj-steps", type=int, default=700)
    parser.add_argument("--n-windows", type=int, default=12)
    parser.add_argument("--obs-size", type=int, default=256)
    parser.add_argument("--fov-px", type=float, default=320.0)
    args = parser.parse_args(argv)

    cfg = InfraConfig(
        geometry=GeometryConfig(obs_width=args.obs_size, obs_height=args.obs_size, fov_px=args.fov_px),
        photometric=PhotometricConfig(),
        trajectory=TrajectoryConfig(),
    )
    paths = save_previews(
        args.map,
        args.out_dir,
        cfg=cfg,
        seed=args.seed,
        traj_steps=args.traj_steps,
        n_windows=args.n_windows,
    )
    for name, p in paths.items():
        print(f"{name:>14}: {p}")


if __name__ == "__main__":
    main()