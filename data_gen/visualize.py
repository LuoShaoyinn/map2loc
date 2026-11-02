"""Visual sanity checks for the camera / noise infra.

Usage:
    python -m data_gen.visualize --map res/example.png --out-dir outputs/previews
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from .augment import draw_mask, overlay_invalid
from .camera import apply_homography, render_validity
from .config import GeometryConfig, InfraConfig, PhotometricConfig, TrajectoryConfig
from .map import FixedMap
from .observation import ObservationGenerator
from .trajectory import generate_trajectory
from .types import Observation

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


def _draw_view_marker(
    image: np.ndarray, obs: Observation, generator: ObservationGenerator
) -> None:
    """Draw the view centre and the camera's ground point."""
    h, w = image.shape[:2]
    cv2.drawMarker(
        image,
        (w // 2, h // 2),
        (255, 60, 60),
        markerType=cv2.MARKER_CROSS,
        markerSize=12,
        thickness=1,
    )
    h_map_to_view = generator.map_to_view(obs.pose, obs.aug.scale)
    cam_px = apply_homography(h_map_to_view, obs.gt_xy[None])[0]
    if 0 <= cam_px[0] < w and 0 <= cam_px[1] < h:
        cv2.circle(
            image,
            (int(round(cam_px[0])), int(round(cam_px[1]))),
            4,
            (0, 255, 0),
            -1,
            cv2.LINE_AA,
        )


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
        x, y, yaw = traj.poses[i, :3]
        p0 = (int(x), int(y))
        p1 = (int(x + 18 * np.cos(yaw)), int(y + 18 * np.sin(yaw)))
        cv2.arrowedLine(img, p0, p1, (255, 255, 0), 1, cv2.LINE_AA, tipLength=0.35)

    sx, sy, _ = traj.poses[0, :3]
    cv2.circle(img, (int(sx), int(sy)), 7, (0, 255, 0), -1, cv2.LINE_AA)
    ex, ey, _ = traj.poses[-1, :3]
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
    gen = ObservationGenerator(
        map_, cfg.geometry, cfg.photometric, cfg.measurement
    )
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
        _draw_view_marker(disp, obs, gen)
        images.append(disp)
        titles.append(f"t={int(i)}")
        yaw_deg = np.rad2deg((obs.pose.yaw + np.pi) % (2 * np.pi) - np.pi)
        subs.append(
            f"gt=({obs.gt_xy[0]:.0f},{obs.gt_xy[1]:.0f}) "
            f"yaw={yaw_deg:.0f} pitch={np.rad2deg(obs.pose.pitch):.1f}"
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
    full = gen.generate(
        pose,
        np.random.default_rng(seed + 3),
        geometry=True,
        photometric=True,
        aug=geom.aug,
    )
    stage_imgs = [clean.image, geom.image, full.image]
    stage_titles = ["1. nominal camera", "2. + zoom/head jitter", "3. + noise / invalid"]
    stage_subs = [
        f"pose=({pose.x:.0f},{pose.y:.0f}) pitch={np.rad2deg(pose.pitch):.1f}",
        f"scale={geom.aug.scale:.2f} "
        f"head=({np.rad2deg(geom.aug.head_pitch):+.1f},"
        f"{np.rad2deg(geom.aug.head_yaw):+.1f})",
        f"sigma={full.photo.get('gauss_sigma', 0):.1f} "
        f"invalid={full.photo.get('invalid_frac', 0):.3f}",
    ]
    for arr, stage in zip(stage_imgs, (clean, geom, full)):
        _draw_view_marker(arr, stage, gen)
    p = out_dir / "preview_augmentation_stages.png"
    cv2.imwrite(
        str(p),
        cv2.cvtColor(
            montage(stage_imgs, stage_titles, stage_subs, cols=3),
            cv2.COLOR_RGB2BGR,
        ),
    )
    paths["augmentation"] = p

    # clean camera views to inspect pose / perspective
    g_imgs, g_titles, g_subs = [], [], []
    for i, obs in zip(idx, obs_list):
        geom_only = gen.generate(
            traj[int(i)],
            np.random.default_rng(int(i) + seed + 100),
            geometry=False,
            photometric=False,
        )
        g_imgs.append(geom_only.image)
        g_titles.append(f"t={int(i)}")
        g_subs.append(
            f"pitch={np.rad2deg(geom_only.pose.pitch):+.1f} "
            f"roll={np.rad2deg(geom_only.pose.roll):+.1f} "
            f"cam-centre={np.linalg.norm(geom_only.center_xy - geom_only.gt_xy):.1f}px"
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

    # validity-mask graph: image with invalid pixels tinted, plus the mask
    v_imgs, v_titles, v_subs = [], [], []
    for i, obs in zip(idx[:4], obs_list[:4]):
        valid = obs.valid_mask
        map_valid = render_validity(
            map_.image, gen.view_to_map(obs.pose, obs.aug.scale), cfg.geometry
        )
        v_imgs.append(overlay_invalid(obs.image, valid))
        v_titles.append(f"t={int(i)} invalid tinted")
        v_subs.append(
            f"valid={valid.mean():.3f} map={map_valid.mean():.3f}"
        )
        v_imgs.append(draw_mask(valid))
        v_titles.append(f"t={int(i)} valid mask")
        v_subs.append("white = trustworthy")
    p = out_dir / "preview_validity.png"
    cv2.imwrite(
        str(p),
        cv2.cvtColor(
            montage(v_imgs, v_titles, v_subs, cols=4, cell=cfg.geometry.obs_width),
            cv2.COLOR_RGB2BGR,
        ),
    )
    paths["validity"] = p

    # validity stages for one frame: image, map coverage, combined mask
    h_view_to_map = gen.view_to_map(full.pose, full.aug.scale)
    map_valid = render_validity(map_.image, h_view_to_map, cfg.geometry)
    validity_stages = [
        full.image,
        draw_mask(map_valid),
        draw_mask(full.valid_mask),
        overlay_invalid(full.image, full.valid_mask),
    ]
    validity_titles = [
        "noisy image",
        "map coverage",
        "valid mask",
        "invalid tinted",
    ]
    validity_subs = [
        "gaussian + random invalid",
        "pixels that sample the map",
        "coverage AND not invalid",
        "red = cannot trust",
    ]
    p = out_dir / "preview_validity_stages.png"
    cv2.imwrite(
        str(p),
        cv2.cvtColor(
            montage(validity_stages, validity_titles, validity_subs, cols=4),
            cv2.COLOR_RGB2BGR,
        ),
    )
    paths["validity_stages"] = p

    return paths


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--map", default="res/example.png")
    parser.add_argument("--out-dir", default="outputs/previews")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--traj-steps", type=int, default=700)
    parser.add_argument("--n-windows", type=int, default=12)
    parser.add_argument("--obs-size", type=int, default=256)
    parser.add_argument("--height", type=float, default=180.0)
    parser.add_argument("--footprint", type=float, default=120.0)
    args = parser.parse_args(argv)

    cfg = InfraConfig(
        geometry=GeometryConfig(
            obs_width=args.obs_size,
            obs_height=args.obs_size,
            height_px=args.height,
            footprint_px=args.footprint,
        ),
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
        print(f"{name:>16}: {p}")


if __name__ == "__main__":
    main()