"""Render an example video of the generated observations.

Each frame shows four synchronised panels:

* **map** -- the fixed map, the full trajectory, the trail travelled so far,
  the current camera footprint and its ground position;
* **clean** -- the exact camera view at the frame's effective pose;
* **noisy** -- the same view after zoom/head jitter and sensor noise;
* **validity** -- the trustworthy-pixel mask, plus invalid pixels tinted.

Usage::

    python -m data_gen.video --out outputs/videos/data_gen_example.mp4
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np

from .augment import draw_mask, overlay_invalid
from .camera import wrap_angle
from .config import GeometryConfig, InfraConfig, TrajectoryConfig
from .map import FixedMap
from .observation import ObservationGenerator
from .trajectory import generate_trajectory
from .types import Observation, Trajectory
from .visualize import _label_panel

# RGB palette
_BG = 245
_INK = (35, 35, 35)
_GREY = (95, 95, 95)
_TRAIL = (255, 125, 0)
_ROUTE = (185, 185, 150)
_VIEW = (0, 220, 255)
_CAM = (240, 65, 65)
_START = (40, 200, 90)
_CENTRE = (80, 110, 255)

_HEADER = 66
_BOTTOM = 10
_GAP = 10


def _to_point(xy: np.ndarray, sx: float, sy: float) -> tuple[int, int]:
    return (int(round(float(xy[0]) * sx)), int(round(float(xy[1]) * sy)))


def _draw_map_panel(
    map_: FixedMap,
    traj: Trajectory,
    obs: Observation,
    traj_idx: int,
    panel_h: int,
) -> np.ndarray:
    """Map with the route, trail, current pose and current view footprint."""
    panel_w = int(round(panel_h * map_.width / map_.height))
    img = cv2.resize(
        map_.image, (panel_w, panel_h), interpolation=cv2.INTER_AREA
    )
    sx = panel_w / map_.width
    sy = panel_h / map_.height

    route = traj.poses[:, :2] * np.array([sx, sy], dtype=np.float32)
    cv2.polylines(
        img,
        [np.round(route).astype(np.int32).reshape(-1, 1, 2)],
        False,
        _ROUTE,
        1,
        cv2.LINE_AA,
    )

    trail = traj.poses[: traj_idx + 1, :2] * np.array(
        [sx, sy], dtype=np.float32
    )
    if len(trail) > 1:
        cv2.polylines(
            img,
            [np.round(trail).astype(np.int32).reshape(-1, 1, 2)],
            False,
            _TRAIL,
            3,
            cv2.LINE_AA,
        )

    corners = np.round(
        obs.corners * np.array([sx, sy], dtype=np.float32)
    ).astype(np.int32)
    cv2.polylines(
        img,
        [corners.reshape(-1, 1, 2)],
        True,
        _VIEW,
        2,
        cv2.LINE_AA,
    )

    start = _to_point(traj.poses[0, :2], sx, sy)
    cv2.circle(img, start, 6, _START, -1, cv2.LINE_AA)

    cam = _to_point(obs.gt_xy, sx, sy)
    unit = np.array([np.cos(obs.pose.yaw) * sx, np.sin(obs.pose.yaw) * sy])
    norm = float(np.linalg.norm(unit))
    if norm > 1e-9:
        unit = unit / norm
    tip = (int(round(cam[0] + 30 * unit[0])), int(round(cam[1] + 30 * unit[1])))
    cv2.arrowedLine(img, cam, tip, _CAM, 2, cv2.LINE_AA, tipLength=0.35)
    cv2.circle(img, cam, 5, _CAM, -1, cv2.LINE_AA)

    centre = _to_point(obs.center_xy, sx, sy)
    cv2.drawMarker(
        img,
        centre,
        _CENTRE,
        markerType=cv2.MARKER_CROSS,
        markerSize=12,
        thickness=1,
    )
    return img


def _frame_info(obs: Observation) -> str:
    yaw_gt = float(np.rad2deg(obs.pose.yaw))
    yaw_in = float(np.rad2deg(obs.yaw_input))
    yaw_err = float(np.rad2deg(wrap_angle(obs.yaw_input - obs.pose.yaw)))
    pitch_gt = float(np.rad2deg(obs.pose.pitch))
    pitch_in = float(np.rad2deg(obs.pitch_input))
    sigma = float(obs.photo.get("gauss_sigma", 0.0))
    valid = float(obs.valid_mask.mean()) if obs.valid_mask is not None else 1.0
    return (
        f"cam=({obs.gt_xy[0]:.0f},{obs.gt_xy[1]:.0f})px  "
        f"yaw {yaw_gt:+.1f}deg (in {yaw_in:+.1f}, err {yaw_err:+.2f})  "
        f"pitch {pitch_gt:+.1f}deg (in {pitch_in:+.1f})  "
        f"zoom {obs.aug.scale:.2f}  sigma {sigma:.1f}  valid {valid:.3f}"
    )


def compose_frame(
    map_: FixedMap,
    traj: Trajectory,
    obs: Observation,
    clean: Observation,
    traj_idx: int,
    t: int,
    total: int,
    *,
    cell: int,
    gap: int = _GAP,
) -> np.ndarray:
    """Build one RGB video frame."""
    map_h = 2 * cell + 34 + gap
    map_panel = _draw_map_panel(map_, traj, obs, traj_idx, map_h)
    map_w = map_panel.shape[1]

    left = _label_panel(
        map_panel,
        "map + trajectory",
        "green start | red camera | yellow FOV | orange trail | blue centre",
    )

    def tile(image: np.ndarray, title: str, subtitle: str) -> np.ndarray:
        if image.shape[0] != cell or image.shape[1] != cell:
            image = cv2.resize(
                image, (cell, cell), interpolation=cv2.INTER_AREA
            )
        return _label_panel(image, title, subtitle)

    tiles = [
        tile(
            clean.image,
            "clean camera view",
            f"pitch {np.rad2deg(clean.pose.pitch):+.1f}deg "
            f"roll {np.rad2deg(clean.pose.roll):+.1f}deg",
        ),
        tile(
            obs.image,
            "noisy observation",
            f"yaw in {np.rad2deg(obs.yaw_input):+.1f}deg "
            f"pitch in {np.rad2deg(obs.pitch_input):+.1f}deg",
        ),
        tile(
            draw_mask(obs.valid_mask),
            "validity mask",
            f"white = trustworthy ({float(obs.valid_mask.mean()):.3f})",
        ),
        tile(
            overlay_invalid(obs.image, obs.valid_mask),
            "invalid pixels tinted",
            "red = invalid (map edge + random)",
        ),
    ]

    right_w = 2 * cell + gap
    width = map_w + right_w + 4 * gap
    height = _HEADER + left.shape[0] + _BOTTOM
    canvas = np.full((height, width, 3), _BG, dtype=np.uint8)

    # header
    cv2.putText(
        canvas,
        "data_gen | fixed-map localization observations",
        (12, 26),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        _INK,
        1,
        cv2.LINE_AA,
    )
    counter = f"frame {t + 1}/{total}"
    (tw, _), _ = cv2.getTextSize(
        counter, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
    )
    cv2.putText(
        canvas,
        counter,
        (width - tw - 12, 26),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        _INK,
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        _frame_info(obs),
        (12, 48),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.38,
        _GREY,
        1,
        cv2.LINE_AA,
    )
    x0, x1 = 12, width - 12
    cv2.rectangle(
        canvas, (x0, _HEADER - 6), (x1, _HEADER - 2), (225, 225, 225), -1
    )
    filled = int(round((x1 - x0) * (t + 1) / max(total, 1)))
    cv2.rectangle(
        canvas,
        (x0, _HEADER - 6),
        (x0 + filled, _HEADER - 2),
        (60, 170, 95),
        -1,
    )

    y = _HEADER
    canvas[y : y + left.shape[0], gap : gap + map_w] = left

    rx = gap + map_w + gap
    for i, panel in enumerate(tiles):
        r, c = divmod(i, 2)
        x = rx + c * (cell + gap)
        yy = _HEADER + r * (cell + 34 + gap)
        canvas[yy : yy + panel.shape[0], x : x + cell] = panel

    # encoders prefer even dimensions
    if canvas.shape[0] % 2 or canvas.shape[1] % 2:
        canvas = cv2.copyMakeBorder(
            canvas,
            0,
            canvas.shape[0] % 2,
            0,
            canvas.shape[1] % 2,
            cv2.BORDER_CONSTANT,
            value=(_BG, _BG, _BG),
        )
    return canvas


def _write_video(
    frames,
    out_path: Path,
    *,
    fps: float,
) -> Path:
    """Write frames with OpenCV, then transcode to H.264 when possible."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    raw_path = out_path.with_name(out_path.stem + ".raw.mp4")
    first = next(frames)
    height, width = first.shape[:2]
    writer = cv2.VideoWriter(
        str(raw_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    if not writer.isOpened():
        raise RuntimeError(
            f"could not open a video writer for {raw_path} "
            "(OpenCV was built without a usable MP4 encoder)"
        )
    try:
        writer.write(cv2.cvtColor(first, cv2.COLOR_RGB2BGR))
        for frame in frames:
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    finally:
        writer.release()

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raw_path.replace(out_path)
        return out_path

    result = subprocess.run(
        [
            ffmpeg,
            "-y",
            "-loglevel",
            "error",
            "-i",
            str(raw_path),
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "20",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(out_path),
        ],
        capture_output=True,
    )
    if result.returncode != 0 or not out_path.exists():
        raw_path.replace(out_path)
    else:
        raw_path.unlink(missing_ok=True)
    return out_path


def render_video(
    map_path: str | Path,
    out_path: str | Path,
    *,
    cfg: InfraConfig | None = None,
    seed: int = 0,
    traj_steps: int = 900,
    n_frames: int = 300,
    fps: float = 25.0,
    cell: int = 256,
    poster: bool = True,
) -> Path:
    """Render one example sequence and write it to ``out_path``.

    The trajectory is generated once; ``n_frames`` evenly spaced poses are
    rendered.  Ground truth, noisy attitude inputs and noise parameters are
    seeded, so the output is reproducible.
    """
    cfg = cfg or InfraConfig()
    out_path = Path(out_path)
    map_ = FixedMap(map_path)
    gen = ObservationGenerator(
        map_, cfg.geometry, cfg.photometric, cfg.measurement
    )
    rng = np.random.default_rng(seed)

    traj_steps = max(int(traj_steps), int(n_frames), 1)
    traj = generate_trajectory(
        map_, cfg.trajectory, n_steps=traj_steps, rng=rng
    )
    idx = np.unique(
        np.linspace(0, traj_steps - 1, int(n_frames)).round().astype(int)
    )

    def frames():
        poster_frame = None
        mid = len(idx) // 2
        for t, traj_idx in enumerate(idx):
            pose = traj[int(traj_idx)]
            obs = gen.generate(pose, rng)
            clean = gen.generate(
                pose,
                rng,
                geometry=True,
                photometric=False,
                aug=obs.aug,
            )
            frame = compose_frame(
                map_,
                traj,
                obs,
                clean,
                int(traj_idx),
                t,
                len(idx),
                cell=cell,
            )
            if poster and t == mid:
                poster_frame = frame.copy()
            yield frame
        if poster and poster_frame is not None:
            p = out_path.with_name(out_path.stem + "_poster.png")
            p.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(
                str(p), cv2.cvtColor(poster_frame, cv2.COLOR_RGB2BGR)
            )

    return _write_video(frames(), out_path, fps=fps)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--map", default="res/example.png")
    parser.add_argument("--out", default="outputs/videos/data_gen_example.mp4")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--traj-steps", type=int, default=900)
    parser.add_argument("--n-frames", type=int, default=300)
    parser.add_argument("--fps", type=float, default=25.0)
    parser.add_argument("--cell", type=int, default=256)
    parser.add_argument("--obs-size", type=int, default=256)
    parser.add_argument("--height", type=float, default=180.0)
    parser.add_argument("--footprint", type=float, default=120.0)
    parser.add_argument("--no-poster", action="store_true")
    args = parser.parse_args(argv)

    cfg = InfraConfig(
        geometry=GeometryConfig(
            obs_width=args.obs_size,
            obs_height=args.obs_size,
            height_px=args.height,
            footprint_px=args.footprint,
        ),
        trajectory=TrajectoryConfig(),
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