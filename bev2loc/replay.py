"""Render a replay video of one benchmark trajectory.

Three synchronised panels per frame:

* **map** -- the fixed map with the true pose (green) and the estimate (red),
  plus the trajectories seen so far;
* **observation** -- the exact corrupted BEV clip the model received;
* **posterior** -- the method's likelihood over position (max over heading),
  with the predicted and true poses marked.

Usage::

    python -m bev2loc.replay --scenario seen_moderate --method learned+filter \\
        --learned-checkpoint outputs/checkpoints/obs_moderate.pt \\
        --out outputs/videos/replay.mp4
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np
import torch

from .evaluation.protocols import build_map_split
from .evaluation.runner import (
    BenchmarkConfig,
    BenchmarkRunner,
    default_scenarios,
)
from .types import Pose

# RGB palette
GREEN = (40, 210, 90)
RED = (255, 70, 70)
YELLOW = (255, 220, 70)
DIM = (70, 70, 70)
WHITE = (245, 245, 245)
BG = (243, 243, 243)
INK = (32, 32, 32)
GREY = (105, 105, 105)

FONT = cv2.FONT_HERSHEY_SIMPLEX


def _polyline(
    img: np.ndarray,
    pts: np.ndarray,
    sx: float,
    sy: float,
    colour: tuple[int, int, int],
    thickness: int = 2,
) -> None:
    if len(pts) < 2:
        return
    p = np.array(
        [[int(round(x * sx)), int(round(y * sy))] for x, y in pts], dtype=np.int32
    ).reshape(-1, 1, 2)
    cv2.polylines(img, [p], False, colour, thickness, cv2.LINE_AA)


def _pt(x: float, y: float, sx: float, sy: float) -> tuple[int, int]:
    return int(round(x * sx)), int(round(y * sy))


def _arrow(
    img: np.ndarray,
    x: float,
    y: float,
    yaw: float,
    sx: float,
    sy: float,
    colour: tuple[int, int, int],
    length: float = 24.0,
    thickness: int = 2,
) -> None:
    p0 = _pt(x, y, sx, sy)
    vx, vy = np.cos(yaw) * sx, np.sin(yaw) * sy
    norm = float(np.hypot(vx, vy))
    if norm < 1e-9:
        return
    p1 = (int(round(p0[0] + vx / norm * length)), int(round(p0[1] + vy / norm * length)))
    cv2.arrowedLine(img, p0, p1, colour, thickness, cv2.LINE_AA, tipLength=0.35)


def _draw_poses(
    img: np.ndarray,
    gt_xy: np.ndarray,
    gt_yaw: float,
    pred: Pose,
    sx: float,
    sy: float,
    *,
    radius: int = 5,
    join: bool = True,
) -> None:
    g = _pt(float(gt_xy[0]), float(gt_xy[1]), sx, sy)
    p = _pt(pred.x, pred.y, sx, sy)
    if join:
        cv2.line(img, g, p, YELLOW, 1, cv2.LINE_AA)
    cv2.circle(img, g, radius + 1, WHITE, -1, cv2.LINE_AA)
    cv2.circle(img, g, radius, GREEN, -1, cv2.LINE_AA)
    _arrow(img, float(gt_xy[0]), float(gt_xy[1]), gt_yaw, sx, sy, GREEN)
    cv2.circle(img, p, radius + 1, WHITE, -1, cv2.LINE_AA)
    cv2.circle(img, p, radius, RED, -1, cv2.LINE_AA)
    _arrow(img, pred.x, pred.y, pred.yaw, sx, sy, RED)


def _caption(img: np.ndarray, title: str, sub: str) -> np.ndarray:
    h, w = img.shape[:2]
    bar = 42
    out = np.full((h + bar, w, 3), BG, dtype=np.uint8)
    out[:h] = img
    cv2.putText(out, title, (8, h + 18), FONT, 0.48, INK, 1, cv2.LINE_AA)
    cv2.putText(out, sub, (8, h + 34), FONT, 0.40, GREY, 1, cv2.LINE_AA)
    return out


def _map_panel(
    map_img: np.ndarray,
    seq,
    estimates: list[Pose],
    t: int,
    width: int,
    height: int,
) -> np.ndarray:
    panel = cv2.resize(map_img, (width, height), interpolation=cv2.INTER_AREA)
    sx, sy = width / map_img.shape[1], height / map_img.shape[0]
    _polyline(panel, seq.poses[:, :2], sx, sy, (215, 215, 215), 1)
    _polyline(panel, seq.gt_xy[: t + 1], sx, sy, GREEN, 2)
    pred_xy = np.array([[e.x, e.y] for e in estimates[: t + 1]], dtype=np.float32)
    _polyline(panel, pred_xy, sx, sy, RED, 2)
    _draw_poses(panel, seq.gt_xy[t], float(seq.gt_yaw[t]), estimates[t], sx, sy)
    cv2.putText(panel, "true", (8, 18), FONT, 0.45, GREEN, 1, cv2.LINE_AA)
    cv2.putText(panel, "estimate", (8, 35), FONT, 0.45, RED, 1, cv2.LINE_AA)
    return panel


def _posterior_panel(
    map_img: np.ndarray,
    volume: torch.Tensor,
    seq,
    estimates: list[Pose],
    t: int,
    width: int,
    height: int,
) -> np.ndarray:
    vol = volume.detach().cpu().float()
    prob = (vol - vol.max()).exp().max(dim=0).values  # (ny, nx), best heading
    prob = (prob / prob.max().clamp_min(1e-12)).numpy()
    prob = cv2.resize(prob, (width, height), interpolation=cv2.INTER_LINEAR)
    idx = (np.clip(prob, 0.0, 1.0) ** 0.55 * 255).astype(np.uint8)
    heat = cv2.cvtColor(cv2.applyColorMap(idx, cv2.COLORMAP_TURBO), cv2.COLOR_BGR2RGB)
    base = cv2.cvtColor(
        cv2.resize(map_img, (width, height), interpolation=cv2.INTER_AREA),
        cv2.COLOR_RGB2GRAY,
    )
    base = cv2.cvtColor(base, cv2.COLOR_GRAY2RGB).astype(np.float32) * 0.5
    panel = (heat.astype(np.float32) * 0.8 + base * 0.2).clip(0, 255).astype(np.uint8)
    sx, sy = width / map_img.shape[1], height / map_img.shape[0]
    _draw_poses(panel, seq.gt_xy[t], float(seq.gt_yaw[t]), estimates[t], sx, sy)
    return panel


def _observation_panel(
    obs, size: int, *, show_mask: bool = False
) -> np.ndarray:
    img = cv2.resize(obs.image, (size, size), interpolation=cv2.INTER_LINEAR)
    if show_mask:
        mask = cv2.resize(
            obs.valid_mask.astype(np.float32), (size, size), interpolation=cv2.INTER_LINEAR
        )[..., None]
        tint = np.array([60, 100, 230], dtype=np.float32)
        alpha = (1.0 - mask) * 0.45
        img = (img.astype(np.float32) * (1 - alpha) + tint * alpha).clip(0, 255).astype(np.uint8)
    cv2.drawMarker(
        img,
        (size // 2, size // 2),
        RED,
        markerType=cv2.MARKER_CROSS,
        markerSize=14,
        thickness=1,
    )
    return img


def _compose(
    map_img: np.ndarray,
    seq,
    estimates: list[Pose],
    volumes: list[torch.Tensor],
    t: int,
    *,
    method: str,
    scenario: str,
    panel_h: int,
    show_mask: bool,
    gap: int = 10,
) -> np.ndarray:
    mh, mw = map_img.shape[:2]
    map_w = int(round(panel_h * mw / mh))
    path_err = float(np.hypot(estimates[t].x - seq.gt_xy[t, 0], estimates[t].y - seq.gt_xy[t, 1]))
    dyaw = float(
        np.arctan2(
            np.sin(estimates[t].yaw - seq.gt_yaw[t]),
            np.cos(estimates[t].yaw - seq.gt_yaw[t]),
        )
    )
    yaw_err = abs(np.rad2deg(dyaw))
    total = len(seq.observations)

    title = (
        f"bev2loc replay | {scenario} | {method} | frame {t + 1}/{total} | "
        f"pos err {path_err:.1f} px | yaw err {yaw_err:.1f} deg"
    )
    bar_h = 46
    panels = [
        _caption(
            _map_panel(map_img, seq, estimates, t, map_w, panel_h),
            "MAP: true pose (green) vs estimate (red)",
            "trajectories: true green / estimate red",
        ),
        _caption(
            _observation_panel(seq.observations[t], panel_h, show_mask=show_mask),
            "OBSERVATION: corrupted BEV clip",
            f"centre=({seq.gt_xy[t, 0]:.0f},{seq.gt_xy[t, 1]:.0f}) "
            f"yaw={np.rad2deg(seq.gt_yaw[t]):.0f} deg",
        ),
        _caption(
            _posterior_panel(map_img, volumes[t], seq, estimates, t, map_w, panel_h),
            "POSTERIOR: likelihood over position",
            "max over heading; brighter = more likely",
        ),
    ]
    ph = panels[0].shape[0]
    width = gap * 4 + map_w + panel_h + map_w
    height = bar_h + gap + ph + gap
    canvas = np.full((height, width, 3), BG, dtype=np.uint8)
    canvas[:bar_h, :] = np.full((bar_h, width, 3), 28, dtype=np.uint8)
    cv2.putText(canvas, title, (14, 30), FONT, 0.6, WHITE, 1, cv2.LINE_AA)
    x = gap
    for panel in panels:
        ph_i, pw_i = panel.shape[:2]
        canvas[bar_h + gap : bar_h + gap + ph_i, x : x + pw_i] = panel
        x += pw_i + gap
    # progress line
    y = height - 4
    cv2.line(canvas, (gap, y), (width - gap, y), (190, 190, 190), 2)
    x_end = int(gap + (width - 2 * gap) * (t + 1) / total)
    cv2.line(canvas, (gap, y), (x_end, y), GREEN, 2)
    return canvas


def _transcode(raw: Path, out: Path) -> Path:
    if shutil.which("ffmpeg") is None:
        raw.replace(out)
        return out
    cmd = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-i",
        str(raw),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-crf",
        "18",
        "-movflags",
        "+faststart",
        str(out),
    ]
    result = subprocess.run(cmd)
    if result.returncode == 0:
        raw.unlink(missing_ok=True)
        return out
    raw.replace(out)
    return out


def render_replay(
    *,
    out: str | Path,
    map_path: str | Path = "res/example.png",
    scenario: str = "seen_moderate",
    method: str = "learned+filter",
    learned_checkpoint: str | Path | None = "outputs/checkpoints/obs_moderate.pt",
    rnn_checkpoint: str | Path | None = "outputs/checkpoints/rnn_yaw.pt",
    map_variants_dir: str | Path = "outputs/maps",
    seq_len: int = 40,
    seed: int | None = None,
    fps: int = 8,
    panel_h: int = 430,
    max_frames: int | None = None,
    show_mask: bool = False,
    device: str = "auto",
    still: str | Path | None = None,
) -> Path:
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    split = build_map_split(map_path, out_dir=map_variants_dir)
    cfg = BenchmarkConfig(device=device)
    runner = BenchmarkRunner(
        split,
        cfg,
        learned_checkpoint=learned_checkpoint,
        rnn_checkpoint=rnn_checkpoint,
    )
    scenarios = {s.name: s for s in default_scenarios(seq_len=seq_len)}
    if scenario not in scenarios:
        raise SystemExit(
            f"unknown scenario {scenario!r}; choose from {sorted(scenarios)}"
        )
    sc = scenarios[scenario]
    if seed is not None:
        sc.seed = int(seed)
    seq = runner._prepare_sequence(sc)
    grid = runner.grid_for(seq.map)

    classical = runner.classical_logits(seq)
    learned = features = None
    if learned_checkpoint is not None:
        learned, features = runner.learned_logits(seq)

    if method == "classical":
        volumes = list(classical)
        estimates = runner._poses_from_logits(classical, grid)
    elif method == "classical+filter":
        volumes, estimates = runner.filter_volumes(
            grid, seq, classical, init=sc.init, seed=sc.seed
        )
    elif method == "learned":
        if learned is None:
            raise SystemExit("--learned-checkpoint is required for learned methods")
        volumes = list(learned)
        estimates = runner._poses_from_logits(learned, grid)
    elif method == "learned+filter":
        if learned is None:
            raise SystemExit("--learned-checkpoint is required for learned methods")
        volumes, estimates = runner.filter_volumes(
            grid, seq, learned, init=sc.init, seed=sc.seed
        )
    elif method == "rnn":
        if learned is None or features is None or rnn_checkpoint is None:
            raise SystemExit("rnn needs --learned-checkpoint and --rnn-checkpoint")
        volumes, estimates = runner.rnn_volumes(
            seq, learned, features, init=sc.init
        )
    else:
        raise SystemExit(
            "unknown method; choose classical, classical+filter, learned, "
            "learned+filter or rnn"
        )

    total = len(seq.observations)
    if max_frames is not None:
        total = min(total, int(max_frames))

    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    raw = out.with_name(out.stem + "_raw.mp4")
    first = _compose(
        seq.map.image,
        seq,
        estimates,
        volumes,
        0,
        method=method,
        scenario=scenario,
        panel_h=panel_h,
        show_mask=show_mask,
    )
    height, width = first.shape[:2]
    writer = cv2.VideoWriter(
        str(raw), cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (width, height)
    )
    if not writer.isOpened():
        raise SystemExit(f"could not open video writer for {raw}")
    still_idx = total // 2
    for t in range(total):
        frame = _compose(
            seq.map.image,
            seq,
            estimates,
            volumes,
            t,
            method=method,
            scenario=scenario,
            panel_h=panel_h,
            show_mask=show_mask,
        )
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        if still is not None and t == still_idx:
            still_path = Path(still)
            still_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(
                str(still_path), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            )
    writer.release()
    return _transcode(raw, out)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--map", default="res/example.png")
    p.add_argument("--scenario", default="seen_moderate")
    p.add_argument(
        "--method",
        default="learned+filter",
        choices=["classical", "classical+filter", "learned", "learned+filter", "rnn"],
    )
    p.add_argument(
        "--learned-checkpoint", default="outputs/checkpoints/obs_moderate.pt"
    )
    p.add_argument("--rnn-checkpoint", default="outputs/checkpoints/rnn_yaw.pt")
    p.add_argument("--map-variants-dir", default="outputs/maps")
    p.add_argument("--out", default="outputs/videos/replay.mp4")
    p.add_argument("--still", default="outputs/videos/replay_frame.png")
    p.add_argument("--seq-len", type=int, default=40)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--fps", type=int, default=8)
    p.add_argument("--panel-height", type=int, default=430)
    p.add_argument("--max-frames", type=int, default=None)
    p.add_argument("--show-mask", action="store_true")
    p.add_argument("--device", default="auto")
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    path = render_replay(
        out=args.out,
        map_path=args.map,
        scenario=args.scenario,
        method=args.method,
        learned_checkpoint=args.learned_checkpoint,
        rnn_checkpoint=args.rnn_checkpoint,
        map_variants_dir=args.map_variants_dir,
        seq_len=args.seq_len,
        seed=args.seed,
        fps=args.fps,
        panel_h=args.panel_height,
        max_frames=args.max_frames,
        show_mask=args.show_mask,
        device=args.device,
        still=args.still,
    )
    print(f"saved {path}")


if __name__ == "__main__":
    main()