"""Render a validation video for one BEV localization scenario.

The video replays a benchmark scenario frame by frame and shows, for the best
method (``learned+filter`` by default):

* the fixed map with the ground-truth route, the filtered estimate trail and
  the current ground-truth / estimate positions;
* the raw perspective observation;
* the attitude-rectified BEV patch that the matcher actually sees;
* the map template at the current estimate;
* the full-map filter posterior and a zoom around the ground truth;
* the per-frame position-error history.

Usage::

    python -m map2loc.replay \
        --map res/example.png \
        --scenario seen_moderate \
        --learned-checkpoint outputs/checkpoints/obs_bev_long.pt \
        --out outputs/videos/replay_seen_moderate.mp4

``--method classical+filter`` works without a checkpoint and is useful for a
quick smoke test.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch

from data_gen.camera import wrap_angle
from data_gen.map import FixedMap
from data_gen.video import _write_video
from data_gen.visualize import _label_panel

from .bev import centre_offset
from .evaluation.protocols import MapSplit, build_map_split
from .evaluation.runner import (
    BenchmarkConfig,
    BenchmarkRunner,
    Scenario,
    default_scenarios,
)
from .matching.grid import PoseGrid

# RGB palette
_BG = 245
_INK = (35, 35, 35)
_GREY = (110, 110, 110)
_ROUTE = (175, 175, 175)
_GT_TRAIL = (230, 150, 40)
_EST_TRAIL = (40, 110, 235)
_PER_FRAME = (200, 60, 190)
_GT = (225, 55, 55)
_EST = (40, 100, 240)
_THRESH = (190, 60, 60)

_HEADER = 68
_BOTTOM = 10
_GAP = 10
_SUCCESS_PX = 16.0
_SUCCESS_DEG = 10.0


# ---------------------------------------------------------------------------
# small drawing helpers
# ---------------------------------------------------------------------------
def _to_px(xy, sx: float, sy: float) -> tuple[int, int]:
    return (int(round(float(xy[0]) * sx)), int(round(float(xy[1]) * sy)))


def _arrow(img, xy, yaw: float, sx: float, sy: float, color, length: int = 26):
    p0 = _to_px(xy, sx, sy)
    dx, dy = np.cos(yaw) * sx, np.sin(yaw) * sy
    norm = float(np.hypot(dx, dy))
    if norm < 1e-9:
        return p0
    p1 = (
        int(round(p0[0] + length * dx / norm)),
        int(round(p0[1] + length * dy / norm)),
    )
    cv2.arrowedLine(img, p0, p1, color, 2, cv2.LINE_AA, tipLength=0.35)
    cv2.circle(img, p0, 5, color, -1, cv2.LINE_AA)
    return p0


def _tint_invalid(
    image: np.ndarray,
    valid: np.ndarray,
    colour=(255, 70, 70),
    alpha: float = 0.45,
) -> np.ndarray:
    """Lightly tint untrustworthy pixels so the BEV content stays visible."""
    out = image.copy()
    invalid = np.asarray(valid) < 0.5
    if invalid.any():
        out[invalid] = (
            (1.0 - alpha) * out[invalid].astype(np.float32)
            + alpha * np.asarray(colour, dtype=np.float32)
        ).astype(np.uint8)
    return out


def _draw_polyline(img, xy, sx: float, sy: float, color, thickness: int = 2):
    if len(xy) < 2:
        return
    pts = np.round(np.asarray(xy) * np.array([sx, sy])).astype(np.int32)
    cv2.polylines(
        img, [pts.reshape(-1, 1, 2)], False, color, thickness, cv2.LINE_AA
    )


# ---------------------------------------------------------------------------
# panels
# ---------------------------------------------------------------------------
def _map_panel(
    map_: FixedMap,
    seq,
    estimates: np.ndarray,
    per_frame: np.ndarray,
    t: int,
    panel_h: int,
) -> tuple[np.ndarray, float, float]:
    panel_w = int(round(panel_h * map_.width / map_.height))
    img = cv2.resize(
        map_.image, (panel_w, panel_h), interpolation=cv2.INTER_AREA
    )
    sx = panel_w / map_.width
    sy = panel_h / map_.height

    _draw_polyline(img, seq.poses[:, :2], sx, sy, _ROUTE, 1)
    _draw_polyline(img, seq.gt_xy[: t + 1], sx, sy, _GT_TRAIL, 3)
    _draw_polyline(img, estimates[: t + 1], sx, sy, _EST_TRAIL, 3)

    # per-frame (no filter) estimates are shown as small dots
    for xy in per_frame[: t + 1]:
        cv2.circle(img, _to_px(xy, sx, sy), 2, _PER_FRAME, -1, cv2.LINE_AA)

    gt = seq.gt_xy[t]
    est = estimates[t]
    cv2.line(
        img, _to_px(gt, sx, sy), _to_px(est, sx, sy), _THRESH, 1, cv2.LINE_AA
    )
    _arrow(img, gt, float(seq.gt_yaw[t]), sx, sy, _GT)
    _arrow(img, est, float(seq.observations[t].yaw_input), sx, sy, _EST)

    cv2.putText(
        img,
        f"t={t}",
        (10, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        _INK,
        1,
        cv2.LINE_AA,
    )
    return img, sx, sy


def _heat_overlay(map_: FixedMap, post: np.ndarray, grid: PoseGrid) -> np.ndarray:
    """Full-map heat overlay at map resolution (RGB)."""
    prob = np.maximum(np.asarray(post, dtype=np.float32), 1e-12)
    logp = np.log(prob)
    rel = np.clip((logp - logp.max()) / 4.0 + 1.0, 0.0, 1.0)
    heat = cv2.applyColorMap(
        np.round(rel * 255.0).astype(np.uint8), cv2.COLORMAP_TURBO
    )
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
    heat = cv2.resize(
        heat, (map_.width, map_.height), interpolation=cv2.INTER_LINEAR
    )
    gray = cv2.cvtColor(
        cv2.cvtColor(map_.image, cv2.COLOR_RGB2GRAY), cv2.COLOR_GRAY2RGB
    )
    mask = rel > 0.03
    mask = cv2.resize(
        mask.astype(np.uint8) * 255,
        (map_.width, map_.height),
        interpolation=cv2.INTER_NEAREST,
    ).astype(bool)
    out = gray.copy()
    blend = np.clip(gray.astype(np.float32) * 0.35 + heat * 0.75, 0, 255)
    out[mask] = blend[mask].astype(np.uint8)
    return out


def _posterior_panels(
    map_: FixedMap,
    heat: np.ndarray,
    gt: np.ndarray,
    est: np.ndarray,
    cell: int,
    *,
    zoom_half: int = 150,
) -> tuple[np.ndarray, np.ndarray]:
    # full map
    full = cv2.resize(heat, (cell, cell), interpolation=cv2.INTER_AREA)
    sx = cell / map_.width
    sy = cell / map_.height
    cv2.drawMarker(
        full,
        _to_px(gt, sx, sy),
        (255, 255, 255),
        markerType=cv2.MARKER_CROSS,
        markerSize=12,
        thickness=1,
    )
    cv2.circle(full, _to_px(est, sx, sy), 5, (255, 255, 255), 1, cv2.LINE_AA)

    # zoom around the ground truth
    cx, cy = float(gt[0]), float(gt[1])
    x0 = int(np.clip(round(cx) - zoom_half, 0, max(map_.width - 1, 0)))
    y0 = int(np.clip(round(cy) - zoom_half, 0, max(map_.height - 1, 0)))
    x1 = int(np.clip(x0 + 2 * zoom_half, 1, map_.width))
    y1 = int(np.clip(y0 + 2 * zoom_half, 1, map_.height))
    x0 = max(0, x1 - 2 * zoom_half)
    y0 = max(0, y1 - 2 * zoom_half)
    crop = heat[y0:y1, x0:x1]
    zoom = cv2.resize(crop, (cell, cell), interpolation=cv2.INTER_NEAREST)
    zs = cell / max(x1 - x0, 1)
    gx = int(round((gt[0] - x0) * zs))
    gy = int(round((gt[1] - y0) * zs))
    cv2.drawMarker(
        zoom,
        (gx, gy),
        (255, 255, 255),
        markerType=cv2.MARKER_CROSS,
        markerSize=14,
        thickness=1,
    )
    ex = int(round((est[0] - x0) * zs))
    ey = int(round((est[1] - y0) * zs))
    if 0 <= ex < cell and 0 <= ey < cell:
        cv2.circle(zoom, (ex, ey), 6, (255, 255, 255), 1, cv2.LINE_AA)
    else:
        # estimate is outside the zoom: point towards it from the border
        p0 = (gx, gy)
        p1 = (
            int(np.clip(ex, 4, cell - 5)),
            int(np.clip(ey, 4, cell - 5)),
        )
        cv2.arrowedLine(
            zoom, p0, p1, (255, 255, 255), 2, cv2.LINE_AA, tipLength=0.25
        )
    return full, zoom


def _error_plot(
    errors: np.ndarray,
    per_frame_errors: np.ndarray,
    t: int,
    size: int,
) -> np.ndarray:
    img = np.full((size, size, 3), 250, dtype=np.uint8)
    left, right, top, bottom = 34, 10, 12, 26
    w = size - left - right
    h = size - top - bottom
    ymax = max(50.0, 1.12 * float(np.nanmax(errors[: t + 1])))

    def pt(i: int, value: float) -> tuple[int, int]:
        x = left + (0 if len(errors) <= 1 else int(round(w * i / (len(errors) - 1))))
        y = top + h - int(round(h * min(value, ymax) / ymax))
        return (x, y)

    # grid + axis labels
    for frac in (0.0, 0.25, 0.5, 0.75, 1.0):
        value = ymax * frac
        y = top + h - int(round(h * frac))
        cv2.line(img, (left, y), (left + w, y), (225, 225, 225), 1)
        cv2.putText(
            img,
            f"{value:.0f}",
            (2, y + 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.32,
            (110, 110, 110),
            1,
            cv2.LINE_AA,
        )

    # success threshold
    if _SUCCESS_PX <= ymax:
        y = top + h - int(round(h * _SUCCESS_PX / ymax))
        for x in range(left, left + w, 8):
            cv2.line(img, (x, y), (min(x + 4, left + w), y), _THRESH, 1)
        cv2.putText(
            img,
            "16 px",
            (left + 4, y - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.3,
            _THRESH,
            1,
            cv2.LINE_AA,
        )

    def series(values, color, thickness):
        pts = [pt(i, float(v)) for i, v in enumerate(values[: t + 1])]
        if len(pts) > 1:
            cv2.polylines(
                img,
                [np.asarray(pts, dtype=np.int32).reshape(-1, 1, 2)],
                False,
                color,
                thickness,
                cv2.LINE_AA,
            )
        cv2.circle(img, pts[-1], 3, color, -1, cv2.LINE_AA)

    series(per_frame_errors, _PER_FRAME, 1)
    series(errors, _EST_TRAIL, 2)
    cv2.putText(
        img,
        "filtered",
        (left + 4, top + 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.34,
        _EST_TRAIL,
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        img,
        "per-frame",
        (left + 66, top + 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.34,
        _PER_FRAME,
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        img,
        "position error (px)",
        (left, size - 7),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.34,
        (90, 90, 90),
        1,
        cv2.LINE_AA,
    )
    return img


def _map_template(
    map_: FixedMap,
    estimate: np.ndarray,
    yaw: float,
    pitch: float,
    height: float,
    span: float,
) -> np.ndarray:
    off = centre_offset(float(yaw), float(pitch), float(height))
    centre = np.asarray(estimate, dtype=np.float64) + off
    px = int(round(span))
    affine = np.array(
        [
            [span / px, 0.0, centre[0] - span / 2.0],
            [0.0, span / px, centre[1] - span / 2.0],
        ],
        dtype=np.float64,
    )
    return cv2.warpAffine(
        map_.image,
        affine,
        (px, px),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )


# ---------------------------------------------------------------------------
# frame composition
# ---------------------------------------------------------------------------
def compose_frame(
    map_: FixedMap,
    seq,
    estimates: np.ndarray,
    per_frame: np.ndarray,
    posteriors: list[np.ndarray],
    t: int,
    *,
    grid: PoseGrid,
    runner: BenchmarkRunner,
    method: str,
    scenario_name: str,
    cell: int,
    gap: int = _GAP,
) -> np.ndarray:
    obs = seq.observations[t]
    gt = seq.gt_xy[t]
    est = estimates[t]
    pos_err = float(np.linalg.norm(est - gt))
    yaw_err = float(
        np.rad2deg(wrap_angle(float(obs.yaw_input) - float(seq.gt_yaw[t])))
    )
    success = pos_err < _SUCCESS_PX and abs(yaw_err) < _SUCCESS_DEG

    map_h = 2 * cell + 34 + gap
    map_panel, sx, sy = _map_panel(map_, seq, estimates, per_frame, t, map_h)
    left = _label_panel(
        map_panel,
        "map + trajectory",
        "red GT | blue filtered | magenta per-frame | orange GT trail",
    )

    # rectified BEV at the nominal focal scale
    scale = float(runner.cfg.scales[len(runner.cfg.scales) // 2])
    patches, masks = runner.bev_hypotheses(seq, t)
    bev = _tint_invalid(patches[len(patches) // 2], masks[len(masks) // 2])
    map_tpl = _map_template(
        map_,
        est,
        obs.yaw_input,
        obs.pitch_input,
        runner.cfg.height_px,
        runner.bev.span_px,
    )
    posterior = np.asarray(posteriors[t], dtype=np.float32)
    heat = _heat_overlay(map_, posterior, grid)
    post_full, post_zoom = _posterior_panels(map_, heat, gt, est, cell)

    errors = np.array(
        [float(np.linalg.norm(e - g)) for e, g in zip(estimates, seq.gt_xy)]
    )
    per_frame_errors = np.array(
        [float(np.linalg.norm(e - g)) for e, g in zip(per_frame, seq.gt_xy)]
    )
    plot = _error_plot(errors, per_frame_errors, t, cell)

    def tile(image: np.ndarray, title: str, subtitle: str) -> np.ndarray:
        if image.shape[0] != cell or image.shape[1] != cell:
            image = cv2.resize(
                image, (cell, cell), interpolation=cv2.INTER_AREA
            )
        return _label_panel(image, title, subtitle)

    tiles = [
        tile(
            obs.image,
            "perspective observation",
            f"{seq.corruption} | valid {float(np.mean(obs.valid_mask)):.2f} "
            f"| yaw in {np.rad2deg(obs.yaw_input):+.1f}deg",
        ),
        tile(
            bev,
            "rectified BEV (matcher input)",
            f"measured attitude, scale {scale:.2f}, patch {runner.cfg.patch_px}px",
        ),
        tile(
            map_tpl,
            "map template at estimate",
            f"span {runner.bev.span_px:.0f} map px",
        ),
        tile(
            post_full,
            "filter posterior (whole map)",
            "white cross = GT | circle = estimate",
        ),
        tile(
            post_zoom,
            "posterior zoom at GT",
            "log posterior, 4 decades",
        ),
        tile(
            plot,
            "error history",
            "blue = filtered | magenta = per-frame | dashes = 16 px",
        ),
    ]

    right_w = 3 * cell + 2 * gap
    width = left.shape[1] + right_w + 4 * gap
    height = _HEADER + left.shape[0] + _BOTTOM
    canvas = np.full((height, width, 3), _BG, dtype=np.uint8)

    cv2.putText(
        canvas,
        f"map2loc | {method} | scenario {scenario_name}",
        (12, 26),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        _INK,
        1,
        cv2.LINE_AA,
    )
    counter = f"frame {t + 1}/{len(seq.observations)}"
    (tw, _), _ = cv2.getTextSize(counter, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
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

    verdict = "SUCCESS" if success else "outside 16px/10deg"
    info = (
        f"pos error {pos_err:5.1f} px | measured yaw error {yaw_err:+5.1f} deg"
        f" | {verdict} | map ({gt[0]:.0f},{gt[1]:.0f}) -> "
        f"({est[0]:.0f},{est[1]:.0f})"
    )
    cv2.putText(
        canvas,
        info,
        (12, 48),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.38,
        _GREY,
        1,
        cv2.LINE_AA,
    )

    x0, x1 = 12, width - 12
    cv2.rectangle(canvas, (x0, _HEADER - 6), (x1, _HEADER - 2), (225, 225, 225), -1)
    filled = int(round((x1 - x0) * (t + 1) / max(len(seq.observations), 1)))
    cv2.rectangle(
        canvas, (x0, _HEADER - 6), (x0 + filled, _HEADER - 2), (60, 170, 95), -1
    )

    y = _HEADER
    canvas[y : y + left.shape[0], gap : gap + left.shape[1]] = left
    rx = gap + left.shape[1] + gap
    for i, panel in enumerate(tiles):
        r, c = divmod(i, 3)
        x = rx + c * (cell + gap)
        yy = _HEADER + r * (cell + 34 + gap)
        canvas[yy : yy + panel.shape[0], x : x + panel.shape[1]] = panel

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


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------
def render_replay(
    map_path: str | Path,
    out_path: str | Path,
    *,
    scenario: str = "seen_moderate",
    method: str = "learned+filter",
    split: MapSplit | None = None,
    cfg: BenchmarkConfig | None = None,
    learned_checkpoint: str | Path | None = None,
    fps: float = 5.0,
    cell: int = 256,
    poster: bool = True,
    seq_len: int = 40,
    map_variants_dir: str | Path = "outputs/maps",
) -> Path:
    """Render one scenario and write an MP4 (plus a poster frame)."""
    cfg = cfg or BenchmarkConfig()
    out_path = Path(out_path)
    split = split or build_map_split(map_path, out_dir=map_variants_dir)
    runner = BenchmarkRunner(split, cfg, learned_checkpoint=learned_checkpoint)

    scenarios = {s.name: s for s in default_scenarios(seq_len=seq_len)}
    if scenario not in scenarios:
        raise KeyError(
            f"unknown scenario {scenario!r}; available: {sorted(scenarios)}"
        )
    sc: Scenario = scenarios[scenario]
    seq = runner._prepare_sequence(sc)
    grid = runner.grid_for(seq.map)

    if method.startswith("learned"):
        if learned_checkpoint is None:
            raise ValueError("a learned checkpoint is required for this method")
        logits, _ = runner.learned_logits(seq)
    elif method.startswith("classical"):
        logits = runner.classical_logits(seq)
    else:
        raise ValueError(f"unknown method {method!r}")

    estimates, posteriors = runner.filter_volumes(
        grid, seq, logits, init=sc.init, seed=sc.seed, return_posteriors=True
    )
    yaw_inputs = np.array(
        [o.yaw_input for o in seq.observations], dtype=np.float32
    )
    per_frame = runner._poses_from_logits(logits, grid, yaw_inputs)

    est_xy = np.stack([e.xy for e in estimates])
    per_xy = np.stack([e.xy for e in per_frame])
    post_np = [
        p.detach().cpu().numpy() if isinstance(p, torch.Tensor) else np.asarray(p)
        for p in posteriors
    ]

    def frames():
        poster_frame = None
        mid = len(seq.observations) // 2
        for t in range(len(seq.observations)):
            frame = compose_frame(
                seq.map,
                seq,
                est_xy,
                per_xy,
                post_np,
                t,
                grid=grid,
                runner=runner,
                method=method,
                scenario_name=scenario,
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


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--map", default="res/example.png")
    p.add_argument("--scenario", default="seen_moderate")
    p.add_argument(
        "--method",
        default="learned+filter",
        choices=["learned+filter", "classical+filter"],
    )
    p.add_argument(
        "--learned-checkpoint", default="outputs/checkpoints/obs_bev_long.pt"
    )
    p.add_argument("--out", default=None)
    p.add_argument("--device", default="auto")
    p.add_argument("--resolution", type=float, default=8.0)
    p.add_argument("--n-yaw", type=int, default=64)
    p.add_argument("--patch-px", type=int, default=164)
    p.add_argument("--map-px-per-px", type=float, default=2.0)
    p.add_argument("--fps", type=float, default=5.0)
    p.add_argument("--cell", type=int, default=256)
    p.add_argument("--seq-len", type=int, default=40)
    p.add_argument("--no-poster", action="store_true")
    p.add_argument("--map-variants-dir", default="outputs/maps")
    return p


def main(argv: list[str] | None = None) -> Path:
    args = build_parser().parse_args(argv)
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    out = args.out or (
        f"outputs/videos/replay_{args.scenario}_"
        f"{args.method.replace('+', '_')}.mp4"
    )
    cfg = BenchmarkConfig(
        grid_resolution=args.resolution,
        n_yaw=args.n_yaw,
        patch_px=args.patch_px,
        map_px_per_px=args.map_px_per_px,
        device=device,
    )
    checkpoint = args.learned_checkpoint
    if args.method.startswith("classical"):
        checkpoint = None
    elif checkpoint and not Path(checkpoint).exists():
        raise FileNotFoundError(
            f"learned checkpoint not found: {checkpoint}; "
            "use --method classical+filter for a checkpoint-free smoke test"
        )
    path = render_replay(
        args.map,
        out,
        scenario=args.scenario,
        method=args.method,
        cfg=cfg,
        learned_checkpoint=checkpoint,
        fps=args.fps,
        cell=args.cell,
        poster=not args.no_poster,
        seq_len=args.seq_len,
        map_variants_dir=args.map_variants_dir,
    )
    print(f"wrote {path}")
    return path


if __name__ == "__main__":
    main()