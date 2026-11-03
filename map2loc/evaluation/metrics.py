"""Localization metrics shared by all methods."""

from __future__ import annotations

import numpy as np

from ..types import Pose


def pose_errors(est: Pose, gt_xy: np.ndarray, gt_yaw: float) -> tuple[float, float]:
    pos = float(np.linalg.norm(est.xy - np.asarray(gt_xy, dtype=np.float32)))
    dyaw = float((est.yaw - gt_yaw + np.pi) % (2 * np.pi) - np.pi)
    return pos, abs(dyaw)


def sequence_errors(
    estimates: list[Pose],
    gt_xy: np.ndarray,
    gt_yaw: np.ndarray,
) -> dict[str, np.ndarray]:
    pos = []
    yaw = []
    for e, xy, y in zip(estimates, gt_xy, gt_yaw):
        p, a = pose_errors(e, xy, float(y))
        pos.append(p)
        yaw.append(a)
    return {"pos_px": np.asarray(pos), "yaw_rad": np.asarray(yaw)}


def summarize(
    errors: dict[str, np.ndarray],
    *,
    pos_thresholds: tuple[float, ...] = (8.0, 16.0, 32.0),
    yaw_thresholds_deg: tuple[float, ...] = (5.0, 10.0, 20.0),
    recovery_hold: int = 3,
    recovery_start: int = 0,
) -> dict:
    pos = errors["pos_px"]
    yaw = np.rad2deg(errors["yaw_rad"])
    out: dict = {
        "n": int(pos.size),
        "pos_px_mean": float(pos.mean()) if pos.size else float("nan"),
        "pos_px_median": float(np.median(pos)) if pos.size else float("nan"),
        "pos_px_p90": float(np.percentile(pos, 90)) if pos.size else float("nan"),
        "yaw_deg_mean": float(yaw.mean()) if yaw.size else float("nan"),
        "yaw_deg_median": float(np.median(yaw)) if yaw.size else float("nan"),
        "yaw_deg_p90": float(np.percentile(yaw, 90)) if yaw.size else float("nan"),
        "success": {},
        "recovery_frame": None,
    }
    for pth in pos_thresholds:
        for yth in yaw_thresholds_deg:
            key = f"pos<{pth:g}px_yaw<{yth:g}deg"
            out["success"][key] = float(
                np.mean((pos < pth) & (yaw < yth))
            )
    # Recovery: first frame after which both errors stay below threshold for
    # ``recovery_hold`` consecutive frames.
    pth = 16.0
    yth = 10.0
    ok = (pos < pth) & (yaw < yth)
    recovery = None
    for start in range(int(recovery_start), max(int(recovery_start), ok.size)):
        if ok[start : start + recovery_hold].all() and start + recovery_hold <= ok.size:
            recovery = int(start)
            break
    out["recovery_frame"] = recovery
    return out