"""Collect a privileged stop-and-observe route and evaluate image retrieval.

Run on the simulator host. Ground truth is used only for database labels and
evaluation; query descriptors consume RGB alone. This is a visual-map baseline,
not yet the planar-map matcher. No gait or odometry is required.
"""
import argparse
import json
import math
from pathlib import Path
import time

import cv2
import numpy as np


def descriptor(rgb):
    small = cv2.resize(rgb, (64, 48)).astype(np.float32) / 255
    gray = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)
    edges = cv2.Sobel(gray, cv2.CV_32F, 1, 0)
    vector = np.concatenate([small.ravel(), edges.ravel()])
    vector -= vector.mean()
    return vector / max(float(np.linalg.norm(vector)), 1e-8)


def checked(reply):
    if not reply.get('ok'):
        raise RuntimeError(str(reply))
    return reply.get('result', reply)


def main():
    from ursoccerlab import AdminClient, RobotClient
    from ursoccerlab.media import camera_to_rgb

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--host', default='127.0.0.1')
    p.add_argument('--port', type=int, default=10000)
    p.add_argument('--actor', default='robot_bp0')
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--points', type=int, default=32)
    p.add_argument('--settle', type=float, default=0.4)
    a = p.parse_args()
    if a.points < 4 or a.settle < 0.2:
        p.error('points must be >= 4 and settle >= 0.2 seconds')
    a.out.mkdir(parents=True, exist_ok=False)
    admin = AdminClient(a.host)
    robot = RobotClient(a.host, a.port)
    original = checked(admin.get_pose(a.actor))
    records, features = [], []
    try:
        # Joint order is obtained from the actual state, never guessed.
        state = None
        deadline = time.monotonic() + 15
        while state is None and time.monotonic() < deadline:
            for kind, value in robot.recv():
                if kind == 'state':
                    state = value
            time.sleep(.005)
        if state is None:
            raise TimeoutError('No robot state within 15 s')
        names = list(state['joints'])
        if len(names) != len(original['joint_qpos']):
            raise RuntimeError('State joint order/dimension does not match admin pose')
        yaw_idx, pitch_idx = names.index('head_yaw_joint'), names.index('head_pitch_joint')
        for split in ('reference', 'query'):
            for i in range(a.points):
                # Queries lie halfway between reference samples, with perturbed head pose.
                t = 2 * math.pi * (i + (0.5 if split == 'query' else 0)) / a.points
                x, y = 2.8 * math.cos(t), 1.7 * math.sin(t)
                yaw = math.atan2(-y, -x)
                head_yaw = .4 * math.sin(2*t) + (.06 if split == 'query' else 0)
                head_pitch = .25 + .12 * math.cos(3*t)
                joints = [0.] * len(names)
                joints[yaw_idx], joints[pitch_idx] = head_yaw, head_pitch
                pose = dict(translation_m=[x, y, .56],
                            rotation_quat_xyzw=[0, 0, math.sin(yaw/2), math.cos(yaw/2)],
                            joint_qpos=joints)
                checked(admin.lock_pose(a.actor, **pose))
                start = checked(admin.get_pose(a.actor))['sim_time_sec']
                deadline = time.monotonic() + 20
                frame = None
                while frame is None and time.monotonic() < deadline:
                    for kind, value in robot.recv():
                        if kind in ('rgb', 'camera'):
                            for im in value:
                                if im.get('camera_name') == 'left_eye' and im['sim_time'] >= start + a.settle:
                                    frame = im
                    time.sleep(.005)
                if frame is None:
                    raise TimeoutError('No fresh left-eye image within 20 s')
                actual = checked(admin.get_pose(a.actor))
                if np.linalg.norm(np.array(actual['translation_m']) - pose['translation_m']) > .01:
                    raise RuntimeError('Privileged pose did not hold')
                rgb = camera_to_rgb(frame)
                filename = f'{split}_{i:04d}.png'
                if not cv2.imwrite(str(a.out / filename), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)):
                    raise IOError(filename)
                row = dict(split=split, index=i, image=filename, command=pose,
                           actual=actual, camera={k:v for k,v in frame.items() if k not in ('data', '_rgb')})
                records.append(row)
                features.append(descriptor(rgb))
                with (a.out / 'frames.jsonl').open('a') as f:
                    f.write(json.dumps(row) + '\n')
                print(f'{split} {i+1}/{a.points}', flush=True)
    finally:
        try:
            checked(admin.unlock_pose(a.actor))
            checked(admin.set_pose(a.actor, **{k:original[k] for k in ('translation_m', 'rotation_quat_xyzw', 'joint_qpos')}))
        finally:
            robot.close()
            admin.close()
    ref = np.stack(features[:a.points])
    query = np.stack(features[a.points:])
    # No query position, orientation, index or commanded motion enters retrieval.
    nearest = (query @ ref.T).argmax(axis=1)
    xy = np.array([r['actual']['translation_m'][:2] for r in records])
    pred = xy[:a.points][nearest]
    error = np.linalg.norm(pred - xy[a.points:], axis=1)
    oracle = np.linalg.norm(xy[a.points:, None] - xy[None, :a.points], axis=2).min(axis=1)
    summary = dict(method='RGB normalized thumbnail and horizontal gradient retrieval',
                   target='robot base xy', units='meters', reference_frames=a.points,
                   query_frames=a.points, mean_error=float(error.mean()),
                   median_error=float(np.median(error)), p90_error=float(np.quantile(error,.9)),
                   within_0_5m=float(np.mean(error < .5)),
                   database_quantization_mean=float(oracle.mean()),
                   limitation='Same scene and overlapping route; no held-out environment or planar-map evaluation')
    (a.out / 'metrics.json').write_text(json.dumps(summary, indent=2)+'\n')
    np.savez_compressed(a.out / 'predictions.npz', predicted_xy=pred, gt_xy=xy[a.points:],
                        error_m=error, reference_index=nearest)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
