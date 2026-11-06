"""Actual-image causal filtering with frozen observation scores and visual motion.

Tune temporal settings only on validation; --selection freezes them for other
routes. No future observations, route waypoints, true position or simulator
velocity enter the filter. GT is used only for validation selection and reports.
"""
import argparse
from dataclasses import asdict
import hashlib
import itertools
import json
from pathlib import Path

import cv2
import numpy as np
from .learned_experiment import load_route
from .optimize import metrics, predictions
from .visual_motion import frame_geometry, estimate_motion, empty_motion
from ..temporal.visual_filter import VisualFilterConfig, VisualPositionFilter


def run_filter(scores, xy, identities, motions, config, *, zero_displacement=False):
    filter_ = VisualPositionFilter(xy, VisualFilterConfig(**config))
    result, variance = [], []
    for ll, identity, motion in zip(scores, identities, motions):
        if identity['index'] == 0:
            filter_.reset()
        if zero_displacement:
            motion = dict(motion, delta_world_m=[0., 0.])
        mean, var = filter_.update(ll, motion)
        result.append(mean)
        variance.append(var)
    return np.asarray(result), variance


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--scores', type=Path, required=True)
    p.add_argument('--fusion', type=Path, default=Path('map2loc/simulator/fusion_selection.json'))
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--selection', type=Path)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=False)
    cv2.setNumThreads(4)
    fusion = json.loads(a.fusion.read_text())
    source = json.loads((a.scores/'manifest.json').read_text())
    if Path(source['data']).resolve() != a.data.resolve():
        raise ValueError('Scores do not belong to this route')
    if source['checkpoint_sha256'] != fusion['checkpoint_sha256']:
        raise ValueError('Wrong learned checkpoint in score cache')
    records = json.loads((a.scores/'predictions.json').read_text())
    cache = np.load(a.scores/'scores.npz')
    xy, learned, costs = cache['grid_xy'], cache['learned_logits'], cache['geometry_cost']
    cfg = fusion['config']
    scores = learned*cfg['learned_weight'] - costs/cfg['geometry_temperature']
    independent = predictions(learned, costs, xy, cfg)
    # Observation identity and supervision are kept separate from motion inputs.
    gt = np.asarray([r['gt_map_px'] for r in records])
    motions = []
    last_samples = None
    for alpha in (0., .02, .05):
        samples = load_route(a.data, alpha, 'motion')
        previous = None
        for s in samples:
            row = s[4]
            identity = records[len(motions)]
            if (identity['alpha'], identity['split'], identity['index']) != (alpha, row['split'], row['index']):
                raise ValueError('Frame order differs from cached observations')
            current = frame_geometry(s[3], row['state']['camera_imu']['quat'])
            if row['index'] == 0:
                previous = None
            motion = empty_motion('new_pass') if previous is None else estimate_motion(previous, current)
            motions.append(motion)
            previous = current
        last_samples = samples
        print('visual motion', alpha, sum(m['accepted'] for m in motions[-len(samples):]),
              '/', len(samples)-2, flush=True)
    if len(motions) != len(scores):
        raise ValueError('Scores and images have different lengths')
    (a.out/'motion.json').write_text(json.dumps(motions, indent=2))
    fusion_hash = hashlib.sha256(a.fusion.read_bytes()).hexdigest()
    if a.selection:
        selected = json.loads(a.selection.read_text())
        if Path(selected['validation_data']).resolve() == a.data.resolve():
            raise ValueError('Final evaluation must use a different route')
        if selected['fusion_sha256'] != fusion_hash:
            raise ValueError('Observation model changed after temporal selection')
        config = selected['config']
    else:
        candidates = []
        for motion_sigma, missing_sigma, weight, mix, power, carry in itertools.product(
                (.1, .25, .5), (.4, .8, 1.4), (.5, 1.), (.001, .03), (.8, 1.), (False, True)):
            config = asdict(VisualFilterConfig(motion_sigma, missing_sigma, weight, mix, power, carry))
            pred, _ = run_filter(scores, xy, records, motions, config)
            candidates.append(dict(config=config, metrics=metrics(pred, gt)))
        selected = min(candidates, key=lambda r:r['metrics']['mse_m2'])
        config = selected['config']
        selected.update(validation_data=str(a.data), fusion_sha256=fusion_hash,
                        checkpoint_sha256=source['checkpoint_sha256'])
        (a.out/'selection.json').write_text(json.dumps(selected, indent=2))
        (a.out/'validation_candidates.json').write_text(json.dumps(candidates, indent=2))
    temporal, variances = run_filter(scores, xy, records, motions, config)
    zero, _ = run_filter(scores, xy, records, motions, config, zero_displacement=True)
    summary, output = {}, []
    for alpha in (0., .02, .05):
        ids = np.array([r['alpha']==alpha for r in records])
        valid = np.array([m['accepted'] and r['index']>0 for r,m in zip(records,motions)]) & ids
        deltas = np.array([m['delta_world_m'] for m in motions])
        gt_delta = (gt - np.roll(gt, 1, axis=0))/100*np.array([1., -1.])
        motion_error = np.linalg.norm(deltas[valid]-gt_delta[valid],axis=1)
        summary[str(alpha)] = dict(independent=metrics(independent[ids],gt[ids]),
                                  temporal=metrics(temporal[ids],gt[ids]),
                                  zero_displacement=metrics(zero[ids],gt[ids]),
                                  accepted_motion=int(valid.sum()), transitions=int(sum(r['index']>0 for r in np.asarray(records,dtype=object)[ids])),
                                  motion_error_median_m=float(np.median(motion_error)) if len(motion_error) else None,
                                  motion_error_p90_m=float(np.quantile(motion_error,.9)) if len(motion_error) else None)
    for i,r in enumerate(records):
        output.append(dict(alpha=r['alpha'],index=r['index'],split=r['split'],
                           gt_map_px=gt[i].tolist(),independent_map_px=independent[i].tolist(),
                           temporal_map_px=temporal[i].tolist(),zero_displacement_map_px=zero[i].tolist(),
                           posterior_variance_m2=variances[i],motion=motions[i]))
    (a.out/'predictions.json').write_text(json.dumps(output,indent=2))
    (a.out/'metrics.json').write_text(json.dumps(summary,indent=2))
    (a.out/'manifest.json').write_text(json.dumps(dict(data=str(a.data),scores=str(a.scores),
        config=config,causal=True,reset='only start of each route pass',
        inputs='current/past RGB, head quaternions, fixed height, frozen image likelihoods',
        simulator_velocity_used=False,ground_truth_initialization=False,
        future_frames_used=False,mse_definition='mean((dx_m**2+dy_m**2)/2)',
        checkpoint_sha256=source['checkpoint_sha256'],fusion_sha256=fusion_hash),indent=2))
    print('selected',config,flush=True)
    print(json.dumps(summary,indent=2),flush=True)
    field=cv2.resize(cv2.imread('res/example.png'),(848,624))
    video=cv2.VideoWriter(str(a.out/'history_comparison.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),6,(1500,760))
    offset=len(records)-len(last_samples)
    for i,s in enumerate(last_samples):
        j=i+offset;c=np.full((760,1500,3),245,np.uint8);c[60:684,:848]=field
        def point(v):return (int(v[0]*.8),int(v[1]*.8)+60)
        cv2.drawMarker(c,point(gt[j]),(0,220,255),cv2.MARKER_CROSS,24,3)
        cv2.circle(c,point(independent[j]),9,(50,50,230),3)
        cv2.circle(c,point(temporal[j]),9,(255,200,0),3)
        c[90:570,860:1500]=cv2.cvtColor(s[3].astype('uint8'),cv2.COLOR_RGB2BGR)
        ei=np.linalg.norm(independent[j]-gt[j])/100;et=np.linalg.norm(temporal[j]-gt[j])/100
        m=motions[j]
        text_rows=[(f'Causal history | {a.data.name} | exposure 0.05',20,35),
                   (f'Independent {ei:.2f} m | history {et:.2f} m',875,615),
                   (f'Flow accepted: {m["accepted"]} | inliers: {m["inliers"]}',875,655),
                   ('Yellow: GT | Red: independent | Cyan: causal history',20,720)]
        for text,x,y in text_rows:cv2.putText(c,text,(x,y),cv2.FONT_HERSHEY_SIMPLEX,.65,(25,25,25),2,cv2.LINE_AA)
        video.write(c)
        if i==8:cv2.imwrite(str(a.out/'poster.png'),c)
    video.release()

if __name__=='__main__':main()
