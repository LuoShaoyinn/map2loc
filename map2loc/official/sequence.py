"""Official RigidAligner with explicitly simulated noisy velocity odometry.

No custom belief dynamics, motion prior, posterior tempering or geometric
matching. Camera position differences are ONLY a simulated odometry source,
explicitly authorized for this experiment; absolute XY never enters aligner.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from .urs import UPSTREAM, PPM, WIDTH, HEIGHT, ORIGIN  # adds pinned upstream path
from maploc.models.sequential import RigidAligner
from maploc.models.voting import mask_yaw_prior
from maploc.osm.raster import Canvas
from maploc.utils.geo import BoundaryBox


def simulate_odometry(records, sigma, seed):
    rng=np.random.default_rng(seed)
    trajectory=[];velocity_log=[];xy=np.zeros(2)
    for i,row in enumerate(records):
        if row['index']==0:
            xy=np.zeros(2);dt=0.;velocity=np.zeros(2);noise=np.zeros(2)
        else:
            prev=records[i-1]
            if row['split']!=prev['split']:raise ValueError('Unmarked pass boundary')
            dt=row['timestamp']-prev['timestamp']
            if dt<=0:raise ValueError('Camera times must increase within each pass')
            velocity=(np.array(row['gt_xy_m'])-prev['gt_xy_m'])/dt
            noise=rng.normal(0,sigma,2)
            xy=xy+(velocity+noise)*dt
        trajectory.append(xy.copy())
        velocity_log.append(dict(dt_s=dt,noisy_camera_velocity_mps=(velocity+noise).tolist(),noise_mps=noise.tolist()))
    return np.asarray(trajectory),velocity_log


def fuse(observations, records, odometry, window, known_yaw):
    # This boundary receives relative odometry, never the absolute XY labels.
    canvas=Canvas(BoundaryBox([-5.6,-4.],[5.6,4.]),PPM)
    assert (canvas.h,canvas.w)==(HEIGHT,WIDTH)
    result=[];aligner=None
    with torch.no_grad():
        for obs, identity, xy in zip(observations,records,odometry):
            if identity['index']%window==0:
                aligner=RigidAligner()
            score=torch.from_numpy(obs).cuda()
            yaw=torch.tensor(identity['measured_yaw_deg'],device='cuda',dtype=torch.float32)
            if known_yaw:
                score=score.clone()
                mask_yaw_prior(score[None],torch.stack([yaw,yaw.new_tensor(5)])[None],score.shape[-1])
            position=torch.tensor(xy,device='cuda',dtype=torch.float32)
            aligner.update(score,canvas,position,yaw)
            if not torch.isfinite(aligner.belief).any():
                raise RuntimeError('No feasible sequence alignment; do not silently reset')
            aligner.compute()
            location,heading=aligner.transform(position,yaw)
            result.append(location.cpu().numpy())
    return np.asarray(result)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('evaluation',type=Path);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--sigma-mps',type=float,default=.05)
    p.add_argument('--seed',type=int,default=17);p.add_argument('--window',type=int,default=10)
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4)
    records=json.loads((a.evaluation/'predictions.json').read_text())
    observations=np.load(a.evaluation/'observations.npz')['log_probs']
    assert len(records)==len(observations)
    summary={};output=[]
    for alpha in sorted({r['alpha'] for r in records}):
        ids=[i for i,r in enumerate(records) if r['alpha']==alpha]
        rows=[records[i] for i in ids]
        odometry,velocity=simulate_odometry(rows,a.sigma_mps,a.seed)
        # Only identity/rotation go into fusion. GT is retained for evaluation.
        identities=[{k:r[k] for k in ('index','measured_yaw_deg')} for r in rows]
        full=fuse(observations[ids],identities,odometry,a.window,False)
        prior=fuse(observations[ids],identities,odometry,a.window,True)
        gt=np.array([r['gt_xy_m'] for r in rows]);metrics={}
        for name,points in [('full_pose_sequence',full),('measured_yaw_sequence',prior),
                            ('full_pose_single',np.array([r['predicted_xy_m'] for r in rows])),
                            ('measured_yaw_single',np.array([r['known_yaw_xy_m'] for r in rows]))]:
            error=np.linalg.norm(points-gt,axis=1)
            metrics[name]=dict(frames=len(error),mse_m2=float(np.mean(error**2)),median_m=float(np.median(error)),within_05m=float(np.mean(error<.5)))
        for i,r in enumerate(rows):
            output.append(dict(**r,sequence_xy_m=full[i].tolist(),known_yaw_sequence_xy_m=prior[i].tolist(),
                               odometry_relative_xy_m=odometry[i].tolist(),velocity=velocity[i]))
        summary[str(alpha)]=metrics
        print(alpha,metrics,flush=True)
    (a.out/'metrics.json').write_text(json.dumps(summary,indent=2))
    (a.out/'predictions.json').write_text(json.dumps(output,indent=2))
    (a.out/'manifest.json').write_text(json.dumps(dict(source=str(a.evaluation),
        source_predictions_sha256=hashlib.sha256((a.evaluation/'predictions.json').read_bytes()).hexdigest(),
        method='unmodified maploc.models.sequential.RigidAligner',
        mode='causal prefixes within non-overlapping 10-frame chunks' if a.window==10 else f'causal prefixes within {a.window}-frame chunks',
        motion='finite differences of camera optical-center ground truth over received camera timestamps; additive iid Gaussian velocity noise; integrate from zero each pass',
        sigma_mps_per_axis=a.sigma_mps,seed=a.seed,window=a.window,
        attitude='exact simulator head quaternion; rotation noise zero',
        limitation='simulated velocity odometry, not visual odometry or measured physical velocity; no process uncertainty modeled by upstream rigid alignment'),indent=2))

if __name__=='__main__':main()
