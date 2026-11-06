"""Fine-tune the actual upstream OrienterNet on recorded URSoccerLab RGB.

Only dataset conversion and the training/evaluation driver live here. The
projection, encoders, matching and NLL are imported from the pinned submodule.
"""
import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[2]
UPSTREAM = ROOT / 'third_party/OrienterNet'
sys.path.insert(0, str(UPSTREAM))
from maploc.models.orienternet import OrienterNet
from maploc.data.image import rectify_image, resize_image, pad_image
from maploc.utils.wrappers import Camera
from maploc.models.voting import argmax_xyr, mask_yaw_prior
from map2loc.simulator.localize import rotation
from map2loc.simulator.motion_blur import blur_from_motion

# Canvas is 11.2 x 8.0 m, including zero padding beyond the original texture.
# 56 x 40 cells: divisible by the map encoder's 8x downsampling.
PPM, WIDTH, HEIGHT = 5., 56, 40
ORIGIN = np.array([-5.6, 4.0])


def uv_from_xy(xy):
    return (np.asarray(xy) - ORIGIN) * np.array([PPM, -PPM]) - .5


def xy_from_uv(uv):
    return (np.asarray(uv) + .5) / np.array([PPM, -PPM]) + ORIGIN


def semantic_map():
    rgb = cv2.cvtColor(cv2.imread(str(ROOT/'res/example.png')), cv2.COLOR_BGR2RGB)
    # Class IDs describe soccer semantics, not invented correspondences to OSM.
    green = (rgb[:,:,1] > rgb[:,:,0]*1.2) & (rgb[:,:,1] > rgb[:,:,2]*1.2)
    white = (rgb.min(2) > 180)
    raster = np.zeros((3, HEIGHT, WIDTH), np.int64)
    for v in range(HEIGHT):
        for u in range(WIDTH):
            x, y = xy_from_uv([u,v])
            px, py = int(round((x+5.3)*100)), int(round((3.9-y)*100))
            x0,x1 = max(0,px-10),min(rgb.shape[1],px+10)
            y0,y1 = max(0,py-10),min(rgb.shape[0],py+10)
            if x1 > x0 and y1 > y0:
                raster[0,v,u] = green[y0:y1,x0:x1].mean() > .25
                raster[1,v,u] = white[y0:y1,x0:x1].any()
    return torch.from_numpy(raster)


def load_route(path, alpha):
    samples=[]
    previous=previous_quat=None
    for row in map(json.loads, (path/'frames.jsonl').read_text().splitlines()):
        if row['index']==0:
            previous=previous_quat=None
        rgb=cv2.cvtColor(cv2.imread(str(path/row['image'])),cv2.COLOR_BGR2RGB).astype(np.float32)
        quat=row['state']['camera_imu']['quat']
        corrupted,_=blur_from_motion(rgb,previous,quat,previous_quat,alpha)
        previous,previous_quat=rgb,quat
        R=rotation(quat)
        # Optical camera basis: right=-head Y, down=-head Z, forward=head X.
        Rwc=R @ np.array([[0,0,1],[-1,0,0],[0,-1,0]])
        up=Rwc.T @ np.array([0.,0.,1.])
        roll=np.degrees(np.arctan2(up[0],-up[1]))
        pitch=np.degrees(np.arcsin(np.clip(-up[2],-1,1)))
        yaw=np.degrees(np.arctan2(R[0,0],R[1,0])) % 360
        camera=Camera.from_dict(dict(model='PINHOLE',width=640,height=480,
                                    params=[320/np.tan(np.pi/6)]*2+[320.,240.])).float()
        image=torch.from_numpy(np.clip(corrupted,0,255)).permute(2,0,1)/255
        # Upstream gravity rectification, NOT inverse ground-plane projection.
        image,valid=rectify_image(image,camera,roll,pitch)
        image,_,camera,valid=resize_image(image,(320,240),camera=camera,valid=valid)
        image,valid,camera=pad_image(image,(320,256),camera=camera,valid=valid)
        # Ground truth is isolated here for labels and metrics only.
        baseR=rotation(np.roll(row['actual']['rotation_quat_xyzw'],1))
        xy=np.array(row['actual']['translation_m'][:2])+(baseR@np.array([-.005,0,.2916]))[:2]+(R@np.array([.06,.05,.045]))[:2]
        samples.append(dict(image=image,camera=camera,uv=torch.tensor(uv_from_xy(xy),dtype=torch.float32),
                            roll_pitch_yaw=torch.tensor([0.,0.,yaw]),valid=valid,
                            identity=dict(route=path.name,split=row['split'],index=row['index'],image=row['image'],alpha=alpha),
                            visible_fraction=float(valid.float().mean())))
    return samples


def batch(samples, ids, raster, device):
    # Labels deliberately omitted from the model's inference inputs.
    data=dict(image=torch.stack([samples[i]['image'] for i in ids]).to(device),
              camera=torch.stack([samples[i]['camera'] for i in ids]).to(device),
              map=raster[None].expand(len(ids),-1,-1,-1).to(device))
    labels={k:torch.stack([samples[i][k] for i in ids]).to(device) for k in ('uv','roll_pitch_yaw')}
    return data, labels


def evaluate(model,samples,raster,out):
    model.eval();records=[]
    with torch.no_grad():
        for i,s in enumerate(samples):
            data,labels=batch(samples,[i],raster,'cuda')
            pred=model(data)
            # Report both paper-style full pose search and allowed measured yaw.
            constrained=pred['scores'].clone()
            yaw=labels['roll_pitch_yaw'][:,-1]
            mask_yaw_prior(constrained,torch.stack([yaw,torch.full_like(yaw,5)],-1),model.conf.num_rotations)
            uvr_known=argmax_xyr(constrained)[0].cpu().numpy()
            uvr=pred['uvr_max'][0].cpu().numpy()
            gt=s['uv'].numpy()
            record=dict(**s['identity'],gt_xy_m=xy_from_uv(gt).tolist(),
                        predicted_xy_m=xy_from_uv(uvr[:2]).tolist(),
                        known_yaw_xy_m=xy_from_uv(uvr_known[:2]).tolist(),
                        error_m=float(np.linalg.norm(uvr[:2]-gt)/PPM),
                        known_yaw_error_m=float(np.linalg.norm(uvr_known[:2]-gt)/PPM),
                        yaw_error_deg=float(abs((uvr[2]-float(yaw[0])+180)%360-180)),
                        nll=float(model.loss(pred,labels)['total'].mean()),
                        visible_fraction=s['visible_fraction'])
            records.append(record)
    metrics={}
    for alpha in sorted({r['alpha'] for r in records}):
        rows=[r for r in records if r['alpha']==alpha]; result={}
        for name,key in [('full_pose','error_m'),('measured_yaw_prior_5deg','known_yaw_error_m')]:
            e=np.array([r[key] for r in rows])
            result[name]=dict(frames=len(e),mse_m2=float((e*e).mean()),median_m=float(np.median(e)),
                              mean_m=float(e.mean()),within_05m=float((e<.5).mean()))
        result['mean_nll']=float(np.mean([r['nll'] for r in rows]))
        result['median_yaw_error_deg']=float(np.median([r['yaw_error_deg'] for r in rows]))
        metrics[str(alpha)]=result
    out.mkdir(parents=True,exist_ok=True)
    (out/'predictions.json').write_text(json.dumps(records,indent=2))
    (out/'metrics.json').write_text(json.dumps(metrics,indent=2))
    print(out,metrics,flush=True)
    return metrics


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--train',type=Path,nargs='+',required=True)
    p.add_argument('--validation',type=Path,required=True)
    p.add_argument('--eval',type=Path,nargs='+',required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--steps',type=int,default=300)
    p.add_argument('--batch-size',type=int,default=2)
    p.add_argument('--seed',type=int,default=7)
    a=p.parse_args()
    routes=[r.resolve() for r in a.train+[a.validation]+a.eval]
    if len(routes)!=len(set(routes)):raise ValueError('Train, validation and evaluation routes must be disjoint')
    a.out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4);cv2.setNumThreads(4)
    torch.manual_seed(a.seed);np.random.seed(a.seed)
    ckpt=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
    conf=OmegaConf.create(OmegaConf.to_container(ckpt['hyper_parameters'].model,resolve=True))
    conf.name='orienternet';conf.image_encoder.backbone.pretrained=False
    # Spatial/semantic changes are dataset adaptations; all neural blocks remain upstream.
    conf.pixel_per_meter=PPM;conf.z_max=8;conf.x_max=8
    conf.num_rotations=64
    conf.map_encoder.num_classes=dict(areas=1,ways=1,nodes=0)
    model=OrienterNet(conf)
    state={k.removeprefix('model.'):v for k,v in ckpt['state_dict'].items()}
    reset={f'map_encoder.embeddings.{k}.weight' for k in ('areas','ways','nodes')}
    incompatible=model.load_state_dict({k:v for k,v in state.items() if k not in reset},strict=False)
    assert set(incompatible.missing_keys)==reset and not incompatible.unexpected_keys,incompatible
    del ckpt,state
    model=model.cuda()
    manifest=dict(upstream_commit=subprocess.check_output(['git','-C',str(UPSTREAM),'rev-parse','HEAD'],text=True).strip(),
                  checkpoint_sha256=hashlib.file_digest(a.checkpoint.open('rb'),'sha256').hexdigest(),
                  model_class=f'{type(model).__module__}.{type(model).__name__}',
                  config=OmegaConf.to_container(conf,resolve=True),reset_parameters=sorted(reset),
                  train=[str(r) for r in a.train],validation=str(a.validation),evaluation=[str(r) for r in a.eval],
                  seed=a.seed,steps=a.steps,batch_size=a.batch_size,optimizer='Adam',lr=1e-4,
                  device=torch.cuda.get_device_name(),torch=torch.__version__,
                  supervision='camera XY and yaw only in NLL and metrics; inference receives RGB, intrinsics, semantic map',
                  sequence='not yet evaluated: official sequence fusion requires an external relative-pose trajectory')
    (a.out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    raster=semantic_map();np.save(a.out/'semantic_map.npy',raster.numpy())
    train=[s for r in a.train for alpha in (0.,.04) for s in load_route(r,alpha)]
    val=load_route(a.validation,.05)
    evaluate(model,val,raster,a.out/'initial_validation')
    opt=torch.optim.Adam(model.parameters(),lr=1e-4)
    best=float('inf');start=time.monotonic()
    with (a.out/'training.jsonl').open('w') as log:
        for step in range(1,a.steps+1):
            model.train()
            ids=np.random.randint(len(train),size=a.batch_size)
            data,labels=batch(train,ids,raster,'cuda')
            pred=model(data);loss=model.loss(pred,labels)['total'].mean()
            if not torch.isfinite(loss):raise RuntimeError(f'Non-finite NLL at step {step}')
            opt.zero_grad(set_to_none=True);loss.backward();opt.step()
            if step==1 or step%10==0:
                row=dict(step=step,nll=float(loss.detach()),seconds=time.monotonic()-start)
                print(row,flush=True);log.write(json.dumps(row)+'\n');log.flush()
            if step%100==0 or step==a.steps:
                metrics=evaluate(model,val,raster,a.out/f'validation_{step:04}')
                nll=metrics['0.05']['mean_nll']
                if nll<best:
                    best=nll
                    torch.save(dict(state_dict=model.state_dict(),config=OmegaConf.to_container(conf,resolve=True),step=step),a.out/'best.pt')
    selected=torch.load(a.out/'best.pt',map_location='cpu',weights_only=False)
    model.load_state_dict(selected['state_dict'],strict=True)
    # Official pretrained evaluation protocol uses 256 rotations vs 64 for training.
    conf.num_rotations=256
    eval_model=OrienterNet(conf).cuda()
    eval_model.load_state_dict(model.state_dict(),strict=True)
    del model,opt,selected
    for route in a.eval:
        samples=[s for alpha in (0.,.02,.05) for s in load_route(route,alpha)]
        evaluate(eval_model,samples,raster,a.out/'evaluation_256rot'/route.name)
    manifest['evaluation_rotations']=256
    manifest['best_validation_nll']=best
    (a.out/'manifest.json').write_text(json.dumps(manifest,indent=2))

if __name__=='__main__':main()
