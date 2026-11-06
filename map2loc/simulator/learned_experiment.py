"""Train/evaluate this repository's OrienterNet-style BEV correlator on URS.

Dynamic motion blur is applied before rectification using image flow.
Use --corruption ghost to reproduce the older temporal-persistence experiment.
Motion blur approximates exposure between sampled views; it is not rendered
sub-frame ground truth. State resets per route pass.
"""
import argparse,json,time
from pathlib import Path
import cv2
import numpy as np
import torch
from .localize import rotation
from .motion_blur import blur_from_motion
from ..matching.learned import LearnedMatcher,LearnedMatcherConfig
from ..matching.grid import PoseGrid


def rectify(image,quat):
    R=rotation(quat)
    height=.56+.2441+.0475+(R@np.array([.06,.05,.045]))[2]
    yy,xx=np.mgrid[:64,:64]
    world=np.stack([(xx-32)*.08,-(yy-32)*.08,np.full_like(xx,-height,dtype=float)],-1)
    local=world@R
    f=320/np.tan(np.pi/6)
    depth=local[:,:,0]
    u=319.5-f*local[:,:,1]/np.maximum(depth,1e-6)
    v=239.5-f*local[:,:,2]/np.maximum(depth,1e-6)
    mask=(depth>.05)&(u>=0)&(u<639)&(v>=0)&(v<479)
    patch=cv2.remap(image,u.astype('float32'),v.astype('float32'),cv2.INTER_LINEAR)
    patch[~mask]=0
    return patch,mask.astype('float32')


def load_route(root,alpha,corruption="motion"):
    rows=[json.loads(x) for x in (root/'frames.jsonl').read_text().splitlines()]
    samples=[];ghost=None;previous=None;previous_quat=None
    for row in rows:
        image=cv2.cvtColor(cv2.imread(str(root/row['image'])),cv2.COLOR_BGR2RGB).astype('float32')
        if row['index']==0:ghost=None;previous=None;previous_quat=None
        quat=row['state']['camera_imu']['quat']
        if corruption == 'motion':
            ghost,stats=blur_from_motion(image,previous,quat,previous_quat,alpha)
            row['blur_stats']=stats
        else:
            ghost=image if ghost is None else (1-alpha)*image+alpha*ghost
        previous=image;previous_quat=quat
        patch,mask=rectify(ghost,quat)
        # Labels are for training/evaluation only, never passed to inference.
        R=rotation(quat);baseR=rotation(np.roll(row['actual']['rotation_quat_xyzw'],1))
        gt=np.array(row['actual']['translation_m'][:2])+(baseR@np.array([-.005,0,.2916]))[:2]+(R@np.array([.06,.05,.045]))[:2]
        xy=np.array([(gt[0]+5.3)*100,(3.9-gt[1])*100],dtype='float32')
        samples.append((patch,mask,xy,ghost.copy(),row))
    return samples


def tensor_batch(samples,indices,device):
    bev=torch.from_numpy(np.stack([samples[i][0] for i in indices])).permute(0,3,1,2).to(device)/127.5-1
    mask=torch.from_numpy(np.stack([samples[i][1] for i in indices])).to(device)
    xy=torch.from_numpy(np.stack([samples[i][2] for i in indices])).to(device)
    return bev,mask,xy


def main():
    p=argparse.ArgumentParser();p.add_argument('--train',type=Path,nargs='+',required=True);p.add_argument('--eval',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--steps',type=int,default=300);p.add_argument('--seed',type=int,default=7);p.add_argument('--corruption',choices=('motion','ghost'),default='motion');a=p.parse_args()
    if a.eval.resolve() in [r.resolve() for r in a.train]:raise ValueError('Evaluation route must be held out')
    a.out.mkdir(parents=True,exist_ok=False)
    torch.manual_seed(a.seed);np.random.seed(a.seed);torch.set_num_threads(4)
    device='cuda' if torch.cuda.is_available() else 'cpu'
    print('device',device,torch.cuda.get_device_name(0) if device=='cuda' else '',flush=True)
    cfg=LearnedMatcherConfig(patch_px=64,map_px_per_px=8.,grid_resolution=16.,obs_stride=2,feature_dim=16,base_channels=16,n_yaw=360)
    grid=PoseGrid(1060,780,resolution=16,n_yaw=360)
    model=LearnedMatcher(cfg,grid).to(device)
    # Raster samples exactly at grid cell centers, including edge padding.
    raw=cv2.cvtColor(cv2.imread('res/example.png'),cv2.COLOR_BGR2RGB)
    gy,gx=np.meshgrid(grid.ys,grid.xs,indexing='ij')
    raster=cv2.remap(raw,gx.astype('float32'),gy.astype('float32'),cv2.INTER_LINEAR,borderMode=cv2.BORDER_REPLICATE)
    map_tensor=torch.from_numpy(raster).permute(2,0,1)[None].float().to(device)/127.5-1
    train=[]
    for root in a.train:
        for alpha in (0., .04 if a.corruption == 'motion' else .6):
            train.extend(load_route(root,alpha,a.corruption))
    opt=torch.optim.AdamW(model.parameters(),lr=.001)
    log=[];start=time.monotonic()
    for step in range(a.steps):
        ids=np.random.randint(len(train),size=8);bev,mask,xy=tensor_batch(train,ids,device)
        out=model(bev,model.encode_map(map_tensor),valid_mask=mask)
        nll=model.nll(out['xy_logits'],xy);loss=nll+.2*out['mask_loss']
        opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5);opt.step()
        if step%25==0 or step==a.steps-1:
            item=dict(step=step+1,nll=float(nll.detach()),seconds=time.monotonic()-start);log.append(item);print(item,flush=True)
    torch.save(dict(state_dict=model.state_dict(),config=cfg.to_dict(),train=[str(r) for r in a.train],seed=a.seed,steps=a.steps),a.out/'model.pt')
    (a.out/'training.json').write_text(json.dumps(log,indent=2))
    model.eval();summary={};allpred=[]
    with torch.no_grad():
        mf=model.encode_map(map_tensor)
        for alpha in ((0.,.02,.05) if a.corruption == 'motion' else (0.,.35,.65)):
            samples=load_route(a.eval,alpha,a.corruption);errors=[]
            writer=cv2.VideoWriter(str(a.out/f'replay_alpha{alpha:.2f}.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),6,(1500,760))
            field=cv2.resize(cv2.imread('res/example.png'),(848,624))
            for i,sample in enumerate(samples):
                bev,mask,_=tensor_batch(samples,[i],device)
                out=model(bev,mf,valid_mask=mask);idx=int(out['xy_logits'].flatten().argmax())
                px=np.array([(idx%grid.nx+.5)*16,(idx//grid.nx+.5)*16]);gt=sample[2]
                err=float(np.linalg.norm(px-gt)/100);errors.append(err)
                row=sample[4];allpred.append(dict(alpha=alpha,corruption=a.corruption,blur_stats=row.get('blur_stats'),index=row['index'],split=row['split'],image=row['image'],predicted_map_px=px.tolist(),gt_map_px=gt.tolist(),error_m=err))
                c=np.full((760,1500,3),245,np.uint8);c[60:684,:848]=field
                def point(v):return (int(v[0]*.8),int(v[1]*.8)+60)
                cv2.drawMarker(c,point(gt),(0,220,255),cv2.MARKER_CROSS,24,3);cv2.circle(c,point(px),9,(255,200,0),3)
                c[90:570,860:1500]=cv2.cvtColor(np.clip(sample[3],0,255).astype('uint8'),cv2.COLOR_RGB2BGR)
                for text,y in [(f'Learned BEV correlator | unseen route seed | input {a.corruption} strength={alpha:.2f}',35),(f'{row["split"]} frame {row["index"]+1} | camera XY error {err:.2f} m',620),('Yellow: GT   Cyan: estimate | simulator quaternion; no velocity input',720)]:
                    cv2.putText(c,text,((875 if y==620 else 20),y),cv2.FONT_HERSHEY_SIMPLEX,(.55 if y==620 else .8),(25,25,25),2,cv2.LINE_AA)
                writer.write(c)
                if i==8:cv2.imwrite(str(a.out/f'poster_alpha{alpha:.2f}.png'),c)
            writer.release();e=np.array(errors)
            summary[str(alpha)]=dict(frames=len(e),median_m=float(np.median(e)),mean_m=float(e.mean()),p90_m=float(np.quantile(e,.9)),within_05m=float((e<.5).mean()))
            print(alpha,summary[str(alpha)],flush=True)
    (a.out/'metrics.json').write_text(json.dumps(summary,indent=2));(a.out/'predictions.json').write_text(json.dumps(allpred,indent=2))
    (a.out/'manifest.json').write_text(json.dumps(dict(method='repository OrienterNet-style LearnedMatcher; not official OrienterNet',train=[str(r) for r in a.train],evaluation=str(a.eval),seed=a.seed,steps=a.steps,device=device,corruption=a.corruption,corruption_definition=('current-frame linear-light exposure integration along attitude-assisted dense image flow; strength is exposure fraction of sample interval' if a.corruption=='motion' else 'G[t]=(1-alpha)*I[t]+alpha*G[t-1]; reset per pass'),height_assumption_m=.56),indent=2))

if __name__=='__main__':main()
