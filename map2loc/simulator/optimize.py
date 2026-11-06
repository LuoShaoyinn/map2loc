"""Validate geometric/learned fusion, freeze configuration, evaluate new routes.

MSE is mean((dx**2+dy**2)/2), in m². The learned checkpoint stays fixed.
Only RGB, measured head attitude, fixed height and map enter candidate scores.
"""
import argparse,hashlib,json
from pathlib import Path
import cv2
import numpy as np
import torch
from .learned_experiment import load_route,tensor_batch
from .localize import rotation,ground
from ..matching.learned import LearnedMatcher,LearnedMatcherConfig
from ..matching.grid import PoseGrid


def geometry_cost(image,quat,world_grid,distance):
    R=rotation(quat);height=.56+.2916+(R@np.array([.06,.05,.045]))[2]
    hsv=cv2.cvtColor(np.clip(image,0,255).astype('uint8'),cv2.COLOR_RGB2HSV)
    green=((hsv[:,:,0]>30)&(hsv[:,:,0]<95)&(hsv[:,:,1]>60)).astype('uint8')
    near=cv2.dilate(green,np.ones((15,15),np.uint8))>0
    white=(hsv[:,:,1]<75)&(hsv[:,:,2]>140)&near
    def points(mask,limit):
        v,u=np.where(mask);xy,good=ground(np.column_stack([u,v]),R,height);xy=xy[good]
        if len(xy)>limit:xy=xy[np.linspace(0,len(xy)-1,limit).astype(int)]
        return xy
    pos=points(white,600);neg=points(green>0,400)
    costs=np.zeros(len(world_grid),dtype='float32')
    if len(pos)<15:return costs,len(pos)
    for begin in range(0,len(world_grid),200):
        centers=world_grid[begin:begin+200,None]
        def sample(pts):
            xy=centers+pts[None]
            x=((xy[:,:,0]+5.3)*100).astype('float32');y=((3.9-xy[:,:,1])*100).astype('float32')
            return cv2.remap(distance,x,y,cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT,borderValue=.5)
        costs[begin:begin+200]=np.minimum(sample(pos),.4).mean(1)
        if len(neg):costs[begin:begin+200]+=.12*np.exp(-.5*(sample(neg)/.035)**2).mean(1)
    return costs,len(pos)


def predictions(logits,cost,xy,config):
    scores=logits*config['learned_weight']-cost/config['geometry_temperature']
    if config['estimator']=='map':return xy[scores.argmax(1)]
    scores-=scores.max(1,keepdims=True)
    p=np.exp(scores);p/=p.sum(1,keepdims=True)
    return p@xy


def metrics(pred,gt):
    e=np.linalg.norm((pred-gt)/100,axis=1)
    return dict(frames=len(e),mse_m2=float(np.mean(e**2)/2),mean_squared_distance_m2=float(np.mean(e**2)),rmse_distance_m=float(np.sqrt(np.mean(e**2))),median_m=float(np.median(e)),p90_m=float(np.quantile(e,.9)),within_05m=float(np.mean(e<.5)))


def main():
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--selection',type=Path);a=p.parse_args()
    a.out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(4)
    torch.manual_seed(7);cv2.setNumThreads(4)
    device='cuda' if torch.cuda.is_available() else 'cpu'
    ck=torch.load(a.checkpoint,map_location='cpu',weights_only=False);cfg=LearnedMatcherConfig.from_dict(ck['config'])
    grid=PoseGrid(1060,780,resolution=cfg.grid_resolution,n_yaw=cfg.n_yaw)
    model=LearnedMatcher(cfg,grid).to(device);model.load_state_dict(ck['state_dict']);model.eval()
    raw=cv2.cvtColor(cv2.imread('res/example.png'),cv2.COLOR_BGR2RGB)
    gy,gx=np.meshgrid(grid.ys,grid.xs,indexing='ij');xy=np.column_stack([gx.ravel(),gy.ravel()])
    world=np.column_stack([xy[:,0]/100-5.3,3.9-xy[:,1]/100])
    raster=cv2.remap(raw,gx.astype('float32'),gy.astype('float32'),cv2.INTER_LINEAR,borderMode=cv2.BORDER_REPLICATE)
    mt=torch.from_numpy(raster).permute(2,0,1)[None].float().to(device)/127.5-1
    distance=cv2.distanceTransform(1-(raw.min(2)>200).astype('uint8'),cv2.DIST_L2,3)/100
    manifest=dict(data=str(a.data),checkpoint=str(a.checkpoint),checkpoint_sha256=hashlib.sha256(a.checkpoint.read_bytes()).hexdigest(),device=device,mse_definition='mean((dx_m**2+dy_m**2)/2)',input='RGB + head quaternion + fixed height/calibration; no XY/velocity',code='frozen learned checkpoint plus full-view geometric scoring')
    logits=[];cost=[];labels=[];identities=[];last_samples=None
    with torch.no_grad():
        mf=model.encode_map(mt)
        for alpha in (0.,.02,.05):
            samples=load_route(a.data,alpha,'motion')
            for i,s in enumerate(samples):
                bev,mask,_=tensor_batch(samples,[i],device)
                scores=model(bev,mf,valid_mask=mask)['xy_logits'].flatten().cpu().numpy()
                costs,n=geometry_cost(s[3],s[4]['state']['camera_imu']['quat'],world,distance)
                logits.append(scores);cost.append(costs);labels.append(s[2]);identities.append(dict(alpha=alpha,index=s[4]['index'],split=s[4]['split'],line_points=n))
            last_samples=samples
            print('scored',alpha,len(samples),flush=True)
    logits=np.stack(logits);cost=np.stack(cost);gt=np.stack(labels)
    baseline=xy[logits.argmax(1)]
    if a.selection:
        chosen=json.loads(a.selection.read_text())
        if Path(chosen['validation_data']).resolve()==a.data.resolve():raise ValueError('Final evaluation must use a new route')
        if chosen['checkpoint_sha256']!=manifest['checkpoint_sha256']:raise ValueError('Checkpoint changed since selection')
        config=chosen['config']
    else:
        candidates=[]
        for weight in (0.,.1,.3,1.):
            for temp in (.015,.03,.06):
                for estimator in ('map','mean'):
                    config=dict(learned_weight=weight,geometry_temperature=temp,estimator=estimator)
                    result=metrics(predictions(logits,cost,xy,config),gt)
                    candidates.append(dict(config=config,metrics=result))
        # Neural-only expectation is also a legitimate MSE-oriented ablation.
        for weight in (.25,.5,1.):
            config=dict(learned_weight=weight,geometry_temperature=1e9,estimator='mean')
            candidates.append(dict(config=config,metrics=metrics(predictions(logits,cost,xy,config),gt)))
        chosen=min(candidates,key=lambda c:c['metrics']['mse_m2']);config=chosen['config']
        chosen.update(validation_data=str(a.data),checkpoint_sha256=manifest['checkpoint_sha256'])
        (a.out/'selection.json').write_text(json.dumps(chosen,indent=2))
        (a.out/'validation_candidates.json').write_text(json.dumps(candidates,indent=2))
    pred=predictions(logits,cost,xy,config);manifest['selected_config']=config
    summary={}
    for alpha in (0.,.02,.05):
        ids=np.array([v['alpha']==alpha for v in identities])
        summary[str(alpha)]=dict(baseline=metrics(baseline[ids],gt[ids]),optimized=metrics(pred[ids],gt[ids]))
    records=[dict(**v,gt_map_px=g.tolist(),baseline_map_px=b.tolist(),optimized_map_px=o.tolist()) for v,g,b,o in zip(identities,gt,baseline,pred)]
    (a.out/'metrics.json').write_text(json.dumps(summary,indent=2));(a.out/'predictions.json').write_text(json.dumps(records,indent=2));(a.out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    np.savez_compressed(a.out/'scores.npz',learned_logits=logits,geometry_cost=cost,grid_xy=xy)
    print('selected',config,flush=True);print(json.dumps(summary,indent=2),flush=True)
    # All three methods display the same current image, with no trajectory tails.
    writer=cv2.VideoWriter(str(a.out/'comparison.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),6,(1500,760));field=cv2.resize(cv2.imread('res/example.png'),(848,624))
    offset=len(pred)-len(last_samples)
    for i,s in enumerate(last_samples):
        c=np.full((760,1500,3),245,np.uint8);c[60:684,:848]=field
        def pt(v):return (int(v[0]*.8),int(v[1]*.8)+60)
        j=offset+i;cv2.drawMarker(c,pt(gt[j]),(0,220,255),cv2.MARKER_CROSS,24,3);cv2.circle(c,pt(baseline[j]),9,(50,50,230),3);cv2.circle(c,pt(pred[j]),9,(255,200,0),3)
        c[90:570,860:1500]=cv2.cvtColor(s[3].astype('uint8'),cv2.COLOR_RGB2BGR)
        eb=np.linalg.norm(baseline[j]-gt[j])/100;eo=np.linalg.norm(pred[j]-gt[j])/100
        for text,x,y in [('Frozen model vs optimized inference | motion exposure 0.05',20,35),(f'Baseline {eb:.2f} m | optimized {eo:.2f} m',875,620),('Yellow: GT | Red: baseline | Cyan: optimized',20,720)]:cv2.putText(c,text,(x,y),cv2.FONT_HERSHEY_SIMPLEX,.7,(25,25,25),2,cv2.LINE_AA)
        writer.write(c)
        if i==8:cv2.imwrite(str(a.out/'poster.png'),c)
    writer.release()

if __name__=='__main__':main()
