"""Ground-line map localization with quaternion-compensated optical flow.

Inputs to the estimator: RGB, head quaternion (wxyz), fixed root height and
MOS9 camera calibration. No commanded pose, base pose or velocity is used.
"""
import argparse,json
from pathlib import Path
import cv2
import numpy as np


def rotation(q):
    w,x,y,z=np.asarray(q)/np.linalg.norm(q)
    return np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],
                     [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],
                     [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])


def ground(uv,R,height):
    f=320/np.tan(np.deg2rad(30))
    rays=np.column_stack([np.ones(len(uv)),-(uv[:,0]-319.5)/f,-(uv[:,1]-239.5)/f])@R.T
    good=rays[:,2]<-.08
    xyz=rays*(-height/np.minimum(rays[:,2],-1e-6))[:,None]
    good &= np.linalg.norm(xyz[:,:2],axis=1)<7
    return xyz[:,:2],good


def main():
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    a.out.mkdir(parents=True,exist_ok=False)
    rows=[json.loads(s) for s in (a.data/'frames.jsonl').read_text().splitlines()]
    mapim=cv2.imread('res/example.png'); white=(mapim.min(2)>200).astype(np.uint8)
    dist=cv2.distanceTransform(1-white,cv2.DIST_L2,3)/100
    xs=np.arange(-4.5,4.501,.1);ys=np.arange(-3,3.001,.1)
    xx,yy=np.meshgrid(xs,ys);grid=np.column_stack([xx.ravel(),yy.ravel()])
    prev=None;prob=None;results=[]
    for row in rows:
        if row['index']==0:prev=None;prob=None
        im=cv2.imread(str(a.data/row['image']));gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
        R=rotation(row['state']['camera_imu']['quat'])
        # Known fixed root height and fixed head-chain translations; no XY label.
        height=.56+.2441+.0475+(R@np.array([.06,.05,.045]))[2]
        hsv=cv2.cvtColor(im,cv2.COLOR_BGR2HSV)
        green=((hsv[:,:,0]>30)&(hsv[:,:,0]<95)&(hsv[:,:,1]>60)).astype(np.uint8)
        near=cv2.dilate(green,np.ones((15,15),np.uint8))>0
        mask=((hsv[:,:,1]<75)&(hsv[:,:,2]>140)&near).astype(np.uint8)
        v,u=np.where(mask);uv=np.column_stack([u,v]).astype(float)
        pts,good=ground(uv,R,height);pts=pts[good]
        if len(pts)>600:pts=pts[np.linspace(0,len(pts)-1,600).astype(int)]
        costs=np.full(len(grid),.4)
        if len(pts)>15:
            for begin in range(0,len(grid),300):
                world=grid[begin:begin+300,None]+pts[None]
                mx=np.rint((world[:,:,0]+5.3)*100).astype(int);my=np.rint((3.9-world[:,:,1])*100).astype(int)
                valid=(mx>=0)&(mx<1060)&(my>=0)&(my<780)
                d=dist[np.clip(my,0,779),np.clip(mx,0,1059)]
                d=np.where(valid,d,.5)
                costs[begin:begin+300]=np.minimum(d,.4).mean(1)
        likelihood=np.exp(-(costs-costs.min())/.035).reshape(xx.shape)
        delta=np.zeros(2);accepted=False;nflow=0
        if prev is not None:
            pg,pR,ph=prev
            features=cv2.goodFeaturesToTrack(pg,500,.01,8)
            if features is not None:
                tracked,status,_=cv2.calcOpticalFlowPyrLK(pg,gray,features,None,winSize=(31,31),maxLevel=4)
                back,ok,_=cv2.calcOpticalFlowPyrLK(gray,pg,tracked,None,winSize=(31,31),maxLevel=4)
                valid=(status.ravel()>0)&(ok.ravel()>0)&(np.linalg.norm(back[:,0]-features[:,0],axis=1)<1.5)
                g0,k0=ground(features[:,0],pR,ph);g1,k1=ground(tracked[:,0],R,height)
                dif=(g0-g1)[valid&k0&k1]
                if len(dif)>=8:
                    med=np.median(dif,axis=0);inside=np.linalg.norm(dif-med,axis=1)<.15
                    nflow=int(inside.sum())
                    if nflow>=8 and nflow/len(dif)>.35:
                        delta=np.median(dif[inside],axis=0);accepted=True
        if prob is None:prob=likelihood/likelihood.sum()
        else:
            prior=cv2.warpAffine(prob,np.array([[1,0,delta[0]/.1],[0,1,delta[1]/.1]],dtype=float),(len(xs),len(ys)))
            prior=cv2.GaussianBlur(prior,(0,0),1.5 if accepted else 6)
            prior=.98*prior/max(prior.sum(),1e-12)+.02/prior.size
            prob=prior*likelihood;prob/=prob.sum()
        pred=grid[prob.argmax()];single=grid[costs.argmin()]
        # Ground truth is read only after inference, for evaluation of CAMERA XY.
        baseR=rotation(np.roll(row['actual']['rotation_quat_xyzw'],1))
        gt=np.array(row['actual']['translation_m'][:2])+(baseR@np.array([-.005,0,.2916]))[:2]+(R@np.array([.06,.05,.045]))[:2]
        result=dict(image=row['image'],split=row['split'],index=row['index'],predicted_camera_xy=pred.tolist(),single_camera_xy=single.tolist(),gt_camera_xy=gt.tolist(),error_m=float(np.linalg.norm(pred-gt)),single_error_m=float(np.linalg.norm(single-gt)),flow_delta_m=delta.tolist(),flow_accepted=accepted,flow_inliers=nflow,line_points=len(pts))
        results.append(result);prev=(gray,R,height)
        print(row['image'],round(result['error_m'],3),accepted,flush=True)
    (a.out/'predictions.json').write_text(json.dumps(results,indent=2))
    summary={}
    for split in ('reference','query'):
        rr=[r for r in results if r['split']==split];e=np.array([r['error_m'] for r in rr]);s=np.array([r['single_error_m'] for r in rr])
        summary[split]=dict(frames=len(rr),median_m=float(np.median(e)),mean_m=float(e.mean()),p90_m=float(np.quantile(e,.9)),within_05m=float((e<.5).mean()),single_median_m=float(np.median(s)),flow_accepted=sum(r['flow_accepted'] for r in rr))
    (a.out/'metrics.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
