"""Quaternion-initialized optical tracking, then robust ground-plane translation.

Only two RGB views, their head attitudes and fixed calibration enter estimation.
World XY is never read. Displacement includes the camera's motion from head turns.
"""
import cv2
import numpy as np
from .localize import rotation, ground


def frame_geometry(rgb, quat):
    R = rotation(quat)
    height = .56+.2916+(R@np.array([.06, .05, .045]))[2]
    image = np.clip(rgb, 0, 255).astype('uint8')
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    green = ((hsv[:,:,0]>30)&(hsv[:,:,0]<95)&(hsv[:,:,1]>60)).astype('uint8')
    near = cv2.dilate(green, np.ones((13,13),np.uint8))>0
    white = (hsv[:,:,1]<90)&(hsv[:,:,2]>130)
    # Track field markings, avoiding robots, balls and distant walls.
    mask = (near & (white | (green>0))).astype('uint8')*255
    return gray, mask, R, height


def empty_motion(reason):
    return dict(accepted=False, delta_world_m=[0.,0.], candidates=0, inliers=0,
                residual_m=None, reason=reason)


def estimate_motion(previous, current):
    pg,pm,pR,ph = previous
    cg,cm,cR,ch = current
    f=320/np.tan(np.pi/6)
    K=np.array([[f,0,319.5],[0,f,239.5],[0,0,1.]])
    A=np.array([[0.,-1,0],[0,0,-1],[1,0,0]])
    H=K@A@cR.T@pR@A.T@np.linalg.inv(K)
    points=cv2.goodFeaturesToTrack(pg,800,.005,7,mask=pm)
    if points is None:return empty_motion('no_features')
    prediction=cv2.perspectiveTransform(points,H)
    finite=np.isfinite(prediction).all((1,2))
    finite &= (prediction[:,0,0]>5)&(prediction[:,0,0]<634)&(prediction[:,0,1]>5)&(prediction[:,0,1]<474)
    points=points[finite];prediction=prediction[finite]
    if len(points)<8:return empty_motion('insufficient_rotational_overlap')
    tracked,status,_=cv2.calcOpticalFlowPyrLK(pg,cg,points,prediction.copy(),
        winSize=(31,31),maxLevel=4,flags=cv2.OPTFLOW_USE_INITIAL_FLOW,
        criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,40,.01))
    back,back_status,_=cv2.calcOpticalFlowPyrLK(cg,pg,tracked,points.copy(),
        winSize=(31,31),maxLevel=4,flags=cv2.OPTFLOW_USE_INITIAL_FLOW)
    good=(status[:,0]>0)&(back_status[:,0]>0)&(np.linalg.norm(back[:,0]-points[:,0],axis=1)<1.5)
    u=np.rint(tracked[:,0,0]).astype(int);v=np.rint(tracked[:,0,1]).astype(int)
    inside=(u>=0)&(u<640)&(v>=0)&(v<480)
    good &= inside & (cm[np.clip(v,0,479),np.clip(u,0,639)]>0)
    g0,k0=ground(points[:,0],pR,ph);g1,k1=ground(tracked[:,0],cR,ch)
    good &= k0 & k1
    # A stationary world point obeys camera_delta = old_relative - new_relative.
    differences=(g0-g1)[good]
    if len(differences)<8:return empty_motion('insufficient_plane_tracks')
    # Deterministic consensus for translation; no pose-informed RANSAC seed.
    d=np.linalg.norm(differences[:,None]-differences[None],axis=2)
    support=(d<.15).sum(1);center=differences[support.argmax()]
    inliers=np.linalg.norm(differences-center,axis=1)<.15
    delta=np.median(differences[inliers],axis=0)
    residual=float(np.median(np.linalg.norm(differences[inliers]-delta,axis=1)))
    fraction=float(inliers.mean());n=int(inliers.sum())
    accepted=n>=8 and fraction>=.4 and residual<.09 and np.linalg.norm(delta)<2.5
    return dict(accepted=bool(accepted),delta_world_m=delta.tolist(),candidates=len(differences),
                inliers=n,inlier_fraction=fraction,residual_m=residual,
                reason='accepted' if accepted else 'weak_consensus')
