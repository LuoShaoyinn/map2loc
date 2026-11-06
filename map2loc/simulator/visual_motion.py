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
    return gray, mask, R, height, *plane_features(rgb, R, height)


def empty_motion(reason):
    return dict(accepted=False, delta_world_m=[0.,0.], candidates=0, inliers=0,
                residual_m=None, reason=reason)


def estimate_feature_motion(previous, current):
    pg,pm,pR,ph = previous[:4]
    cg,cm,cR,ch = current[:4]
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

def plane_features(rgb, R, height):
    """World-aligned line/visibility rasters centered on the unknown camera XY."""
    y,x=np.mgrid[:151,:151]
    local=np.stack([(x-75)*.08,-(y-75)*.08,np.full_like(x,-height,dtype=float)],-1)@R
    f=320/np.tan(np.pi/6)
    u=(319.5-f*local[:,:,1]/np.maximum(local[:,:,0],1e-6)).astype('float32')
    v=(239.5-f*local[:,:,2]/np.maximum(local[:,:,0],1e-6)).astype('float32')
    inside=(local[:,:,0]>.05)&(u>=0)&(u<639)&(v>=0)&(v<479)
    im=cv2.remap(rgb,u,v,cv2.INTER_LINEAR)
    hsv=cv2.cvtColor(np.clip(im,0,255).astype('uint8'),cv2.COLOR_RGB2HSV)
    green=(hsv[:,:,0]>30)&(hsv[:,:,0]<95)&(hsv[:,:,1]>60)
    near=cv2.dilate(green.astype('uint8'),np.ones((5,5),np.uint8))>0
    white=(hsv[:,:,1]<90)&(hsv[:,:,2]>130)&near
    valid=(inside & (green|white)).astype('float32')
    line=cv2.GaussianBlur(white.astype('float32'),(3,3),.5)*valid
    return line,valid


def estimate_plane_motion(previous,current):
    """Masked correlation with both footprints changing; rejects ambiguous peaks."""
    pl,pv=previous[4:];cl,cv=current[4:]
    pad=32
    def corr(a,b):
        return cv2.matchTemplate(cv2.copyMakeBorder(a,pad,pad,pad,pad,cv2.BORDER_CONSTANT),
                                 b,cv2.TM_CCORR)
    n=np.maximum(corr(pv,cv),1)
    sp=corr(pl,cv);sc=corr(pv,cl)
    vp=np.maximum(corr(pl*pl,cv)-sp*sp/n,0)
    vc=np.maximum(corr(pv,cl*cl)-sc*sc/n,0)
    numerator=corr(pl,cl)-sp*sc/n
    ncc=numerator/np.sqrt(vp*vc+1e-6)
    good=(n>max(70,.2*min(float(pv.sum()),float(cv.sum()))))&(vp>3)&(vc>3)
    score=np.where(good,np.clip(ncc,-1,1),-2)
    iy,ix=np.unravel_index(score.argmax(),score.shape)
    yy,xx=np.indices(score.shape)
    separated=(xx-ix)**2+(yy-iy)**2>25
    second=float(score[separated].max())
    best=float(score[iy,ix]);margin=best-second
    # Fractional peak refinement from the local parabola.
    def refine(v0,v1,v2):
        denominator=v0-2*v1+v2
        return float(np.clip(.5*(v0-v2)/denominator,-.5,.5)) if denominator < -1e-6 else 0.
    dx=refine(score[iy,ix-1],best,score[iy,ix+1]) if 0<ix<2*pad else 0.
    dy=refine(score[iy-1,ix],best,score[iy+1,ix]) if 0<iy<2*pad else 0.
    delta=np.array([ix+dx-pad,-(iy+dy-pad)])*.08
    accepted=best>.55 and margin>.08 and 0<ix<2*pad and 0<iy<2*pad
    return dict(accepted=bool(accepted),delta_world_m=delta.tolist(),method='plane_correlation',
                candidates=int(good.sum()),inliers=int(min(sp[iy,ix],sc[iy,ix])),
                residual_m=None,correlation=best,peak_margin=margin,
                overlap_cells=int(n[iy,ix]),reason='accepted' if accepted else 'ambiguous_plane_alignment')


def estimate_motion(previous,current):
    return estimate_plane_motion(previous,current)
