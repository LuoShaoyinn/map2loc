"""Image-flow exposure approximation for privileged stop-and-observe captures.

No previous-frame colors are accumulated. The previous view and measured head
attitudes estimate displacement only. Current-frame radiance is integrated over
that spatially varying motion field. Exposure is a fraction of the sampled frame
interval, not calibrated milliseconds or simulator velocity.
"""
import cv2
import numpy as np
from .localize import rotation


def blur_from_motion(image, previous, quat, previous_quat, fraction):
    if previous is None or fraction == 0:
        return image.copy(), dict(flow_p95_px=0., blur_p95_px=0., samples=1)
    h,w=image.shape[:2]
    f=w/2/np.tan(np.pi/6)
    K=np.array([[f,0,(w-1)/2],[0,f,(h-1)/2],[0,0,1.]])
    # Head axes -> camera right/down/forward.
    A=np.array([[0.,-1,0],[0,0,-1],[1,0,0]])
    H=K@A@rotation(quat).T@rotation(previous_quat)@A.T@np.linalg.inv(K)
    aligned=cv2.warpPerspective(previous,H,(w,h),borderMode=cv2.BORDER_REFLECT_101)
    gray=lambda im:cv2.cvtColor(np.clip(im,0,255).astype('uint8'),cv2.COLOR_RGB2GRAY)
    dis=cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    residual=dis.calc(gray(image),gray(aligned),None)
    y,x=np.mgrid[:h,:w].astype('float32')
    # Transform residual correspondences back to original previous-image pixels.
    coord=np.stack([x+residual[:,:,0],y+residual[:,:,1],np.ones_like(x)],-1)@np.linalg.inv(H).T
    valid=coord[:,:,2]>.1
    previous_uv=coord[:,:,:2]/np.maximum(coord[:,:,2:],.1)
    flow=(previous_uv-np.stack([x,y],-1)).astype('float32')
    flow[~valid]=0
    magnitude=np.linalg.norm(flow,axis=2)
    p95=float(np.quantile(magnitude,.95))
    samples=int(np.clip(np.ceil(p95*fraction)*2+1,17,97))
    color=image/255.
    linear=np.where(color<=.04045,color/12.92,((color+.055)/1.055)**2.4)
    exposure=np.zeros_like(linear)
    # Centered shutter keeps the supplied current quaternion at mid-exposure.
    for t in np.linspace(-fraction/2,fraction/2,samples):
        t=np.float32(t)
        exposure+=cv2.remap(linear,x+t*flow[:,:,0],y+t*flow[:,:,1],cv2.INTER_LINEAR,borderMode=cv2.BORDER_REFLECT_101)
    exposure/=samples
    srgb=np.where(exposure<=.0031308,12.92*exposure,1.055*np.maximum(exposure,0)**(1/2.4)-.055)
    return np.clip(srgb*255,0,255).astype('float32'),dict(flow_p95_px=p95,blur_p95_px=p95*fraction,samples=samples)
