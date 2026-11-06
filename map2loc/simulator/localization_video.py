"""Render current localization and image-flow arrows without trajectory tails."""
import argparse,json
from pathlib import Path
import cv2,numpy as np

def main():
 p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--result',type=Path,required=True);a=p.parse_args()
 rows=json.loads((a.result/'predictions.json').read_text());field=cv2.resize(cv2.imread('res/example.png'),(848,624))
 writer=cv2.VideoWriter(str(a.result/'localization.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),6,(1500,760))
 prev=None
 for i,r in enumerate(rows):
  c=np.full((760,1500,3),245,np.uint8);c[60:684,:848]=field
  def txt(s,x,y,size=.6):cv2.putText(c,s,(x,y),cv2.FONT_HERSHEY_SIMPLEX,size,(25,25,25),2,cv2.LINE_AA)
  def pt(xy):return (round((xy[0]+5.3)*80),round((3.9-xy[1])*80)+60)
  cv2.drawMarker(c,pt(r['gt_camera_xy']),(0,220,255),cv2.MARKER_CROSS,24,3)
  cv2.circle(c,pt(r['predicted_camera_xy']),9,(255,200,0),3)
  im=cv2.imread(str(a.data/r['image']));gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
  if prev is not None and r['index']:
   points=cv2.goodFeaturesToTrack(prev,100,.02,15)
   if points is not None:
    to,status,_=cv2.calcOpticalFlowPyrLK(prev,gray,points,None,winSize=(31,31),maxLevel=4)
    for b,e,s in zip(points[:,0],to[:,0],status[:,0]):
     if s:cv2.arrowedLine(im,tuple(b.astype(int)),tuple(e.astype(int)),(255,220,0),1,cv2.LINE_AA,tipLength=.2)
  prev=gray;c[90:570,860:1500]=im
  txt('Map localization | head quaternion + RGB | no simulator velocity',20,35,.8)
  txt(f'{r["split"]} frame {r["index"]+1} | camera XY error {r["error_m"]:.2f} m',875,610)
  txt(f'Flow accepted: {r["flow_accepted"]} | inliers: {r["flow_inliers"]}',875,650)
  txt('Yellow cross: ground truth    Cyan circle: estimate',20,720)
  txt('6 fps settled-frame replay; arrows show image flow',875,720,.52)
  writer.write(c)
  if i==20:cv2.imwrite(str(a.result/'poster.png'),c)
 writer.release()
if __name__=='__main__':main()
