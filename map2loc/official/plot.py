"""Plot completed official-model experiment predictions, without running tests."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    p=argparse.ArgumentParser();p.add_argument('evaluation',type=Path)
    p.add_argument('--alpha',type=float,default=.05);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();rows=json.loads((a.evaluation/'predictions.json').read_text())
    rows=[r for r in rows if r['alpha']==a.alpha]
    background=plt.imread(Path(__file__).resolve().parents[2]/'res/example.png')
    fig,axes=plt.subplots(1,3,figsize=(15,4.5))
    for ax,key,title in zip(axes[:2],['predicted_xy_m','known_yaw_xy_m'],['Official OrienterNet: full pose','Official OrienterNet: measured yaw prior']):
        ax.imshow(background,extent=(-5.3,5.3,-3.9,3.9))
        for split in dict.fromkeys(r['split'] for r in rows):
            subset=[r for r in rows if r['split']==split]
            gt=np.array([r['gt_xy_m'] for r in subset]);pred=np.array([r[key] for r in subset])
            label=split==rows[0]['split']
            ax.plot(*gt.T,'-',color='yellow',lw=1.3,label='Actual' if label else None)
            ax.plot(*pred.T,'o-',color='cyan',lw=.6,ms=2,label='Estimated' if label else None)
        ax.set(title=title,xlabel='World X (m)',ylabel='World Y (m)');ax.legend(fontsize=8)
    for key,label in [('error_m','Full pose'),('known_yaw_error_m','Measured yaw prior')]:
        axes[2].plot([r[key] for r in rows],label=label,lw=1)
    axes[2].set(xlabel='Frame (reference then query pass)',ylabel='Camera XY error (m)',title=f'Held-out route; exposure strength {a.alpha}')
    axes[2].legend();axes[2].grid(alpha=.3)
    fig.tight_layout();a.out.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(a.out,dpi=160)

if __name__=='__main__':main()
