# bev2loc — fixed-map localization observation infra

A known bird's-eye-view map (e.g. `res/example.png`) is given. At test time the
agent only ever sees a small, heavily degraded patch of it. This package builds
the **data infrastructure** for that problem: procedural trajectories, cropped
observation windows, and realistic geometric + photometric corruptions. No
model is included yet.

```
res/example.png              the fixed map
bev2loc/map.py               map loading, coordinate frames, regions
bev2loc/trajectory.py        smooth waypoint / random-walk pose sequences
bev2loc/observation.py       crop windows out of the map, exact ground truth
bev2loc/augment.py           scale, aspect, shear, perspective, noise, scratches
bev2loc/dataset.py           on-the-fly torch Datasets (single frame / sequence)
bev2loc/visualize.py         preview renders for sanity checks
outputs/                     previews and other generated artefacts (gitignored)
tests/                       local geometry / determinism / dataset tests (gitignored)
scripts/                     one-off helper scripts (gitignored)
```

Only `res/example.png` and the `bev2loc` package are tracked; generated outputs,
tests and scratch scripts stay local.

## Quick start (uv)

```bash
# ROCm machine (this repo was developed on one)
uv sync --extra rocm
uv run --extra rocm python -m bev2loc preview --out-dir outputs/previews
uv run --extra rocm pytest
```

Other extras: `--extra cuda`, `--extra metal`.

Preview outputs:

* `outputs/previews/preview_trajectory.png` — trajectory with sampled view windows
* `outputs/previews/preview_observations.png` — fully corrupted observations
* `outputs/previews/preview_geometry.png` — the same crops without noise
* `outputs/previews/preview_augmentation_stages.png` — clean → geometric → full

## Coordinate and target conventions

* Map pixel coordinates: `x` right, `y` **down**, origin at image top-left.
* `Pose(x, y, yaw)`: camera position in map pixels; `yaw` in radians measured
  from `+x` (positive turns are clockwise on screen because `y` points down).
* An observation is a rotated crop centred on the pose, covering
  `fov_px` map pixels across the observation width.
* **The localization target is the map pixel under the centre of the
  observation** (`Observation.gt_xy`), not necessarily the nominal pose.
  Geometric translation jitter shifts the crop content, so `gt_xy` is always
  recomputed from the composed homography. Datasets expose:

  | key | meaning |
  | --- | --- |
  | `pos` | target `(x, y)` normalised to `[0, 1]` by map size |
  | `pos_px` | target in map pixels |
  | `yaw`, `yaw_sincos` | camera heading |
  | `pose_px` | `(x, y, yaw)` used to render the crop |
  | `corners` | 4 observation corners in map pixels |
  | `meta` | sampled augmentation / photometric parameters |

  `PoseDataset` yields `(C, H, W)` images; `TrajectoryDataset` yields
  `(T, C, H, W)` sequences with targets per frame.

## Usage

```python
import numpy as np
from bev2loc import (
    FixedMap, InfraConfig, GeometryConfig, ObservationGenerator,
    PoseDataset, TrajectoryDataset, generate_trajectory,
)

map_ = FixedMap("res/example.png")
cfg = InfraConfig(
    geometry=GeometryConfig(obs_width=256, obs_height=256, fov_px=320),
)

# one trajectory + observations
traj = generate_trajectory(map_, cfg.trajectory, n_steps=300, rng=np.random.default_rng(0))
gen = ObservationGenerator(map_, cfg.geometry, cfg.photometric)
obs = gen.generate(traj[0], np.random.default_rng(1))
print(obs.image.shape, obs.gt_xy, obs.corners)

# torch datasets
ds = PoseDataset(map_, cfg, length=10_000, seed=0, split="train")
item = ds[0]
print(item["image"].shape, item["pos"], item["yaw_sincos"])

seq_ds = TrajectoryDataset(map_, cfg, length=2_000, seq_len=8, step_stride=8)
seq = seq_ds[0]
print(seq["images"].shape, seq["pos"].shape)
```

## What is randomised

**Geometry** (`GeometryConfig`): zoom, x/y aspect, affine shear, extra roll,
small translation of the content, and mild perspective (oblique view). All are
composed into one homography, so ground truth stays exact.

**Photometry** (`PhotometricConfig`): defocus/motion blur, brightness/gamma/
colour tint, vignetting, illumination gradient, Gaussian + salt-and-pepper
noise, lens scratches (polylines), dust specks, smudges, JPEG artefacts.

**Trajectory** (`TrajectoryConfig`): waypoint navigation by default (smooth
bounded-curvature arcs, broad coverage). Set `mode="random_walk"` for a local
OU-curvature meander. Poses stay inside the map's non-background bounding box
(the pitch), shrunk slightly.

## Notes and design choices

* Everything is procedural and seeded: `PoseDataset[i]` is reproducible, and
  train/val/test use different seed offsets automatically.
* Unknown area outside the map is filled with the map's background colour by
  default (`border="background"`), so the border itself does not leak the
  position. `border="reflect"` and `"zero"` are also available.
* Many map patches are almost featureless green, so some observations are
  genuinely ambiguous even without corruption. That is a property of the task,
  not a bug; a model must learn to use the white pitch markings and reject
  uninformative frames.
* `fov_px` is a useful difficulty knob: smaller means less context and a
  harder localization problem.