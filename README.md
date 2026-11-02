# data_gen — fixed-map localization observation infra

A known bird's-eye-view map (e.g. `res/example.png`) is given. At test time the
agent only ever sees a small, noisy view from a downward-looking camera. This
package builds the **data infrastructure** for that problem: a real pinhole
camera with a fixed height, smooth camera trajectories, noisy attitude inputs,
and a validity mask that marks pixels which cannot be trusted. No model is
included yet.

```
res/example.png              the fixed map
data_gen/map.py               map loading, coordinate frames, regions
data_gen/camera.py            pinhole projection: pose -> view homography
data_gen/trajectory.py        smooth x/y/yaw/pitch/roll camera curves
data_gen/observation.py       render + clip a camera view, exact ground truth
data_gen/augment.py           zoom/head jitter, Gaussian noise, invalid pixels
data_gen/dataset.py           on-the-fly torch Datasets (single frame / sequence)
data_gen/tensor_generator.py  batched on-device camera renderer for training
data_gen/visualize.py         preview renders, including the validity-mask graph
data_gen/video.py             example video: map pose, clean/noisy view, mask
outputs/                     previews and other generated artefacts (gitignored)
tests/                       local geometry / determinism / dataset tests (gitignored)
scripts/                     one-off helper scripts (gitignored)
```

Only `res/example.png` and the `data_gen` package are tracked; generated outputs,
tests and scratch scripts stay local.

## Quick start (uv)

```bash
# ROCm machine (this repo was developed on one)
uv sync --extra rocm
uv run --extra rocm python -m data_gen preview --out-dir outputs/previews
uv run --extra rocm python -m data_gen video --out outputs/videos/data_gen_example.mp4
uv run --extra rocm pytest
```

Other extras: `--extra cuda`, `--extra metal`.

The installed console scripts are `data-gen-preview` and `data-gen-video`.

Preview outputs:

* `preview_trajectory.png` — trajectory with sampled camera footprints
* `preview_observations.png` — fully corrupted views
* `preview_geometry.png` — the same views without noise
* `preview_augmentation_stages.png` — nominal → zoom/head jitter → noise
* `preview_validity.png` — image with invalid pixels tinted, plus the mask
* `preview_validity_stages.png` — image, map coverage, combined mask, overlay

## Example video

`data_gen.video` renders a seeded sequence of the generated data:

```bash
python -m data_gen video --out outputs/videos/data_gen_example.mp4
```

Default settings produce a 12 s, 25 fps clip (300 frames) at
`1318x666`. Each frame has four synchronised panels:

* **map + trajectory** — the fixed map, the full route (grey), the trail so far
  (orange), the current view footprint (yellow) and the current camera ground
  position / heading (red). The blue cross is the map point shown at the image
  centre, which differs from the camera position when the camera is pitched.
* **clean camera view** — the exact perspective view at the effective pose,
  before sensor noise.
* **noisy observation** — the same view after zoom / head jitter and Gaussian
  noise; this is the input a model would receive.
* **validity** — the trustworthy-pixel mask, and the noisy view with all
  invalid pixels (map edge / horizon or random) tinted red.

The header shows the current ground-truth position, the noisy yaw / pitch
inputs, the sampled zoom and noise sigma, and the valid fraction. A poster
frame is saved next to the video as `data_gen_example_poster.png`.

Useful options: `--n-frames`, `--fps`, `--seed`, `--cell`, `--obs-size`,
`--height`, `--footprint`, and `--no-poster`.

## Camera model

The map is treated as a textured **ground plane** (`z = 0`). A pinhole camera
with a **fixed height** `height_px` looks down at it. Its pose is

```
CameraPose(x, y, yaw, pitch, roll)
```

* `x, y` — camera position on the ground plane, in map pixels.
* `yaw` — heading in radians from `+x`. Positive yaw turns clockwise on screen
  because `y` points down.
* `pitch` — tilt measured from the **horizontal** plane. `0` looks straight
  ahead along the heading, `-90°` looks straight down (nadir), and negative
  values tilt towards the ground. The trajectory keeps it in `[-80°, 0°]`
  (mean `-45°` by default), so the camera looks obliquely at the ground and
  its footprint on the map is a perspective quadrilateral, not a rectangle.
* `roll` — small rotation about the optical axis (a few degrees, smooth).
* `height_px` — fixed at `GeometryConfig.height_px`; `Pose.height` can override
  it per frame, but the generator never samples it.

At `yaw = roll = 0` and `pitch = -90°`, the optical axis points straight down,
the image top points towards `+x`, and image right points towards `+y`. The
focal length is derived from the camera height and `footprint_px`
(ground-plane width covered at nadir):

```
focal_px = height_px * obs_width / footprint_px
```

The default `footprint_px = 120` with `height_px = 180` gives a horizontal
field of view of about **37°** — much narrower than a wide-angle BEV crop.

A random zoom is applied per frame. With a fixed camera height, a zoom is
exactly equivalent to moving the camera up/down, so this is the only scale
augmentation needed. Small per-frame **head / gimbal** rotations
(`head_pitch`, `head_yaw`, `head_roll`) are sampled around zero and added to
the smooth camera pose before rendering.

The view is rendered by inverse-warping the ground plane through the camera
homography and clipping to `(obs_height, obs_width)`. Because the scene is
planar the projection is exact.

## Trajectories

`generate_trajectory` returns `(N, 5)` poses `(x, y, yaw, pitch, roll)`:

* `x, y, yaw` use waypoint navigation by default (smooth bounded-curvature
  arcs, broad coverage), or an OU-curvature random walk.
* `pitch` follows a smooth Ornstein-Uhlenbeck curve in `[-80°, 0°]` (mean
  `-45°` by default), so the camera keeps a real forward/down perspective;
  `roll` is a small OU curve around zero. Both have a per-step slew limit.

All poses stay inside the map's non-background bounding box (the pitch),
shrunk slightly.

## Noisy attitude inputs

The localization target is the camera's ground position (`Observation.gt_xy`).
The image centre usually shows a slightly different map point when pitched;
that point is stored separately as `Observation.center_xy`.

The model is **given** camera yaw and pitch as inputs:

```
yaw_input   = wrap(yaw_gt   + N(0, MeasurementConfig.yaw_sigma_deg))
pitch_input =      pitch_gt + N(0, MeasurementConfig.pitch_sigma_deg)
```

Position `(x, y)` is not given. `Observation` and the datasets expose both the
exact ground truth and the noisy inputs.

## Noise and validity masks

`PhotometricConfig` is deliberately simple:

* one Gaussian sigma per view, applied independently to every pixel/channel;
* a random fraction of pixels is flagged invalid. Their rendered value is kept;
  the mask says they cannot be trusted.

`Observation.valid_mask` combines:

1. **map coverage** — 1 only where the view pixel samples the known map, 0 for
   border padding outside the image;
2. **random invalid pixels** — Bernoulli mask at the sampled fraction.

Datasets expose the mask as `valid_mask` (single frame) / `valid_masks`
(sequence), and the previews include a dedicated mask graph.

## Coordinate and target conventions

* Map pixel coordinates: `x` right, `y` **down**, origin at image top-left.
* Ground plane: `z = 0`, world `z` points up; camera height is positive.
* An observation is a perspective camera view clipped to a fixed image size.
* **The localization target is the camera's ground position**
  (`Observation.gt_xy`), not necessarily the image centre.
* Datasets expose:

  | key | meaning |
  | --- | --- |
  | `pos` | target `(x, y)` normalised to `[0, 1]` by map size |
  | `pos_px` | target in map pixels |
  | `center_px` | map point seen at the image centre |
  | `yaw`, `yaw_sincos`, `pitch`, `roll` | camera ground-truth attitude |
  | `yaw_input`, `pitch_input` | noisy attitude inputs given to the model |
  | `yaw_input_sincos` | `(sin, cos)` of the noisy yaw input |
  | `pose_px` | `(x, y, yaw, pitch, roll)` used to render the view |
  | `corners` | 4 view corners in map pixels |
  | `valid_mask(s)` | trustworthy-pixel mask |
  | `meta` | sampled augmentation / noise parameters / camera intrinsics |

  `PoseDataset` yields `(C, H, W)` images; `TrajectoryDataset` yields
  `(T, C, H, W)` sequences with targets per frame.

## Usage

```python
import numpy as np
from data_gen import (
    FixedMap, InfraConfig, GeometryConfig, ObservationGenerator,
    PoseDataset, TrajectoryDataset, generate_trajectory,
)

map_ = FixedMap("res/example.png")
cfg = InfraConfig(
    geometry=GeometryConfig(obs_width=256, obs_height=256,
                            height_px=180, footprint_px=120),
)

# one smooth trajectory + observations
traj = generate_trajectory(map_, cfg.trajectory, n_steps=300,
                           rng=np.random.default_rng(0))
gen = ObservationGenerator(
    map_, cfg.geometry, cfg.photometric, cfg.measurement
)
obs = gen.generate(traj[0], np.random.default_rng(1))
print(obs.image.shape, obs.gt_xy, obs.yaw_input, obs.pitch_input)
print(obs.valid_mask.shape, obs.valid_mask.mean())

# torch datasets
ds = PoseDataset(map_, cfg, length=10_000, seed=0, split="train")
item = ds[0]
print(item["image"].shape, item["pos"], item["yaw_input"], item["pitch_input"])

seq_ds = TrajectoryDataset(map_, cfg, length=2_000, seq_len=8, step_stride=8)
seq = seq_ds[0]
print(seq["images"].shape, seq["pos"].shape, seq["valid_masks"].shape)
```

Online training can use `TensorObservationGenerator`, which implements the same
camera model and noise on the device and returns `image`, `valid_mask`,
`gt_xy`, `yaw_input` and `pitch_input`.

## What is randomised

**Camera geometry** (`GeometryConfig`): random zoom (equivalent to a height
change), and small head pitch/yaw/roll jitter around zero. The camera height
itself is fixed.

**Noise** (`PhotometricConfig`): uniform-sigma Gaussian noise plus randomly
invalid pixels.

**Attitude measurements** (`MeasurementConfig`): Gaussian noise added to the
ground-truth yaw and pitch that are passed to the model.

**Trajectory** (`TrajectoryConfig`): smooth waypoint navigation (or an OU
random walk) for `x, y, yaw`, a smooth oblique OU curve for `pitch`, and a
small OU curve for `roll`.

## Notes and design choices

* Everything is procedural and seeded: `PoseDataset[i]` is reproducible, and
  train/val/test use different seed offsets automatically.
* Unknown area outside the map is filled with the map's background colour by
  default (`border="background"`), so the border itself does not leak the
  position. The validity mask still marks it invalid.
* Because the camera is oblique and the field of view is finite, the horizon
  can enter the image for near-horizontal pitches. Rays at or above the
  horizon do not hit the ground plane; those pixels are filled with the
  background and marked invalid instead of being projected as mirrored map
  content.
* Many map patches are almost featureless green, so some observations are
  genuinely ambiguous even without corruption. That is a property of the task,
  not a bug; a model must learn to use the white pitch markings and reject
  uninformative frames.
* `footprint_px` is the useful difficulty knob: smaller means less context and
  a harder localization problem. Changing `height_px` while keeping
  `footprint_px` fixed changes the perspective strength.