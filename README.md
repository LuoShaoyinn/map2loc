# map2loc — attitude-aware BEV localization

## Actual URSoccerLab experiment

`python -m map2loc.simulator.experiment --out outputs/ursoccerlab/run01`
connects to a running simulator, uses admin `lock_pose` to move the MOS9 robot
without a gait, scans its head, and records decoded left-eye views. Install the
URSoccerLab `ursoccerlab_client-0.1.1` wheel into the experiment environment first.
The defaults target robot `robot_bp0`, robot TCP port 10000, admin port 11000.
Only run when that robot is available for experiment control.

The route is an ellipse with radii 2.8 m and 1.7 m, at root height 0.56 m,
looking inward with sinusoidal head yaw/pitch. This is discrete stop-and-observe
capture, not continuous locomotion. Each pose is held until a camera timestamp
is at least 0.4 simulation seconds newer than the post-command pose timestamp.
The original pose is restored and the lock released on normal completion or
Python exceptions. A forcibly terminated process may require manual unlock.

Outputs contain PNG images, `frames.jsonl` (commands, measured admin poses and
camera timestamps), `predictions.npz`, and `metrics.json`. The baseline creates
a visual database from the reference pass and retrieves a location from RGB
alone for the query pass. Queries are halfway between reference route samples
and have a 0.06 rad additional head yaw. Query ground truth enters only metrics;
this does not yet connect real images to the planar-map BEV matcher.

Actual A843-RX9070xt run, 2026-11-06, code `5fe17b1`, 24 frames per pass:

| Metric | Result |
| --- | ---: |
| Mean base XY error | 2.458 m |
| Median base XY error | 2.341 m |
| 90th percentile error | 4.845 m |
| Error below 0.5 m | 8/24 |
| Mean nearest-reference spatial distance | 0.293 m |

Artifacts are on that host at
`~/workspace/map2loc-urs-experiment/outputs/ursoccerlab/route24_v1`.
This deterministic same-scene route is an initial integration experiment, not
evidence of unseen-scene generalization. Received images include arm occlusion;
global thumbnail retrieval frequently confuses locations on the symmetric field.
The next localization step is calibrated camera-to-ground rectification and
field-line matching with attitude inputs, followed by temporal fusion and
separate routes for evaluation. No local tests or synthetic smoke were run for
this integration; validation used the running packaged simulator.

A known bird's-eye-view map (e.g. `res/example.png`) is given. At test time the
agent only ever sees a small, heavily corrupted view from an **oblique
perspective camera** at a fixed height. Camera yaw, pitch and roll are measured
(±5° sigma) and given to the algorithm; the ground position `(x, y)` is not.

This branch (`bev-v2`) combines two packages:

* **`data_gen`** — the observation infrastructure: a real pinhole camera with a
  fixed height, smooth trajectories, noisy attitude inputs, heavy sensor
  corruption and a validity mask.
* **`map2loc`** — the localization algorithms: inverse-perspective
  rectification to BEV, a masked-NCC classical baseline, a learned dense
  correlator, and a position-only temporal histogram filter.

```
map2loc/bev.py                  inverse-perspective rectification (BEV)
map2loc/tensor_generator.py     on-device renderer + BEV rectification
map2loc/matching/classical.py   masked NCC over BEV, residual-yaw/scale search
map2loc/matching/learned.py     learned BEV/map encoders + dense correlation
map2loc/temporal/histogram_filter.py  position-only Bayes filter
map2loc/evaluation/             benchmark scenarios and metrics
map2loc/train.py                observation-model training
map2loc/evaluate.py             benchmark CLI
```

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

Both `data_gen` and `map2loc` are tracked. Generated outputs, tests and scratch
scripts stay local.

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

A random zoom in `[0.70, 1.40]` is applied per frame. With a fixed camera
height, a zoom is exactly equivalent to moving the camera up/down, so this is
the only scale augmentation needed. Larger per-frame **head / gimbal**
rotations (`head_pitch=±3°`, `head_yaw=±5°`, `head_roll=±1.5°`) are sampled
around zero and added to the smooth camera pose before rendering.

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

The model is **given** camera yaw, pitch and roll as inputs:

```
yaw_input   = wrap(yaw_gt   + N(0, MeasurementConfig.yaw_sigma_deg))
pitch_input =      pitch_gt + N(0, MeasurementConfig.pitch_sigma_deg)
roll_input  =      roll_gt  + N(0, MeasurementConfig.roll_sigma_deg)
```

The defaults are **5° on each axis**. Position `(x, y)` is not given.
`Observation` and the datasets expose both the exact ground truth and the noisy
inputs.

## Noise and validity masks

`PhotometricConfig` combines several independent corruption effects:

* **per-view Gaussian noise** — one sigma per view, applied independently to
  every pixel/channel;
* **per-pixel Gaussian noise** — an extra independent sigma drawn for each
  pixel, so the noise is heteroscedastic;
* **random brightness scaling** — a per-view multiplicative scale and additive
  shift;
* **random scratches** — thin lines with random orientation, width and
  brightness delta; they are marked invalid by default;
* **random invalid pixels** — a configurable Bernoulli rate
  (`invalid_pixel_rate`, default **0.5**). Their rendered value is kept; the
  mask says they cannot be trusted.

`Observation.valid_mask` combines:

1. **map coverage** — 1 only where the view pixel samples the known map, 0 for
   border padding outside the image or rays at/above the horizon;
2. **random scratches** — 0 on scratch pixels when `scratch_invalid=True`;
3. **random invalid pixels** — Bernoulli mask at `invalid_pixel_rate`.

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

**Camera geometry** (`GeometryConfig`): random zoom in `[0.70, 1.40]`
(equivalent to a height change), and head pitch/yaw/roll jitter of
`±3° / ±5° / ±1.5°` around the smooth pose. The camera height itself is fixed.

**Noise** (`PhotometricConfig`): per-view Gaussian noise, independent
per-pixel Gaussian noise, random brightness scaling, random scratches, and a
configurable invalid-pixel rate (default `0.5`).

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

## Localization algorithms (`map2loc`)

### Rectification to BEV

Because the scene is a plane, the map-to-view projection is a homography. For a
known attitude and focal scale the inverse homography samples the observation
into a map-aligned **BEV patch** (`map2loc/bev.py`). The patch is centred on the
ground point seen at the image centre; that point is a known offset from the
camera along the measured heading:

```
centre = camera - height / tan(pitch) * [cos(yaw), sin(yaw)]
```

Matching therefore only has to estimate a translation. After matching, the
camera position is recovered by subtracting the same offset.

The focal scale (the generator's random zoom) is not measured, so the matcher
tries seven log-spaced hypotheses in `[0.71, 1.39]`. The measured attitude has
5° noise, so a small residual yaw (three grid bins, ±one grid step) is searched
as well. Pitch and roll are not searched over a range: they are measurements,
not free parameters.

### Classical baseline

`ClassicalNCCMatcher` runs **masked normalised cross-correlation** over the
whole map for every (residual yaw, scale) hypothesis. Only valid template
pixels contribute, so border padding, scratches, the 50–80 % random invalid
pixels and the invalid part of the oblique-to-BEV warp do not create false
peaks. The result is a dense log-likelihood over map position.

### Learned matcher

`LearnedMatcher` embeds the BEV patch with a small CNN and the global map at
the matching-grid resolution, then correlates the patch feature template
against the map feature map with a single convolution. A **reliability head**
is supervised with the rectified validity mask, so the encoder learns which
pixels to trust. Training renders oblique views with the same 5° attitude
noise, 50 % invalid pixels, scratches and photometric corruption as evaluation.

### Temporal fusion

Attitude is measured, so the temporal state is only `(x, y)`.
`XYHistogramFilter` translates the discrete posterior with the available
ego-motion (expressed in the measured heading frame), blurs it, and multiplies
in the yaw-marginalised image likelihood in log space. A small global-mixing
term (`global_mix`, default 5 %) resets part of the posterior to each frame's
global likelihood, which lets the filter escape a wrong lock after distractor
frames without discarding the temporal evidence. This removes the
180°-symmetry failure mode of searching full headings.

### Train and evaluate

```bash
# observation model (GPU recommended)
python -m map2loc.train --device cuda --steps 4000 --batch-size 32 \
    --out outputs/checkpoints/obs_bev.pt

# benchmark: classical + temporal filter, plus learned variants if a
# checkpoint is supplied
python -m map2loc.evaluate --device cuda --seq-len 40 \
    --learned-checkpoint outputs/checkpoints/obs_bev.pt \
    --out outputs/eval/results_bev.json
```

The benchmark scenarios are clean / mild / moderate / severe corruption,
scratches, pixel dropout, a rotated map, an unseen spatial region, a wrong
global initialisation, and recovery after distractor frames.

### Validation video

`map2loc.replay` replays one benchmark scenario with the best method
(`learned+filter`) and writes an MP4 plus a poster frame. Each frame shows the
map with the ground-truth route, the filtered estimate trail and the per-frame
estimate dots; the raw perspective observation; the rectified BEV patch that
the matcher sees; the map template at the estimate; the full-map filter
posterior and a zoom around the ground truth; and the position-error history.

```bash
python -m map2loc.replay --device cuda \
    --scenario seen_moderate \
    --learned-checkpoint outputs/checkpoints/obs_bev_long.pt \
    --out outputs/videos/replay_seen_moderate.mp4
```

`--method classical+filter` runs the same renderer without a checkpoint, which
is handy for a quick smoke test.

To render an unbiased random example set (corruptions, maps, seen/unseen
regions, initialisation and distractor frames are sampled independently; no
example is selected by outcome):

```bash
python -m map2loc.replay --device cuda --random 12 --seed 0 \
    --learned-checkpoint outputs/checkpoints/obs_bev_long.pt \
    --out-dir outputs/videos/random
```

The MP4s and poster frames are generated locally and are intentionally not
committed to git.

### Quaternion + field-line localization on received images

Run `python -m map2loc.simulator.localize --data outputs/ursoccerlab/yaw_head48_v1 --out outputs/ursoccerlab/line_flow_v2` on the simulator host.
This geometric baseline projects RGB-extracted white markings onto the ground,
then searches the supplied map on a 10 cm XY grid. Inputs are RGB, head quaternion
(wxyz), known fixed root height 0.56 m, and MOS9 camera geometry. It does not read
query XY, commanded trajectory or simulator velocity for inference. Ground truth
is read afterwards for camera-XY evaluation (not base-XY evaluation).
The packaged Unreal camera uses 60 degrees horizontal FOV: `FieldOfView` receives
the XML `fovy` value directly. Calibration uses focal length 554.26 pixels at 640x480.

Pyramidal Lucas-Kanade feature tracks are checked forward/backward, intersected
with the ground using successive head attitudes, and robustly reduced to camera
translation. Rejected flow broadens the temporal prior. No velocity integration
or assumed movement between teleported poses is used. Height/planarity remain
assumptions, and non-ground features can still contaminate motion estimates.

Remote actual-image replay `line_flow_v2`: query median 0.0538 m, mean 0.6107 m,
p90 1.3223 m, 27/48 within 0.5 m; flow accepted on 23/47 transitions.
Single-frame median is 0.0516 m, so temporal fusion has not improved that metric.
Reference-pass median is 0.0474 m. Large false matches remain; this is a same-scene
integration result, not a generalization benchmark. The earlier `line_flow_v1`
used an incorrect vertical-FOV interpretation and is retained for comparison.

`python -m map2loc.simulator.localization_video --data outputs/ursoccerlab/yaw_head48_v1 --result outputs/ursoccerlab/line_flow_v2`
renders current GT/estimated positions and raw image-flow arrows without trajectory
tails. Raw displayed tracks include outliers; the estimator uses the gated subset.
Replay is 6 fps over settled captures, not continuous real-time walking.

### Random corner routes, input persistence and learned matching

`map2loc.simulator.experiment --route corners --seed 101 --points 32 --out <new-directory>`
records a seeded route alternating randomized interior waypoints and all four
corner targets at `(±4.05, ±2.55)` m. It independently varies body yaw, head yaw
and head pitch (approximately 11–69 degrees downward). `run.json` records the seed,
waypoints and phases. These remain privileged stop-and-observe trajectories.

`map2loc.simulator.learned_experiment` trains the existing `LearnedMatcher`, an
OrienterNet-style BEV observation/map encoder and dense correlator. It is not the
official OrienterNet model. Known head quaternion, calibrated horizontal FOV and
fixed root height rectify camera images into world-aligned ground patches. Its
position estimates use no simulator XY or velocity; position labels supervise
training and score evaluation only. This experiment evaluates per-frame learned
matching, without the classical optical-flow temporal filter.

Temporal persistence is applied **before BEV rectification**, in camera pixels:
`G[t] = (1-alpha) * I[t] + alpha * G[t-1]`. Memory resets at each route pass.
This creates the requested dragging/ghosting in the model's input; it is a
controlled corruption of sampled frames, not a physically calibrated shutter
simulation. Training includes clean and alpha=0.6 inputs; evaluation compares
alpha=0, 0.35 and 0.65 on exactly the same held-out captures.

```bash
python -m map2loc.simulator.learned_experiment \
  --train outputs/ursoccerlab/yaw_head48_v1 outputs/ursoccerlab/corners101 \
  --eval outputs/ursoccerlab/corners202 \
  --steps 3000 --corruption ghost --out outputs/ursoccerlab/learned_corners_3000
```

The old ellipse and seed 101 provide 160 training frames (320 after adding the
second corruption variant). Seed 202 supplies 64 held-out frames. The evaluation
route is distinct, but the field and many viewed regions overlap. Output includes
checkpoint, loss log, input-corrupted replay videos, per-frame predictions,
metrics and a manifest. All captures and GPU training run on A843-RX9070xt.

Actual 3,000-step GPU run (`learned_corners_3000`, seed 7):

| Input persistence alpha | Median camera XY error | Mean error | Within 0.5 m |
| --- | ---: | ---: | ---: |
| 0 (clean) | 1.634 m | 1.878 m | 29/64 |
| 0.35 | 1.135 m | 1.702 m | 27/64 |
| 0.65 | 1.382 m | 2.193 m | 22/64 |

Clean-input corner subset (within 0.8 m of a target, 12 frames): median 4.237 m,
0/12 within 0.5 m. Clean head-down subset (pitch >50 degrees, 24 frames): median
3.018 m, 4/24 within 0.5 m. Strong ghosting gives medians 4.152 m and 3.399 m
on those subsets. These subsets may overlap. Training sampled-batch NLL fell to
0.0091 at the final step, while held-out errors remained large: fitting this small
training set does not establish generalization. Intermediate ghosting's lower
median is not evidence of a general benefit (fewer frames are within 0.5 m).
A preliminary 300-step run is preserved as `learned_corners_v1`; the final run
uses a fixed longer budget rather than selecting a checkpoint by held-out score.


### Correction: dynamic motion blur instead of persistence

The previous ghost experiment averages colors from old frames and does not model
exposure blur. The default is now `--corruption motion`; `--corruption ghost`
retains the old experiment explicitly.

Motion blur uses successive RGB frames and measured head quaternions to estimate
a spatially varying displacement field. Quaternion compensation aligns the
previous view before dense DIS optical flow estimates residual displacement.
Only **current-frame colors** are sampled along that field, averaged in linear
light over a centered exposure. The previous frame supplies displacement only.
The effect is applied before rectification and network inference.

Because captures jump between settled poses, exposure strength is a fraction of
the sampled frame interval, not a calibrated camera shutter in milliseconds.
Training uses clean images and fraction 0.04; evaluation uses 0, 0.02 and 0.05
on the same held-out route. The first frame of each pass remains clean. Flow at
occlusions or across large jumps remains approximate; this does not simulate
true continuous motion, newly revealed surfaces or rolling shutter. The initial
0.65 exposure experiment was stopped after image inspection showed excessive
smearing for these sparse captures.

```bash
python -m map2loc.simulator.learned_experiment \
  --train outputs/ursoccerlab/yaw_head48_v1 outputs/ursoccerlab/corners101 \
  --eval outputs/ursoccerlab/corners202 --corruption motion --steps 3000 \
  --out outputs/ursoccerlab/learned_motion_short_exposure
```

Corrected 3,000-step motion-blur experiment on the 64 held-out views:

| Exposure fraction | Median camera XY error | Mean error | Within 0.5 m |
| --- | ---: | ---: | ---: |
| 0 (clean) | 0.956 m | 1.916 m | 30/64 |
| 0.02 | 1.294 m | 1.907 m | 29/64 |
| 0.05 | 1.257 m | 1.853 m | 29/64 |

These rows use the same newly trained motion-augmented model. They are not a
fixed-checkpoint comparison against the previous ghost-trained model. Per-frame
`blur_stats` record the estimated flow and blur-length 95th percentiles in pixels.
Large localization failures still occur (90th percentile above 4 m).

### MSE optimization with frozen-checkpoint geometric fusion

`map2loc.simulator.optimize` retains the learned checkpoint and scores full-view
field markings against the map. It penalizes observed lines far from map lines,
and green observations that contradict map lines. This supplements the learned
5.12 m BEV crop with geometry from farther visible markings. Inputs remain RGB,
measured head quaternion, known camera geometry and fixed root height; simulator
XY and velocity do not enter prediction.

MSE is explicitly `mean((dx_m**2 + dy_m**2)/2)` (m², averaged over two coordinates).
`mean_squared_distance_m2` is twice that value. Median and p90 Euclidean errors,
and the fraction within 0.5 m, are also recorded so MSE cannot hide tradeoffs.

The existing seed-202 route is now **validation**, not final test data. The search
compares 24 geometry/learned combinations and three neural-only expectation
baselines over clean, 0.02 and 0.05 exposure versions. Validation selects learned
weight 1, geometry temperature 0.015 m, and posterior-mean prediction. Across all
192 validation variants, coordinate MSE decreases from 3.828 to 1.226 m². The
posterior mean can lie between ambiguous modes; this is an MSE-oriented decision,
not a claim that every output is a single confident match. Best geometry-only
expectation gives 2.133 m² and best hybrid peak selection 1.695 m² on validation.

```bash
python -m map2loc.simulator.optimize \
  --data outputs/ursoccerlab/corners202 \
  --checkpoint outputs/ursoccerlab/learned_motion_short_exposure/model.pt \
  --out outputs/ursoccerlab/mse_validation202
python -m map2loc.simulator.optimize \
  --data outputs/ursoccerlab/corners303 \
  --checkpoint outputs/ursoccerlab/learned_motion_short_exposure/model.pt \
  --selection outputs/ursoccerlab/mse_validation202/selection.json \
  --out outputs/ursoccerlab/mse_test303
```

Final evaluation uses newly captured seeds 303 and 404 with 64 frames each. The
selection file and checkpoint SHA256 are frozen before evaluation. Training data
and weights are unchanged; baseline and optimized methods consume identical
blurred images. These are new routes in the same simulator/field, not independent
environments. Frames and three corruption versions are correlated; no statistical
significance or broad generalization claim is made. Scores, per-frame predictions,
selected settings, metric definitions and comparison videos are retained.

Final test results, pooled across seeds 303 and 404 (128 frames per condition):

| Exposure fraction | Baseline MSE | Optimized MSE | Baseline median | Optimized median |
| --- | ---: | ---: | ---: | ---: |
| 0 | 5.156 m² | 1.673 m² | 1.326 m | 0.591 m |
| 0.02 | 4.875 m² | 1.586 m² | 1.297 m | 0.421 m |
| 0.05 | 4.959 m² | 1.554 m² | 1.326 m | 0.451 m |

At exposure 0.05, 52/128 baseline estimates and 65/128 optimized estimates are
within 0.5 m. The 90th percentile improves from 5.072 to 3.112 m, so significant
outliers remain. Near-corner frames (22) improve from 9.136 to 3.127 m² MSE;
head-down frames above 50 degrees (50) improve from 5.573 to 3.187 m². However,
head-down estimates within 0.5 m decrease from 14/50 to 10/50: lower MSE is not
uniformly better close-range accuracy. Geometry cannot resolve every ambiguous
view of plain turf or a single line, and posterior averaging can trade precision
for smaller catastrophic errors.

The frozen selection is also committed as
`map2loc/simulator/fusion_selection.json`; use it with `--selection` and the exact
motion-trained checkpoint (SHA256 recorded in that file). Remote artifacts:
`outputs/ursoccerlab/mse_validation202`, `mse_test303`, and `mse_test404` under
`~/workspace/map2loc-urs-experiment`. The local combined video is
`outputs/ursoccerlab/mse_comparison.mp4`. The two new routes were captured on the
running simulator; all score evaluation ran on A843-RX9070xt's RX 9070 XT.

### Causal history over the trajectory

`VisualPositionFilter` (`map2loc/temporal/visual_filter.py`) carries a 2D position
probability distribution between frames. It is initialized uniformly at each
route pass, never at the true starting position. Each prediction shifts that
belief by image-derived camera displacement, spreads uncertainty, then updates
it with the current learned/geometric likelihood. It uses fractional warps of
probability mass with zero padding, rather than wrapping map boundaries or
blurring log probabilities. Prediction returns the posterior mean and variance.
The API is causal: one `update(log_likelihood, motion)` per incoming frame, with
no future observations or end-of-trajectory smoothing.

Relative motion comes from two camera images rectified into world-aligned ground
views using measured head quaternions and known fixed camera-height geometry.
Masked correlation estimates translation despite changing visible footprints.
Poor overlap, ambiguous peaks and fewer than 20 line-support cells reject a
measurement. Missing motion invokes a short decaying extrapolation of the last
accepted visual increment and larger process uncertainty. Simulator velocities,
commanded routes and true positions are never motion/filter inputs. Pose labels
are read afterwards for errors and diagnostic motion accuracy only.

Settings are selected on seed 202 and saved in
`map2loc/simulator/temporal_selection.json`. The observation checkpoint and fusion
settings are unchanged. Comparison runs include independent-frame inference and
a zero-displacement ablation with the same acceptance/noise schedule. The initial
feature-tracking and low-line-support experiments are retained as diagnostics;
only the final supported-plane alignment is used by the committed configuration.

```bash
python -m map2loc.simulator.temporal_experiment \
  --data outputs/ursoccerlab/corners303 \
  --scores outputs/ursoccerlab/mse_test303 \
  --selection map2loc/simulator/temporal_selection.json \
  --out outputs/ursoccerlab/history_supported303
```

The `--scores` directory must contain matching frozen observation scores from
`map2loc.simulator.optimize`. Each blurred image stream is processed in temporal
order; history resets only at the start of a route pass. The saved manifest states
causality and permitted inputs, while `motion.json` and `predictions.json` retain
every accepted/rejected motion and the resulting belief variance.

A new seed-505 route has 96 steps per pass (192 images), with median body movement
0.373 m between observations, versus 1.377/1.170 m on seeds 303/404. These remain
privileged stop-and-observe simulator captures; temporal inference respects their
ordered continuous path, but they do not validate physical walking or continuous
camera exposure. First-frame initialization and subsequent history are scored
together without dropping difficult startup frames.

Final causal-history results (coordinate MSE in m²):

| Sequence set | Exposure | Independent MSE | Causal-history MSE |
| --- | ---: | ---: | ---: |
| Seeds 303 + 404, 128 frames | 0 | 1.673 | 1.030 |
| Seeds 303 + 404 | 0.02 | 1.586 | 1.022 |
| Seeds 303 + 404 | 0.05 | 1.554 | 0.952 |
| Dense seed 505, 192 frames | 0 | 2.176 | 1.767 |
| Dense seed 505 | 0.02 | 2.125 | 1.721 |
| Dense seed 505 | 0.05 | 2.105 | 1.725 |

At exposure 0.05 on the dense route, median error improves from 0.558 to 0.242 m,
and estimates within 0.5 m increase from 92/192 to 108/192. The p90 error is still
3.564 m (independent: 3.521 m), so persistent wrong hypotheses remain. Sparse-route
MSE improves 39%, but median error worsens from 0.451 to 0.598 m and within-0.5-m
estimates decrease from 65/128 to 62/128. History is not uniformly beneficial.

Controls separate history from calibration and displacement. On the dense route
at exposure 0.05, independent estimates using the filter's same softened
likelihood give 1.866 m² MSE; causal history gives 1.725 m². Zeroing motion while
keeping the same confidence/noise schedule gives 1.723 m²: most of the dense-route
gain comes from accumulated belief, with no demonstrated MSE gain from motion
shifts there. On the sparse routes, the zero-displacement control gives 1.121 m²
versus 0.952 m² with visual motion. All three controls are stored per frame.

Validation chose 0.5 m process sigma for accepted motion, 1.4 m for rejected motion,
observation weight 0.5, global mixing 0.001, prior exponent 0.8 and decaying visual
motion carry. These settings remain fixed across test routes. The line-support
check was motivated by motion errors on validation seed 202. Earlier feature and
plane-alignment variants are preserved in separate artifact directories rather
than overwritten. Evaluation on these routes is iterative engineering evidence,
not a blind statistical generalization study.

Final remote output directories are `history_supported_validation202`,
`history_supported303`, `history_supported404`, and `history_supported_dense505`
under `~/workspace/map2loc-urs-experiment/outputs/ursoccerlab`. The local dense-route
video is `outputs/ursoccerlab/history_dense_comparison.mp4` (yellow GT, red
independent inference, cyan causal history). No trajectory tail is needed to make
the estimator temporal: the actual probability state persists between frames.
