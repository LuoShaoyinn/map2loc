# bev2loc — BEV-to-map localization

A known bird's-eye-view (BEV) map (e.g. `res/example.png`) is given. At test
time the agent only sees a small, heavily degraded crop of it. This package
contains the data infrastructure, several localization baselines, temporal
fusion methods, and an identical-protocol benchmark for all of them.

Branch layout:

* `main` — infra and data generator: rendering, corruptions, validity masks,
  torch datasets and the batched tensor generator.
* `OrienterNet-ds` — models and evaluation: classical / learned matching, F3Loc
  style SE(2) filtering, ConvGRU baseline, training and benchmark CLIs.
* `F3loc-ds` — archived perspective-depth reproduction, not used here.

```
res/example.png              the fixed map
bev2loc/map.py               map loading, coordinate frames, regions
bev2loc/trajectory.py        smooth waypoint / random-walk pose sequences
bev2loc/observation.py       crop windows out of the map, exact ground truth
bev2loc/augment.py           geometry, noise, scratches, occlusions + validity mask
bev2loc/tensor_generator.py  batched on-device renderer for online training
bev2loc/dataset.py           on-the-fly torch Datasets (single frame / sequence)
bev2loc/matching/grid.py     shared (x, y, yaw) grid and likelihood volumes
bev2loc/matching/classical.py rotation-aware masked NCC baseline
bev2loc/matching/learned.py  CNN encoders + dense correlation (OrienterNet-style)
bev2loc/temporal/histogram_filter.py  F3Loc-style SE(2) Bayes filter
bev2loc/temporal/rnn.py      ConvGRU sequential baseline
bev2loc/evaluation/          corruption levels, map splits, scenarios, metrics
bev2loc/train.py             observation / RNN training CLIs
bev2loc/evaluate.py          benchmark CLI
```

## Quick start (uv)

```bash
# ROCm machine (this repo was developed on one)
uv sync --extra rocm
uv run --extra rocm python -m bev2loc preview --out-dir outputs/previews
uv run --extra rocm python -m pytest
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
  recomputed from the composed homography.
* Every matcher produces a dense log-likelihood over the same lattice:
  `(n_yaw, ny, nx)` with `resolution` map pixels per cell. The benchmark uses a
  single `PoseGrid`, so per-frame methods, filters and the RNN all decode the
  same way (argmax over the volume).

## Models

### Classical matching

`ClassicalNCCMatcher` resamples the observation into the map frame for every
heading bin and runs masked normalised cross-correlation over the whole map.
All moments are computed over valid template pixels only, so border padding,
scratches, smudges and occluders do not create false peaks. Near-constant
templates (featureless grass) are rejected. Output is a log-likelihood volume
over `(x, y, yaw)`.

### Learned matching (OrienterNet-style)

`LearnedMatcher` has two small CNNs:

* a local BEV encoder that also predicts a **reliability map**, and
* a global map encoder at matching-grid resolution.

The local feature map is rotated into every heading bin, weighted by predicted
reliability, and correlated with the map features by one batched convolution.
The result is a dense likelihood over `(x, y, yaw)`. The reliability head is
supervised with the corruption validity mask: out-of-map border, salt-and-pepper
outliers, scratches/dust/smudges and occluders are marked invalid. Invalid
pixels are weighted 5× in the BCE loss, so the model learns to downweight
regions that cannot be trusted.

The same frozen observation model is used by the per-frame learned matcher,
the learned + histogram filter, and the RNN. That keeps temporal comparisons
fair: only the fusion method changes.

### Temporal filtering

`SE2HistogramFilter` keeps F3Loc's useful part—a discrete Bayes filter on
`(x, y, yaw)` whose transition is a grouped 2D convolution (one group per
heading, Gaussian translation kernel expressed in that heading's body frame)
followed by a circular 1D convolution over heading. It fuses any likelihood
volume, uses available ego-motion `(forward, right, dyaw)`, and works in log
space. F3Loc's perspective-depth and multiview front ends are intentionally not
implemented.

### RNN baseline

`ConvGRULocalizer` runs a ConvGRU over a coarse learned likelihood volume plus
globally pooled observation features and is trained with pose NLL while the
observation encoder is frozen. The recurrent state refines the heading
distribution of every frame. The optional coarse spatial-residual head is
disabled for the final checkpoint: at evaluation time over long trajectories it
was strong enough to flip the argmax away from a good per-frame peak, whereas
the heading refinement is stable. Temporal position fusion is left to the
histogram filter, which uses ego-motion explicitly.

## Training

Online tensor generation is the default: `--generator tensor` renders batches
on the GPU with `TensorObservationGenerator` while the model trains, so no
DataLoader workers or disk cache are needed. `--generator dataset` keeps the
numpy/cv2 `PoseDataset`/`TrajectoryDataset` path for reference.

```bash
# 1. Observation model (shared by learned, filter and RNN comparisons)
python -m bev2loc.train --mode observation --generator tensor \
    --steps 3000 --batch-size 8 --corruption mild \
    --out outputs/checkpoints/obs_mild.pt

# Optional robustness fine-tune with occlusions / scratches
python -m bev2loc.train --mode observation --generator tensor \
    --steps 1000 --batch-size 8 --corruption moderate \
    --init-checkpoint outputs/checkpoints/obs_mild.pt \
    --out outputs/checkpoints/obs_moderate.pt

# 2. ConvGRU on frozen observation features
python -m bev2loc.train --mode rnn --generator tensor \
    --learned-checkpoint outputs/checkpoints/obs_moderate.pt \
    --steps 1200 --batch-size 2 --seq-len 8 --step-stride 6 \
    --out outputs/checkpoints/rnn.pt

# stable heading-refining variant (spatial residual disabled)
python -m bev2loc.train --mode rnn --generator tensor \
    --learned-checkpoint outputs/checkpoints/obs_moderate.pt \
    --init-checkpoint outputs/checkpoints/rnn.pt --no-spatial-residual \
    --steps 900 --batch-size 2 --seq-len 12 --step-stride 6 \
    --out outputs/checkpoints/rnn_yaw.pt
```

Training logs report NLL, reliability-mask loss and training top-1 bin accuracy.
On the RX 7900 XTX, the tensor generator produces ~500 observations/s; the full
observation training step (batch 8, 256 px crops, 64 headings) runs at roughly
40–45 observations/s including forward, backward and optimizer.

## Evaluation

The benchmark renders one observation sequence per scenario and feeds the exact
same images and masks to every method.

```bash
python -m bev2loc.evaluate \
    --learned-checkpoint outputs/checkpoints/obs_moderate.pt \
    --rnn-checkpoint outputs/checkpoints/rnn_yaw.pt \
    --repeats 3 \
    --out outputs/eval/results.json
```

Scenarios in `default_scenarios()` cover:

* clean, mild, moderate and severe corruption;
* scratches, occlusions and strong geometric warping;
* an unseen structural map variant (`rot90`);
* an unseen spatial region of the training map;
* global initialization with a deliberately wrong prior (a broad
  200 px / 60° prior centred on the mirrored pose, so temporal methods can
  recover rather than being trapped by a confident prior);
* recovery after five distractor frames from the wrong place.

Metrics include mean / median / p90 position and heading error, success rates at
several thresholds, and the first frame after which the estimate stays within
16 px / 10° for three consecutive frames (recovery). Results are JSON; a compact
table is printed and `summary_table()` is available programmatically.

### Replay video

Render one trajectory as a synchronised three-panel video: the map with the true
pose (green) and the estimate (red), the corrupted BEV observation the model
actually received, and the method's posterior over position (max over heading).

```bash
python -m bev2loc.replay \
    --scenario seen_moderate --method learned+filter \
    --learned-checkpoint outputs/checkpoints/obs_moderate.pt \
    --rnn-checkpoint outputs/checkpoints/rnn_yaw.pt \
    --out outputs/videos/bev2loc_replay.mp4 \
    --still outputs/videos/bev2loc_replay_frame.png
```

`--method` accepts `classical`, `classical+filter`, `learned`, `learned+filter`
and `rnn`. `--show-mask` tints unreliable pixels in the observation panel.
Checkpoints default to the ones above; the generated sample is in
`outputs/videos/bev2loc_replay.mp4`.

### Reference results

Three independent trajectory seeds per scenario, 40 frames each, with the
checkpoints above. Each cell is **success@16 px / 10° / median position error**
(median of the three repeats):

| scenario | classical | classical+filter | learned | learned+filter | rnn |
|---|---|---|---|---|---|
| seen_clean | 43% / 222px | 61% / 6px | 88% / 5px | **94% / 4px** | 68% / 6px |
| seen_mild | 31% / 313px | 20% / 560px | 82% / 8px | **99% / 5px** | 61% / 12px |
| seen_moderate | 6% / 324px | 24% / 226px | 37% / 18px | **72% / 10px** | 29% / 30px |
| seen_severe | 3% / 360px | 5% / 277px | 10% / 135px | **26% / 21px** | 7% / 195px |
| scratches | 10% / 359px | 20% / 474px | 31% / 72px | **76% / 10px** | 26% / 119px |
| occlusion | 9% / 379px | 28% / 305px | 22% / 102px | **86% / 7px** | 17% / 201px |
| geometric | 0% / 348px | 0% / 335px | 5% / 225px | **19% / 24px** | 6% / 230px |
| unseen_map_rot90 | 8% / 360px | 8% / 386px | 29% / 31px | **68% / 11px** | 22% / 165px |
| unseen_region | 6% / 333px | **35% / 18px** | 0% / 529px | 0% / 522px | 5% / 500px |
| wrong_global_init | 12% / 318px | 0% / 350px | 38% / 13px | **78% / 10px** | 20% / 149px |
| recovery_distractor | 6% / 403px | 5% / 92px | 25% / 23px | **56% / 12px** | 19% / 26px |

The learned observation model dominates per-frame localization on clean and
mild data, and the SE(2) filter adds the most under moderate/severe corruption,
scratches and occlusions (e.g. 86% success under occlusion versus 22% per
frame). The filter also recovers from a deliberately wrong prior, while
per-frame decoding cannot. The RNN baseline is stable but does not beat the
filter; with a coarse spatial residual it could lock onto wrong modes, so the
released checkpoint refines heading only.

Two limitations are visible in the table. `geometric` is hard for every method
because the perspective warp violates the planar BEV crop assumption. On
`unseen_region` the learned matcher locks onto the 180°-rotated copy of the
pitch (median yaw ≈ 170°); that region is nearly symmetric with the training
region, so the observation alone is genuinely ambiguous, and a consistently
wrong observation cannot be fixed by temporal fusion. Less symmetric maps (or a
map-context/sequence-level matcher) are needed there.

For a true unseen-map evaluation, use
`build_map_split_from_paths(train_path, [eval_path, ...])`; with only one map the
benchmark uses deterministic structural variants (`rot90`, `mirror`, `recolor`)
as held-out maps.

## What is randomised

**Geometry** (`GeometryConfig`): zoom, x/y aspect, affine shear, extra roll,
small translation of the content, and mild perspective (oblique view). All are
composed into one homography, so ground truth stays exact.

**Photometry** (`PhotometricConfig`): defocus/motion blur, brightness/gamma/
colour tint, vignetting, illumination gradient, Gaussian + salt-and-pepper
noise, lens scratches (polylines), dust specks, smudges, large occluders and
JPEG artefacts. A per-pixel validity mask marks pixels that are not trustworthy
map content.

**Trajectory** (`TrajectoryConfig`): waypoint navigation by default (smooth
bounded-curvature arcs, broad coverage). Set `mode="random_walk"` for a local
OU-curvature meander. Poses stay inside the map's non-background bounding box
(the pitch), shrunk slightly.

## Notes and design choices

* Everything is procedural and seeded: datasets and benchmark sequences are
  reproducible.
* Unknown area outside the map is filled with the map's background colour by
  default (`border="background"`), so the border itself does not leak the
  position. The validity mask still marks it invalid.
* Many map patches are almost featureless green, so some observations are
  genuinely ambiguous even without corruption. That is a property of the task,
  not a bug; a model must learn to use the white pitch markings and reject
  uninformative frames.
* F3Loc is not applicable end-to-end to this task because its first stage needs
  a perspective RGB camera with floor/wall/ceiling cues. Only its SE(2)
  histogram filter is retained here.