# Results — attitude-aware BEV localization (`bev-v2`)

Hardware: AMD Radeon RX 9070 XT (gfx1201), ROCm 7.2.4, PyTorch 2.11.
Observation model: 256×256 oblique pinhole camera, fixed height 180 map px,
footprint 120 map px (≈37° FOV), yaw/pitch/roll measurement sigma 5°.

Training command (4000 steps + 8000 fine-tune steps, batch 32):

```bash
python -m map2loc.train --device cuda --steps 4000 --batch-size 32 \
    --out outputs/checkpoints/obs_bev.pt
python -m map2loc.train --device cuda --init-checkpoint outputs/checkpoints/obs_bev.pt \
    --steps 8000 --batch-size 32 --lr 2e-4 \
    --out outputs/checkpoints/obs_bev_long.pt
```

Final training log: `nll 3.89`, `mask 0.84`, exact-cell `top1 0.31`
(grid resolution 8 px).

Evaluation command:

```bash
python -m map2loc.evaluate --device cuda --seq-len 40 \
    --learned-checkpoint outputs/checkpoints/obs_bev_long.pt \
    --out outputs/eval/results_bev_final.json
```

All methods see identical trajectories, observations, measured attitude and
validity masks. `succ@16/10` = fraction of frames with position error < 16 px
and yaw error < 10° (yaw is the measured input for every method).

| scenario | method | pos med (px) | pos p90 (px) | succ@16/10 |
|---|---|---:|---:|---:|
| seen_clean | classical | 280.3 | 587.8 | 0.050 |
| seen_clean | classical+filter | 28.9 | 50.5 | 0.250 |
| seen_clean | learned | 31.0 | 209.2 | 0.175 |
| seen_clean | **learned+filter** | **15.2** | **46.5** | **0.525** |
| seen_mild | classical | 317.9 | 532.8 | 0.025 |
| seen_mild | classical+filter | 27.7 | 343.2 | 0.175 |
| seen_mild | learned | 53.1 | 376.5 | 0.075 |
| seen_mild | **learned+filter** | 28.8 | **64.7** | **0.250** |
| seen_moderate | classical | 564.5 | 765.7 | 0.075 |
| seen_moderate | classical+filter | 620.9 | 742.0 | 0.150 |
| seen_moderate | learned | 34.7 | 260.9 | 0.175 |
| seen_moderate | **learned+filter** | **14.6** | **31.3** | **0.650** |
| seen_severe | classical | 324.3 | 533.5 | 0.050 |
| seen_severe | classical+filter | 67.4 | 390.3 | 0.250 |
| seen_severe | learned | 21.7 | 279.8 | 0.300 |
| seen_severe | **learned+filter** | **7.9** | **16.8** | **0.850** |
| scratches | classical | 412.9 | 766.4 | 0.025 |
| scratches | classical+filter | 16.8 | 102.8 | 0.425 |
| scratches | learned | 33.7 | 274.5 | 0.275 |
| scratches | **learned+filter** | 19.3 | 45.1 | **0.425** |
| pixel_dropout | classical | 475.2 | 651.8 | 0.000 |
| pixel_dropout | classical+filter | 91.9 | 463.5 | 0.000 |
| pixel_dropout | learned | 180.4 | 490.8 | 0.025 |
| pixel_dropout | **learned+filter** | 128.0 | 406.7 | 0.025 |
| unseen_map_rot90 | classical | 277.1 | 595.0 | 0.100 |
| unseen_map_rot90 | classical+filter | 17.6 | 334.5 | 0.425 |
| unseen_map_rot90 | learned | 19.9 | 303.1 | 0.325 |
| unseen_map_rot90 | **learned+filter** | **7.8** | 47.5 | **0.725** |
| unseen_region | classical | 322.6 | 763.5 | 0.100 |
| unseen_region | classical+filter | 24.8 | 39.1 | 0.250 |
| unseen_region | learned | 304.4 | 676.5 | 0.150 |
| unseen_region | **learned+filter** | **14.4** | **24.8** | **0.525** |
| wrong_global_init | classical | 293.6 | 788.4 | 0.050 |
| wrong_global_init | classical+filter | 36.9 | 529.3 | 0.200 |
| wrong_global_init | learned | 34.9 | 242.0 | 0.175 |
| wrong_global_init | **learned+filter** | **21.7** | **40.6** | **0.450** |
| recovery_distractor | classical | 619.9 | 701.0 | 0.000 |
| recovery_distractor | classical+filter | 646.5 | 696.6 | 0.000 |
| recovery_distractor | learned | 398.4 | 576.5 | 0.000 |
| recovery_distractor | learned+filter | 344.4 | 561.1 | 0.000 |

## Validation video (best method)

`map2loc.replay` replays one benchmark scenario through exactly the same
rectification, matcher and filter code used by the benchmark. The rendered
`seen_severe` run reproduces the table above frame for frame: **7.9 px median,
16.8 px p90, 0.85 success** over 40 frames.

```bash
python -m map2loc.replay --device cuda --scenario seen_severe --seq-len 40 \
    --learned-checkpoint outputs/checkpoints/obs_bev_long.pt \
    --out outputs/videos/replay_seen_severe_learned_filter.mp4
```

Artifacts: `docs/replay_seen_severe.mp4` (H.264, 8 s at 5 fps) and
`docs/replay_seen_severe_poster.png`. Every frame shows the map with the
ground-truth route, the filtered and per-frame estimates, the raw perspective
observation, the rectified BEV patch, the map template at the estimate, the
filter posterior (whole map and zoom), and the position-error history. The same
command also renders `seen_moderate` (14.6 px / 0.65) and `unseen_map_rot90`
(7.8 px / 0.725).

## What changed and why

1. **Rectify to BEV using the measured attitude.** The observation is a
   perspective quadrilateral. A single homography, built from measured
   `(yaw, pitch, roll)`, height and focal scale, resamples it into a
   map-axis-aligned patch. Verified against the map at nadir and at oblique
   pitches (mean absolute difference 0.1–0.5 grey levels over valid pixels).
2. **Search only what is not measured.** Residual yaw (±one 5.6° grid step) and
   seven focal-scale hypotheses cover the unmeasured zoom and 5° yaw noise.
   Residual pitch is marginalised over `{-5°, 0°, +5°}` so the image-centre
   lever arm (`−h/tan(pitch)`) does not bias the recovered camera position.
3. **Position-only temporal filter.** Because attitude is measured, the filter
   tracks `(x, y)` instead of `(x, y, yaw)`. It uses the provided ego-motion and
   the yaw-marginalised image likelihood, with a 5 % global-mixing term for
   relocalisation.
4. **Reliability-supervised learned matcher.** A CNN embeds the BEV patch and
   predicts a per-pixel reliability map supervised by the rectified validity
   mask. The map is embedded once at the matching-grid resolution and the
   patch template is correlated against it.

## Caveats

* Many map patches are almost featureless green. On such frames the
  per-frame likelihood is genuinely uninformative; the filter has to bridge
  them with odometry.
* `pixel_dropout` (80 % invalid pixels, so only ≈4 % of the patch survives)
  remains very hard; `recovery_distractor` for seed 21 happens to run through a
  mostly featureless part of the map, so even the per-frame learned matcher
  stays hundreds of pixels away there.
* `classical+filter` on `seen_moderate` locks onto a false peak; the learned
  matcher is far more stable under heavy corruption.