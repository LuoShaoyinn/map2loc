# Initial official-method URSoccerLab reproduction

Stopped after the bounded reproduction run, at the user's request. No accuracy
tuning, replacement model, or rectification workaround was attempted afterward.
This is the official implementation adapted to URS, not a reproduction of the
paper's original dataset scores or full training schedule.

## Provenance and protocol

- Upstream: `0280b6e248046dc7d714801d7d0651a162fd5a63`, unmodified submodule.
- Observation adapter run: `0d7ab5c`; sequence dtype correction: `33243ff`.
- Released checkpoint SHA256:
  `47b65161e4ea0a6e4d8f1865dec677dd09756d51fde1435265346d0779d23d23`.
- Device: A843-RX9070xt, AMD Radeon RX 9070 XT, GPU 0.
- Train: corners101, 64 recorded frames with clean/.04 blur variants.
  Validation: corners202, 64 frames with .05 blur.
  Evaluation: corners303 (64 frames), corners505_dense (192 frames),
  each at blur strengths 0/.02/.05. Routes share the same soccer field.
- Full official network fine-tuned for 300 Adam steps, batch 2, seed 7,
  learning rate 1e-4. Validation NLL selected step 200 (20.6762).
  Training used 64 rotations; final evaluation used 256.
- Dataset changes and input/label boundaries: [README](README.md).
- Sequence: upstream RigidAligner, causal prefixes within 10-frame chunks.
  Camera velocity is derived from pose differences and camera timestamps
  because hacked motion reports no useful horizontal velocity. Add iid
  Gaussian noise, sigma .05 m/s per axis, seed 17, then integrate from zero.
  Exact head rotation is allowed. This is simulated odometry, not visual
  odometry; absolute positions are not supplied to the aligner.

## Completed results

Strong exposure approximation (.05), all frames retained. MSE is mean squared
Euclidean camera XY error in m², not per-axis MSE. Single noise realization;
these figures do not establish robustness across odometry noise seeds.

| Route | Full-pose single MSE | Full-pose sequence MSE | Measured-yaw single MSE | Measured-yaw sequence MSE |
| --- | ---: | ---: | ---: | ---: |
| corners303 | 16.403 | 15.771 | 10.847 | 5.689 |
| corners505_dense | 19.054 | Not run | 11.604 | Not run |

The measured-yaw variant uses the upstream 5 degree yaw prior. On corners303,
its median XY error changed from 2.816 m to 1.794 m with sequence fusion;
only 1/64 sequence estimates was within 0.5 m. Accuracy remains poor.
The denser route's sequence experiment was not launched before the user asked
to leave the result as-is.

## Observed limitation, left unchanged

With the current fixed output view, upstream gravity rectification produces
completely blank images for 28/64 validation frames and 28/64 corners303 frames
when the narrow-FOV camera points steeply down. Median validation valid-pixel
fraction is 0.141. No difficult frames were removed. This is an immediate
observed input limitation; it does not establish that it explains all error.
The small fine-tuning dataset and urban-to-soccer domain change also limit
what can be concluded from this run. No claim of successful localization or
paper-level accuracy is made.

## Artifacts

Remote root:
`~/workspace/map2loc-urs-experiment/outputs/official_orienternet/urs_v2/`

- `best.pt`: selected official-network checkpoint.
- `manifest.json`, `environment.json`, `training.jsonl`: observation run provenance.
- `evaluation_256rot/{corners303,corners505_dense}/`: predictions, metrics and
  float32 likelihood archives.
- `sequence303_noise005/`: sequence predictions, metrics, odometry and manifest.
- `sequence303.png`: trajectory/error plot.

Small artifacts and the plot are also copied under the same `outputs/` path
locally. The observation manifest was written before sequence evaluation and
its original sequence-status string is stale; `sequence303_noise005/manifest.json`
is the authoritative record of the completed sequence run.

All validation used recorded simulator images on the remote GPU. No local
unit tests or synthetic smoke experiments were run. Both experiment processes
completed, and no official experiment process remained running at closeout.
