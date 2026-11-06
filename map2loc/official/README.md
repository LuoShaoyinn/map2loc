# Official OrienterNet on URSoccerLab

This path uses the unmodified Facebook Research implementation at submodule
commit `0280b6e248046dc7d714801d7d0651a162fd5a63`, with the authors' released
`orienternet_mgl.ckpt`. It is an adaptation to URS, not reproduction of the
MGL/KITTI benchmark numbers. The repository's older LearnedMatcher, geometric
fusion and temporal filter are custom baselines, not the paper implementation.

Paper: https://arxiv.org/abs/2304.02009
Source: https://github.com/facebookresearch/OrienterNet
Weights: https://cvg-data.inf.ethz.ch/OrienterNet_CVPR2023/orienternet_mgl.ckpt

## Preserved method

`maploc.models.orienternet.OrienterNet` supplies the ResNet101 image FPN,
33-bin learned scale/depth lifting, polar-to-Cartesian BEV projection, BEV
confidence network, VGG19 semantic map encoder, FFT correlation over XY/yaw,
and trilinearly interpolated pose NLL. Adam uses the upstream LR of 1e-4.
All layers are fine-tuned. The released checkpoint includes a learned map
prior. Its VGG19 configuration is authoritative for this implementation;
the paper's prose names VGG16. No custom geometric matching or XY filter is
combined with the official network.

## Explicit dataset adaptations

* `res/example.png` has 100 px/m and covers 10.6 x 7.8 m. The semantic map uses
  soccer surface (area ID 1), white markings (way ID 1), and empty nodes.
  A 20 cm cell is marked as a line if any source pixel is white. These are
  soccer labels, not OSM labels. Only the three embedding tables are randomly
  reinitialized; all other checkpoint parameters must match exactly.
* A padded 11.2 x 8 m map at 5 pixels/m replaces the urban crop at 2 pixels/m.
  BEV forward/lateral extent is 8 m instead of 32 m. This changes spatial
  sampling, not the learned projection or matching architecture.
* Simulator head quaternion supplies gravity. The official `rectify_image`
  rotates camera rays onto a horizontal optical axis; it does not project
  them onto the ground. The head-to-optical basis and 60 degree horizontal
  FOV follow the URS renderer. RGB is resized to 320 x 240 and padded to
  320 x 256. Looking steeply down loses content after gravity rectification;
  per-frame valid fractions are recorded without removing difficult frames.
* Camera optical-center XY (including head offset), not robot root XY, is the
  target. Simulator XY is used only for training labels and evaluation.
  Full-pose inference receives RGB, calibration and the semantic map.
  A separately reported result applies upstream's 5 degree measured-yaw
  prior, permitted by this experiment's accurate head attitude assumption.
* Training uses recorded routes, clean RGB and the previously accepted
  flow-based exposure blur (strength .04). Evaluation uses 0/.02/.05.
  This corruption approximates exposure from discrete frames; it is not
  ground-truth renderer motion blur.
* Training uses 64 yaw bins and batch 2 on one RX 9070 XT. Evaluation uses
  256 bins, as upstream pretrained evaluation does. Validation NLL selects
  the checkpoint. This bounded fine-tuning run is not the paper's multiday
  training schedule. Route splits share a scene and are not unseen-map tests.

## Sequence scope

The paper's sequence inference requires relative camera poses from an external
SLAM/odometry system. Hacked URS motion provides no usable velocity. Accurate
head rotation alone cannot provide relative translation. The initial run here
is therefore single-image localization. It neither substitutes the earlier
custom filter nor feeds simulator XY into sequence inference. Official sequence
fusion remains to be evaluated once an image-derived relative-pose source is
provided and its uncertainty/scale are documented.

## Run on A843-RX9070xt

Initialize the submodule with `git submodule update --init`. Use an environment
with matching ROCm torch/torchvision, OmegaConf 2.3.0, pytorch-lightning 2.5.1,
torchmetrics 1.7.1, scipy, OpenCV and Pillow. Keep the simulator's environment
unchanged; the experiment uses `.venv-official` with a path to its existing
ROCm torch installation. Never let a dependency resolver replace that torch
with a CUDA wheel. Runtime versions and checkpoint SHA256 are recorded.

```bash
CUDA_VISIBLE_DEVICES=0 .venv-official/bin/python -m map2loc.official.urs \
  --checkpoint outputs/official_orienternet/orienternet_mgl.ckpt \
  --train outputs/ursoccerlab/corners101 \
  --validation outputs/ursoccerlab/corners202 \
  --eval outputs/ursoccerlab/corners303 outputs/ursoccerlab/corners505_dense \
  --steps 300 --out outputs/official_orienternet/urs_v1
```

MSE is mean squared Euclidean camera XY error, in m², not per-axis MSE.
Validation uses actual recorded simulator images on the remote GPU. No local
unit tests or synthetic smoke runs are used.
