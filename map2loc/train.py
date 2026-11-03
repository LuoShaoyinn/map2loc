"""Train the learned observation model (and optional temporal baseline).

The online tensor generator renders oblique camera views on the GPU and
rectifies them to BEV with the measured attitude.  Training uses the same
5 deg attitude noise and the same heavy corruption model as evaluation.
"""

from __future__ import annotations

import argparse
import copy
import time
from pathlib import Path

import numpy as np
import torch

from data_gen.tensor_generator import TensorPoseSampler

from .bev import BEVSpec
from .evaluation.protocols import build_map_split, corruption_level
from .matching.grid import PoseGrid, map_raster
from .matching.learned import LearnedMatcher, LearnedMatcherConfig, grid_indices
from .tensor_generator import BEVTensorGenerator


def _device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _make_generator(device: torch.device, seed: int) -> torch.Generator:
    gen = torch.Generator(device=device)
    gen.manual_seed(int(seed))
    return gen


def train_observation(args) -> Path:
    device = _device(args.device)
    split = build_map_split(args.map, out_dir=args.map_variants_dir)
    map_ = split.train_map
    grid = PoseGrid.from_map(
        map_, resolution=args.resolution, n_yaw=args.n_yaw
    )
    level = corruption_level(args.corruption)
    cfg = copy.deepcopy(level.cfg)
    cfg.geometry.obs_width = args.obs_size
    cfg.geometry.obs_height = args.obs_size

    bev_spec = BEVSpec(args.patch_px, args.map_px_per_px)
    lcfg = LearnedMatcherConfig(
        patch_px=args.patch_px,
        map_px_per_px=args.map_px_per_px,
        feature_dim=args.feature_dim,
        base_channels=args.base_channels,
        obs_stride=args.obs_stride,
        grid_resolution=grid.resolution,
        n_yaw=grid.n_yaw,
    )
    matcher = LearnedMatcher(lcfg, grid).to(device)
    if args.init_checkpoint:
        init = torch.load(args.init_checkpoint, map_location="cpu", weights_only=False)
        matcher.load_state_dict(init["state_dict"])
        print(f"initialised from {args.init_checkpoint}", flush=True)
    opt = torch.optim.AdamW(matcher.parameters(), lr=args.lr, weight_decay=1e-4)

    tgen = BEVTensorGenerator(
        map_.image,
        cfg.geometry,
        cfg.photometric,
        cfg.measurement,
        bev_spec=bev_spec,
        device=device,
    )
    sampler = TensorPoseSampler(split.train_region, device=device)
    gen = _make_generator(device, args.seed)
    data = map_raster(map_, grid, grayscale=False, normalize=True, device=device)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    running = None
    for step in range(1, args.steps + 1):
        poses = sampler.sample(args.batch_size, gen)
        batch = tgen.sample_bev(
            poses,
            gen,
            geometry=level.geometry,
            photometric=level.photometric,
            scale_jitter=args.scale_jitter,
        )
        bev = batch["bev"]
        masks = batch["bev_valid"]
        center = batch["center_xy"]
        map_feats = matcher.encode_map(data["image"])
        out = matcher(bev[:, None], map_feats, valid_mask=masks[:, None])
        nll = matcher.nll(out["xy_logits"], center)
        loss = nll + args.mask_weight * out.get(
            "mask_loss", torch.zeros((), device=device)
        )
        top1 = (
            out["xy_logits"].flatten(1).argmax(1)
            == grid_indices(grid, center)
        ).float().mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(matcher.parameters(), args.grad_clip)
        opt.step()

        running = float(loss.detach()) if running is None else 0.95 * running + 0.05 * float(loss.detach())
        if step % args.log_every == 0 or step == 1:
            print(
                f"step {step:5d}  loss {float(loss.detach()):8.4f}  "
                f"nll {float(nll.detach()):8.4f}  "
                f"mask {float(out.get('mask_loss', torch.zeros(()))):.4f}  "
                f"top1 {float(top1):.3f}  avg {running:8.4f}  {time.time()-t0:6.1f}s",
                flush=True,
            )
        if args.save_every and step % args.save_every == 0:
            _save_ckpt(out_path, lcfg, matcher, grid, args, step)

    _save_ckpt(out_path, lcfg, matcher, grid, args, args.steps)
    print(f"saved {out_path}")
    return out_path


def _save_ckpt(path: Path, cfg: LearnedMatcherConfig, model, grid, args, step: int):
    torch.save(
        {
            "config": cfg.to_dict(),
            "grid": {"resolution": grid.resolution, "n_yaw": grid.n_yaw},
            "state_dict": model.state_dict(),
            "step": step,
            "args": vars(args),
        },
        path,
    )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--map", default="res/example.png")
    p.add_argument("--out", default="outputs/checkpoints/obs_bev.pt")
    p.add_argument("--mode", choices=("observation",), default="observation")
    p.add_argument("--corruption", default="moderate")
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="auto")
    p.add_argument("--resolution", type=float, default=8.0)
    p.add_argument("--n-yaw", type=int, default=64)
    p.add_argument("--obs-size", type=int, default=256)
    p.add_argument("--patch-px", type=int, default=164)
    p.add_argument("--map-px-per-px", type=float, default=2.0)
    p.add_argument("--feature-dim", type=int, default=32)
    p.add_argument("--base-channels", type=int, default=32)
    p.add_argument("--obs-stride", type=int, default=4)
    p.add_argument("--scale-jitter", type=float, default=0.06)
    p.add_argument("--mask-weight", type=float, default=0.5)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--log-every", type=int, default=25)
    p.add_argument("--save-every", type=int, default=0)
    p.add_argument("--map-variants-dir", default="outputs/maps")
    p.add_argument("--init-checkpoint", default=None)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    return train_observation(args)


if __name__ == "__main__":
    main()