"""Train the learned observation model and the ConvGRU temporal baseline.

Examples
--------
Train the observation matcher (the model shared by all temporal comparisons)::

    python -m bev2loc.train --mode observation --steps 400 --out checkpoints/obs.pt

Then train the ConvGRU on frozen observation features::

    python -m bev2loc.train --mode rnn --learned-checkpoint checkpoints/obs.pt \\
        --steps 100 --out checkpoints/rnn.pt
"""

from __future__ import annotations

import argparse
import copy
import itertools
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .config import InfraConfig
from .dataset import PoseDataset, TrajectoryDataset, localization_collate
from .evaluation.protocols import build_map_split, corruption_level
from .matching.grid import PoseGrid, map_raster
from .matching.learned import LearnedMatcher, LearnedMatcherConfig
from .tensor_generator import (
    TensorObservationGenerator,
    TensorPoseSampler,
    TensorTrajectorySampler,
)
from .temporal.rnn import ConvGRULocalizer, pose_nll


def _device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _make_generator(device: torch.device, seed: int) -> torch.Generator:
    gen = torch.Generator(device=device)
    gen.manual_seed(int(seed))
    return gen


def _target_indices(grid: PoseGrid, pos_px: torch.Tensor, yaw: torch.Tensor) -> torch.Tensor:
    """Flat volume indices of ground-truth poses (for train top-1 logging)."""
    xs = pos_px[:, 0].detach().cpu().numpy()
    ys = pos_px[:, 1].detach().cpu().numpy()
    yaws = yaw.detach().cpu().numpy().reshape(-1)
    idx = np.empty(len(xs), dtype=np.int64)
    for i in range(len(xs)):
        ix = int(np.clip(round(xs[i] / grid.resolution - 0.5), 0, grid.nx - 1))
        iy = int(np.clip(round(ys[i] / grid.resolution - 0.5), 0, grid.ny - 1))
        iyaw = int(round((yaws[i] + np.pi) / grid.dyaw)) % grid.n_yaw
        idx[i] = (iyaw * grid.ny + iy) * grid.nx + ix
    return torch.from_numpy(idx).to(pos_px.device)


def _pose_batches(args, device, split, level, tgen, seed):
    """Infinite batches of tensor-rendered single-frame observations."""
    sampler = TensorPoseSampler(split.train_region, device=device)
    gen = _make_generator(device, seed)
    while True:
        poses = sampler.sample(args.batch_size, gen)
        out = tgen.sample(
            poses,
            gen,
            geometry=level.geometry,
            photometric=level.photometric,
        )
        yield {
            "image": out["image"],
            "valid_mask": out["valid_mask"],
            "pos_px": out["gt_xy"],
            "yaw": out["yaw"],
        }


def _sequence_batches(args, device, split, level, tgen, seed):
    """Infinite batches of tensor-rendered smooth observation sequences."""
    sampler = TensorTrajectorySampler(
        split.train_region,
        seq_len=args.seq_len,
        step_stride=args.step_stride,
        device=device,
    )
    gen = _make_generator(device, seed)
    b, t = args.batch_size, args.seq_len
    while True:
        poses = sampler.sample(b, gen)  # B,T,3
        out = tgen.sample(
            poses.reshape(b * t, 3),
            gen,
            geometry=level.geometry,
            photometric=level.photometric,
        )
        yield {
            "images": out["image"].reshape(b, t, *out["image"].shape[1:]),
            "valid_masks": out["valid_mask"].reshape(b, t, *out["valid_mask"].shape[1:]),
            "pos_px": out["gt_xy"].reshape(b, t, 2),
            "yaw": out["yaw"].reshape(b, t),
        }


def train_observation(args) -> Path:
    device = _device(args.device)
    split = build_map_split(args.map, out_dir=args.map_variants_dir)
    map_ = split.train_map
    grid = PoseGrid.from_map(
        map_, resolution=args.resolution, n_yaw=args.n_yaw
    )
    level = corruption_level(args.corruption)
    # The observation renderer must use the same crop geometry as the model.
    cfg = copy.deepcopy(level.cfg)
    cfg.geometry = replace(
        cfg.geometry,
        obs_width=args.obs_size,
        obs_height=args.obs_size,
        fov_px=args.fov_px,
    )
    lcfg = LearnedMatcherConfig(
        fov_px=args.fov_px,
        obs_width=args.obs_size,
        obs_height=args.obs_size,
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

    length = max(args.steps * args.batch_size + 16, 64)
    tgen = None
    if args.generator == "tensor":
        tgen = TensorObservationGenerator(
            map_.image,
            cfg.geometry,
            cfg.photometric,
            device=device,
        )
        it = _pose_batches(args, device, split, level, tgen, args.seed)
    else:
        ds = PoseDataset(
            map_,
            cfg,
            length=length,
            seed=args.seed,
            split="train",
            region=split.train_region,
            geometry=level.geometry,
            photometric=level.photometric,
        )
        loader = DataLoader(
            ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=localization_collate,
            drop_last=True,
        )
        it = itertools.cycle(loader)
    data = map_raster(map_, grid, grayscale=False, normalize=True, device=device)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    running = None
    for step in range(1, args.steps + 1):
        batch = next(it)
        images = batch["image"].to(device)
        masks = batch["valid_mask"].to(device)
        pos_px = batch["pos_px"].to(device)
        yaw = batch["yaw"].to(device).reshape(-1)

        map_feats = matcher.encode_map(data["image"])
        out = matcher(images, map_feats, valid_mask=masks)
        nll = matcher.nll(out["logits"], pos_px, yaw)
        loss = nll + args.mask_weight * out.get(
            "mask_loss", torch.zeros((), device=device)
        )
        top1 = (
            out["logits"].flatten(1).argmax(1) == _target_indices(grid, pos_px, yaw)
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
        if step % args.save_every == 0:
            _save_ckpt(out_path, lcfg.to_dict(), matcher, grid, args, step)

    _save_ckpt(out_path, lcfg.to_dict(), matcher, grid, args, args.steps)
    print(f"saved {out_path}")
    return out_path


def _save_ckpt(path: Path, cfg: dict, model: torch.nn.Module, grid: PoseGrid, args, step: int):
    torch.save(
        {
            "config": cfg,
            "grid": {
                "resolution": grid.resolution,
                "n_yaw": grid.n_yaw,
            },
            "state_dict": model.state_dict(),
            "step": step,
            "args": vars(args),
        },
        path,
    )


def _load_learned(path: str | Path, grid: PoseGrid, device) -> LearnedMatcher:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    cfg = LearnedMatcherConfig.from_dict(ckpt["config"])
    model = LearnedMatcher(cfg, grid)
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def train_rnn(args) -> Path:
    device = _device(args.device)
    if not args.learned_checkpoint:
        raise SystemExit("--learned-checkpoint is required for --mode rnn")
    split = build_map_split(args.map, out_dir=args.map_variants_dir)
    map_ = split.train_map
    grid = PoseGrid.from_map(map_, resolution=args.resolution, n_yaw=args.n_yaw)
    matcher = _load_learned(args.learned_checkpoint, grid, device)
    data = map_raster(map_, grid, grayscale=False, normalize=True, device=device)
    with torch.no_grad():
        map_feats = matcher.encode_map(data["image"])

    level = corruption_level(args.corruption)
    cfg = copy.deepcopy(level.cfg)
    cfg.geometry = replace(
        cfg.geometry,
        obs_width=args.obs_size,
        obs_height=args.obs_size,
        fov_px=args.fov_px,
    )
    tgen = None
    if args.generator == "tensor":
        tgen = TensorObservationGenerator(
            map_.image,
            cfg.geometry,
            cfg.photometric,
            device=device,
        )
        it = _sequence_batches(args, device, split, level, tgen, args.seed)
    else:
        ds = TrajectoryDataset(
            map_,
            cfg,
            length=max(args.steps * args.batch_size + 16, 32),
            seq_len=args.seq_len,
            step_stride=args.step_stride,
            seed=args.seed,
            split="train",
            region=split.train_region,
            geometry=level.geometry,
            photometric=level.photometric,
        )
        loader = DataLoader(
            ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=localization_collate,
            drop_last=True,
        )
        it = itertools.cycle(loader)

    rcfg = {
        "feature_dim": matcher.cfg.feature_dim,
        "hidden_ch": args.hidden_ch,
        "context_ch": args.context_ch,
        "latent_yaw": args.latent_yaw,
        "latent_xy": args.latent_xy,
        "num_layers": args.rnn_layers,
        "spatial_residual": not args.no_spatial_residual,
        "yaw_residual": not args.no_yaw_residual,
    }
    model = ConvGRULocalizer(grid, **rcfg).to(device)
    if args.init_checkpoint:
        init = torch.load(args.init_checkpoint, map_location="cpu", weights_only=False)
        model.load_state_dict(init["state_dict"])
        print(f"initialised from {args.init_checkpoint}", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    for step in range(1, args.steps + 1):
        batch = next(it)
        images = batch["images"].to(device)  # B,T,3,H,W
        masks = batch["valid_masks"].to(device)
        pos_px = batch["pos_px"].to(device)  # B,T,2
        yaw = batch["yaw"].to(device)  # B,T,1
        b, t = images.shape[:2]
        flat_imgs = images.reshape(b * t, *images.shape[2:])
        flat_masks = masks.reshape(b * t, *masks.shape[2:])
        with torch.no_grad():
            outs = matcher(flat_imgs, map_feats, valid_mask=flat_masks)
            logits = outs["logits"].reshape(b, t, *outs["logits"].shape[1:])
            feats = outs["features"].reshape(b, t, *outs["features"].shape[1:])
        refined = model.forward_sequence(logits, feats)
        loss = pose_nll(refined, grid, pos_px.reshape(b * t, 2), yaw.reshape(b * t))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        opt.step()
        if step % args.log_every == 0 or step == 1:
            print(
                f"rnn step {step:5d}  nll {float(loss.detach()):8.4f}  {time.time()-t0:6.1f}s",
                flush=True,
            )
        if step % args.save_every == 0:
            torch.save(
                {
                    "config": rcfg,
                    "grid": {"resolution": grid.resolution, "n_yaw": grid.n_yaw},
                    "learned_checkpoint": str(args.learned_checkpoint),
                    "state_dict": model.state_dict(),
                    "step": step,
                },
                out_path,
            )
    torch.save(
        {
            "config": rcfg,
            "grid": {"resolution": grid.resolution, "n_yaw": grid.n_yaw},
            "learned_checkpoint": str(args.learned_checkpoint),
            "state_dict": model.state_dict(),
            "step": args.steps,
        },
        out_path,
    )
    print(f"saved {out_path}")
    return out_path


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=["observation", "rnn"], required=True)
    p.add_argument("--generator", choices=["tensor", "dataset"], default="tensor",
                   help="tensor = batched on-device rendering (online training)")
    p.add_argument("--map", default="res/example.png")
    p.add_argument("--map-variants-dir", default="outputs/maps")
    p.add_argument("--out", required=True)
    p.add_argument("--learned-checkpoint", default=None)
    p.add_argument("--init-checkpoint", default=None,
                   help="initialise the observation model from a checkpoint")
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--seq-len", type=int, default=6)
    p.add_argument("--step-stride", type=int, default=6)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--corruption", default="moderate")
    p.add_argument("--resolution", type=float, default=8.0)
    p.add_argument("--n-yaw", type=int, default=64)
    p.add_argument("--fov-px", type=float, default=320.0)
    p.add_argument("--obs-size", type=int, default=256)
    p.add_argument("--feature-dim", type=int, default=32)
    p.add_argument("--base-channels", type=int, default=32)
    p.add_argument("--obs-stride", type=int, default=4)
    p.add_argument("--mask-weight", type=float, default=0.25)
    p.add_argument("--grad-clip", type=float, default=5.0)
    p.add_argument("--hidden-ch", type=int, default=32)
    p.add_argument("--context-ch", type=int, default=4)
    p.add_argument("--latent-yaw", type=int, default=8)
    p.add_argument("--latent-xy", type=int, default=24)
    p.add_argument("--rnn-layers", type=int, default=1)
    p.add_argument("--no-spatial-residual", action="store_true",
                   help="disable the coarse spatial residual (unstable at eval time)")
    p.add_argument("--no-yaw-residual", action="store_true",
                   help="disable the heading residual")
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--save-every", type=int, default=100)
    p.add_argument("--device", default="auto")
    return p


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    if args.mode == "observation":
        train_observation(args)
    else:
        train_rnn(args)


if __name__ == "__main__":
    main()