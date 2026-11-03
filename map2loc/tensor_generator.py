"""Batched perspective renderer + BEV rectification for training/evaluation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from data_gen.tensor_generator import TensorObservationGenerator

from .bev import BEVSpec, SCALE_HYPOTHESES, rectify_torch


class BEVTensorGenerator(TensorObservationGenerator):
    """``TensorObservationGenerator`` plus inverse-perspective rectification.

    The base generator renders the oblique camera view, the heavy photometric
    corruption and the validity mask.  This subclass additionally rectifies
    every view into a map-aligned BEV patch using the *measured* attitude
    (yaw / pitch / roll), which is the input the matchers consume.
    """

    def __init__(self, *args, bev_spec: BEVSpec | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.bev_spec = bev_spec or BEVSpec()

    def sample_bev(
        self,
        poses: torch.Tensor,
        generator: torch.Generator | None = None,
        *,
        geometry: bool = True,
        photometric: bool = True,
        rect_scales: torch.Tensor | None = None,
        scale_jitter: float = 0.0,
        measurement=None,
    ) -> dict:
        """Render views and rectify them into one or more BEV patches.

        ``rect_scales`` selects the focal scale used for rectification.  When
        omitted, a single scale is drawn around the true zoom (training mode);
        pass an explicit ``(B,)`` or scalar tensor to rectify at fixed
        hypotheses (evaluation mode).
        """
        base = super().sample(
            poses,
            generator,
            geometry=geometry,
            photometric=photometric,
            measurement=measurement,
        )
        b = base["image"].shape[0]
        dev = self.device
        dtype = base["image"].dtype

        img01 = base["image"] * 0.5 + 0.5
        if rect_scales is None:
            if scale_jitter > 0:
                jitter = torch.exp(
                    (2.0 * torch.rand(b, device=dev, generator=generator) - 1.0)
                    * float(scale_jitter)
                )
            else:
                jitter = torch.ones(b, device=dev)
            rect = base["scale"] * jitter
        else:
            rect = torch.as_tensor(rect_scales, device=dev, dtype=dtype)
            if rect.ndim == 0:
                rect = rect.expand(b)
            if rect.shape[0] == 1:
                rect = rect.expand(b)

        patch, valid = rectify_torch(
            img01,
            base["valid_mask"],
            base["yaw_input"],
            base["pitch_input"],
            base["roll_input"],
            self.geometry,
            rect,
            self.bev_spec,
        )
        out = dict(base)
        out.update(
            {
                "bev": (patch.clamp(0, 1) - 0.5) / 0.5,
                "bev_valid": valid,
                "rect_scale": rect,
            }
        )
        return out

    @torch.no_grad()
    def sample_bev_hypotheses(
        self,
        poses: torch.Tensor,
        generator: torch.Generator | None = None,
        *,
        geometry: bool = True,
        photometric: bool = True,
        scales: tuple[float, ...] = SCALE_HYPOTHESES,
    ) -> dict:
        """Rectify each view at every focal-scale hypothesis.

        Returns ``bev`` / ``bev_valid`` with shape ``(B, S, 3, P, P)`` /
        ``(B, S, P, P)`` and ``scales`` ``(S,)``.
        """
        base = super().sample(
            poses,
            generator,
            geometry=geometry,
            photometric=photometric,
        )
        b = base["image"].shape[0]
        dev = self.device
        img01 = base["image"] * 0.5 + 0.5
        patches, masks = [], []
        for s in scales:
            p, v = rectify_torch(
                img01,
                base["valid_mask"],
                base["yaw_input"],
                base["pitch_input"],
                base["roll_input"],
                self.geometry,
                torch.full((b,), float(s), device=dev),
                self.bev_spec,
            )
            patches.append(p)
            masks.append(v)
        out = dict(base)
        out["bev"] = torch.stack(patches, dim=1)
        out["bev_valid"] = torch.stack(masks, dim=1)
        out["scales"] = torch.tensor(scales, device=dev)
        return out