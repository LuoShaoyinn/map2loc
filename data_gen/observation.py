"""Render camera views of the fixed map and corrupt them.

An observation is produced by a real pinhole camera at a fixed height above
the ground-plane map.  The camera pose has smooth ``x, y, yaw, pitch, roll``
curves; small head/gimbal rotations and a random zoom are applied per frame.
The view is clipped to a fixed-size image, the map region actually covered by
the view is stored as a validity mask, and the noisy attitude measurements
(yaw / pitch) are stored alongside the exact ground truth.
"""

from __future__ import annotations

import numpy as np

from .augment import corrupt, sample_augmentation
from .camera import (
    _apply_h,
    apply_homography,
    render_validity,
    render_view,
    view_border_corners,
    view_to_map,
    wrap_angle,
)
from .config import GeometryConfig, MeasurementConfig, PhotometricConfig
from .map import FixedMap
from .types import Augmentation, Observation, Pose


class ObservationGenerator:
    """Renders partial, noisy pinhole views of the known map.

    Parameters
    ----------
    map_:
        Fixed map image.
    geometry:
        Camera intrinsics / fixed height and augmentation ranges.
    photometric:
        Gaussian noise + random invalid-pixel model.
    measurement:
        Standard deviations of the noisy yaw / pitch inputs.
    """

    def __init__(
        self,
        map_: FixedMap,
        geometry: GeometryConfig | None = None,
        photometric: PhotometricConfig | None = None,
        measurement: MeasurementConfig | None = None,
    ):
        self.map = map_
        self.geometry = geometry or GeometryConfig()
        self.photometric = photometric or PhotometricConfig()
        self.measurement = measurement or MeasurementConfig()
        self.width = self.geometry.obs_width
        self.height = self.geometry.obs_height

    # --------------------------------------------------------------- geometry
    def view_to_map(self, pose: Pose, scale: float = 1.0) -> np.ndarray:
        """Homography mapping view pixels to map pixels."""
        return view_to_map(pose, self.geometry, scale)

    def map_to_view(self, pose: Pose, scale: float = 1.0) -> np.ndarray:
        """Homography mapping map pixels to view pixels."""
        return np.linalg.inv(self.view_to_map(pose, scale))

    def effective_pose(self, pose: Pose, aug: Augmentation) -> Pose:
        """Apply the sampled head/gimbal rotation to the smooth pose."""
        return Pose(
            x=float(pose.x),
            y=float(pose.y),
            yaw=wrap_angle(float(pose.yaw) + aug.head_yaw),
            pitch=float(pose.pitch) + aug.head_pitch,
            roll=float(pose.roll) + aug.head_roll,
            height=float(pose.height),
        )

    def _measure(self, pose: Pose, rng: np.random.Generator) -> tuple[float, float]:
        yaw_sigma = np.deg2rad(self.measurement.yaw_sigma_deg)
        pitch_sigma = np.deg2rad(self.measurement.pitch_sigma_deg)
        yaw = wrap_angle(pose.yaw + float(rng.normal(0.0, yaw_sigma)))
        pitch = pose.pitch + float(rng.normal(0.0, pitch_sigma))
        return yaw, pitch

    # -------------------------------------------------------------- generate
    def generate(
        self,
        pose: Pose,
        rng: np.random.Generator | None = None,
        *,
        geometry: bool = True,
        photometric: bool = True,
        aug: Augmentation | None = None,
    ) -> Observation:
        """Render one camera view for ``pose``.

        Parameters
        ----------
        geometry:
            Sample the random zoom and head rotations.  Set false for an
            unaugmented (but still pitched/rolled) render.
        photometric:
            Add Gaussian noise and random invalid pixels.
        aug:
            Optional pre-sampled augmentation (useful for replay / ablations).
        """
        rng = rng or np.random.default_rng()
        if aug is None:
            aug = sample_augmentation(rng, self.geometry, enabled=geometry)
        elif not geometry:
            aug = Augmentation()

        eff = self.effective_pose(pose, aug)
        h_view_to_map = self.view_to_map(eff, aug.scale)

        image = render_view(
            self.map.image,
            h_view_to_map,
            self.geometry,
            self.map.background,
        )

        # Which view pixels actually sample the known map?  Border padding is
        # not real content even when it is colour-matched.
        map_valid = render_validity(self.map.image, h_view_to_map, self.geometry)
        valid_mask = map_valid

        photo: dict = {"enabled": False}
        if photometric:
            image, photo, noise_valid = corrupt(
                image,
                rng,
                self.photometric,
                background=self.map.background,
                return_mask=True,
            )
            valid_mask = map_valid * noise_valid
        photo["map_valid_frac"] = float(map_valid.mean())
        photo["valid_frac"] = float(valid_mask.mean())

        corners = apply_homography(h_view_to_map, view_border_corners(self.geometry))
        limit = 4.0 * max(self.map.width, self.map.height)
        corners = np.nan_to_num(
            corners, nan=0.0, posinf=limit, neginf=-limit
        )
        corners = np.clip(corners, -limit, limit).astype(np.float32)
        cx = (self.width - 1) / 2.0
        cy = (self.height - 1) / 2.0
        centre_depth = float(
            h_view_to_map[2, 0] * cx
            + h_view_to_map[2, 1] * cy
            + h_view_to_map[2, 2]
        )
        if centre_depth > 0.0:
            center_xy = apply_homography(
                h_view_to_map, np.array([[cx, cy]], dtype=np.float32)
            )[0]
        else:
            # The image centre points at/above the horizon: no ground point.
            center_xy = eff.xy.astype(np.float32)

        yaw_input, pitch_input = self._measure(eff, rng)

        return Observation(
            image=image,
            pose=eff,
            gt_xy=eff.xy.astype(np.float32),
            center_xy=center_xy.astype(np.float32),
            corners=corners,
            warp=h_view_to_map,
            yaw_input=yaw_input,
            pitch_input=pitch_input,
            aug=aug,
            photo=photo,
            valid_mask=valid_mask.astype(np.float32),
        )

    def generate_many(
        self,
        poses,
        rng: np.random.Generator | None = None,
        *,
        geometry: bool = True,
        photometric: bool = True,
    ) -> list[Observation]:
        rng = rng or np.random.default_rng()
        return [
            self.generate(
                p, rng, geometry=geometry, photometric=photometric
            )
            for p in poses
        ]

    # --------------------------------------------------------------- targets
    def normalize_position(self, xy: np.ndarray) -> np.ndarray:
        return self.map.normalize_xy(xy)

    def target_dict(self, obs: Observation) -> dict[str, np.ndarray]:
        """Network targets for a single observation."""
        pos = self.normalize_position(obs.gt_xy)
        yaw = obs.pose.yaw
        return {
            "pos": pos.astype(np.float32),
            "pos_px": obs.gt_xy.astype(np.float32),
            "yaw": np.array([yaw], dtype=np.float32),
            "yaw_sincos": np.array([np.sin(yaw), np.cos(yaw)], dtype=np.float32),
            "pitch": np.array([obs.pose.pitch], dtype=np.float32),
            "roll": np.array([obs.pose.roll], dtype=np.float32),
            "yaw_input": np.array([obs.yaw_input], dtype=np.float32),
            "pitch_input": np.array([obs.pitch_input], dtype=np.float32),
        }