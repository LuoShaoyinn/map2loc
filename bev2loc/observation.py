"""Cut observation windows out of the fixed map and corrupt them."""

from __future__ import annotations

import cv2
import numpy as np

from .augment import build_geometry_matrix, corrupt, sample_augmentation
from .config import GeometryConfig, PhotometricConfig
from .map import FixedMap
from .types import Augmentation, Observation, Pose


def _apply_h(h: np.ndarray, pts: np.ndarray) -> np.ndarray:
    pts = np.asarray(pts, dtype=np.float64)
    ones = np.ones((pts.shape[0], 1), dtype=np.float64)
    hom = np.concatenate([pts, ones], axis=1) @ h.T
    return (hom[:, :2] / hom[:, 2:3]).astype(np.float32)


class ObservationGenerator:
    """Renders partial, noisy views of the known map.

    The nominal observation is a rotated crop centred on the camera pose.
    The geometric corruption homography ``A`` is composed on the observation
    side, so the exact target is always ``H_total @ centre`` in map pixels.
    """

    def __init__(
        self,
        map_: FixedMap,
        geometry: GeometryConfig | None = None,
        photometric: PhotometricConfig | None = None,
    ):
        self.map = map_
        self.geometry = geometry or GeometryConfig()
        self.photometric = photometric or PhotometricConfig()
        self.width = self.geometry.obs_width
        self.height = self.geometry.obs_height
        self._border_value = tuple(int(round(v)) for v in map_.background.tolist())

    # ------------------------------------------------------------------ warp
    def view_to_map(self, pose: Pose) -> np.ndarray:
        """Homography mapping the nominal (uncorrupted) observation to the map."""
        w = float(self.width)
        s = w / float(self.geometry.fov_px)  # observation px per map px
        cx = (self.width - 1) / 2.0
        cy = (self.height - 1) / 2.0

        c_yaw, s_yaw = np.cos(pose.yaw), np.sin(pose.yaw)
        # observation +u (right) and -v (up) in map coordinates
        rx, ry = -s_yaw, c_yaw
        h = np.array(
            [
                [rx / s, -c_yaw / s, pose.x - (rx * cx - c_yaw * cy) / s],
                [ry / s, -s_yaw / s, pose.y - (ry * cx - s_yaw * cy) / s],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        return h

    def _warp(
        self,
        h_total: np.ndarray,
    ) -> np.ndarray:
        border = self.geometry.border
        if border == "background":
            mode = cv2.BORDER_CONSTANT
            value = self._border_value
        elif border == "zero":
            mode = cv2.BORDER_CONSTANT
            value = (0, 0, 0)
        elif border == "reflect":
            mode = cv2.BORDER_REFLECT_101
            value = self._border_value
        else:
            raise ValueError(f"unknown border mode: {border!r}")

        return cv2.warpPerspective(
            self.map.image,
            h_total,
            (self.width, self.height),
            flags=cv2.INTER_LINEAR,
            borderMode=mode,
            borderValue=value,
        )

    # -------------------------------------------------------------- generate
    def generate(
        self,
        pose: Pose,
        rng: np.random.Generator | None = None,
        *,
        geometry: bool = True,
        photometric: bool = True,
        aug: Augmentation | None = None,
        h_geom: np.ndarray | None = None,
    ) -> Observation:
        """Render one observation for ``pose``.

        Parameters
        ----------
        geometry:
            Apply the random scale/shear/perspective corruption.
        photometric:
            Apply blur/illumination/noise/scratch/JPEG corruption.
        aug, h_geom:
            Optional pre-sampled geometry (useful for replay / ablations).
        """
        rng = rng or np.random.default_rng()
        if h_geom is None:
            if aug is None:
                if geometry:
                    aug, h_geom = sample_augmentation(
                        rng, self.geometry, self.width, self.height
                    )
                else:
                    aug = Augmentation()
                    h_geom = np.eye(3, dtype=np.float64)
            else:
                h_geom = (
                    build_geometry_matrix(aug, self.width, self.height)
                    if geometry
                    else np.eye(3, dtype=np.float64)
                )
        elif aug is None:
            aug = Augmentation()

        h_total = self.view_to_map(pose) @ h_geom
        image = self._warp(h_total)

        photo: dict = {"enabled": False}
        if photometric:
            image, photo = corrupt(
                image,
                rng,
                self.photometric,
                background=self.map.background,
            )

        corners = _apply_h(
            h_total,
            np.array(
                [
                    [0, 0],
                    [self.width - 1, 0],
                    [self.width - 1, self.height - 1],
                    [0, self.height - 1],
                ],
                dtype=np.float32,
            ),
        )
        gt_xy = _apply_h(
            h_total,
            np.array([[(self.width - 1) / 2.0, (self.height - 1) / 2.0]]),
        )[0]

        return Observation(
            image=image,
            pose=pose,
            gt_xy=gt_xy.astype(np.float32),
            corners=corners,
            warp=h_total,
            aug=aug,
            photo=photo,
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
        return {
            "pos": pos.astype(np.float32),
            "pos_px": obs.gt_xy.astype(np.float32),
            "yaw_sincos": np.array(
                [np.sin(obs.pose.yaw), np.cos(obs.pose.yaw)], dtype=np.float32
            ),
            "yaw": np.array([obs.pose.yaw], dtype=np.float32),
        }