"""Causal position belief with image-derived world-frame displacement.

Transitions operate on probability mass with fractional translations and zero
padding (no wrapping at map edges). Observation inputs are log likelihoods, not
positions. No ground-truth initializer, path, simulator velocity or future frame
is used. Each route pass starts from a uniform belief.
"""
from dataclasses import dataclass
import cv2
import numpy as np


@dataclass(frozen=True)
class VisualFilterConfig:
    sigma_motion_m: float = .25
    sigma_missing_m: float = 1.0
    observation_weight: float = .5
    global_mix: float = .01
    history_power: float = 1.0
    carry_motion: bool = False


class VisualPositionFilter:
    def __init__(self, grid_xy, config):
        self.xy = np.asarray(grid_xy, dtype=np.float64)
        self.nx = len(np.unique(self.xy[:, 0]))
        self.ny = len(np.unique(self.xy[:, 1]))
        self.cell_m = float(np.unique(self.xy[:, 0])[1] - self.xy[0, 0]) / 100
        self.config = config
        self.reset()

    def reset(self):
        self.probability = np.full((self.ny, self.nx), 1 / (self.nx * self.ny))
        self.initialized = False
        self.last_delta = np.zeros(2)
        self.missing_count = 0

    def update(self, log_likelihood, motion):
        cfg = self.config
        prior = self.probability
        if self.initialized:
            accepted = bool(motion['accepted'])
            if accepted:
                delta = np.asarray(motion['delta_world_m'], dtype=float)
                self.last_delta = delta
                self.missing_count = 0
            else:
                self.missing_count += 1
                # Optional short extrapolation from the last visual increment,
                # never from simulator velocity or differences of true poses.
                delta = self.last_delta * .5**self.missing_count if cfg.carry_motion else np.zeros(2)
            sigma = cfg.sigma_motion_m if accepted else cfg.sigma_missing_m
            # Map pixel y is opposite to world y.
            shift = delta * np.array([1., -1.]) / self.cell_m
            prior = cv2.warpAffine(prior, np.array([[1., 0, shift[0]], [0, 1., shift[1]]]),
                                   (self.nx, self.ny), borderMode=cv2.BORDER_CONSTANT)
            prior = cv2.GaussianBlur(prior, (0, 0), max(sigma / self.cell_m, .25),
                                     borderType=cv2.BORDER_CONSTANT)
            prior = np.maximum(prior, 1e-30)**cfg.history_power
            prior /= prior.sum()
            prior = (1-cfg.global_mix)*prior + cfg.global_mix/prior.size
        ll = np.asarray(log_likelihood, dtype=np.float64).reshape(self.ny, self.nx)
        ll = cfg.observation_weight*(ll-ll.max())
        posterior = prior*np.exp(np.maximum(ll, -80))
        self.probability = posterior/posterior.sum()
        self.initialized = True
        mean = self.probability.ravel() @ self.xy
        variance_m2 = float(self.probability.ravel() @ (((self.xy-mean)/100)**2).sum(1))
        return mean, variance_m2
