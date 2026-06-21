# -*- coding: utf-8 -*-
from __future__ import annotations

import numpy as np


class ArchimedesSpiralSearch:
    """2D Archimedes spiral in a local XY search plane.

    The spiral is defined as:

        theta = angular_speed * t
        r = pitch * theta / (2*pi)

    This module is independent of MuJoCo. The task layer decides how to map the
    local XY offset to a hole frame or a world-frame command.
    """

    def __init__(self, radius_max: float = 0.03, pitch: float = 0.003, angular_speed: float = 2.0):
        self.radius_max = max(0.0, float(radius_max))
        self.pitch = max(0.0, float(pitch))
        self.angular_speed = max(0.0, float(angular_speed))

    def angle(self, t: float) -> float:
        return self.angular_speed * max(0.0, float(t))

    def radius(self, t: float) -> float:
        theta = self.angle(t)
        r = self.pitch * theta / (2.0 * np.pi)
        return float(min(r, self.radius_max))

    def sample_offset(self, t: float) -> np.ndarray:
        theta = self.angle(t)
        r = self.radius(t)
        return np.array([r * np.cos(theta), r * np.sin(theta), 0.0], dtype=float)

    def sample_xy(self, t: float) -> np.ndarray:
        return self.sample_offset(t)[:2]
