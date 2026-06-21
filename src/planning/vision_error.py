# -*- coding: utf-8 -*-
from __future__ import annotations

import numpy as np


def sample_random_xy_offset(radius: float, rng: np.random.Generator | None = None) -> np.ndarray:
    """Sample a uniform random XY offset inside a disk."""
    radius = max(0.0, float(radius))
    generator = rng if rng is not None else np.random.default_rng()
    distance = radius * float(np.sqrt(generator.random()))
    angle = 2.0 * np.pi * float(generator.random())
    return np.array([distance * np.cos(angle), distance * np.sin(angle)], dtype=float)
