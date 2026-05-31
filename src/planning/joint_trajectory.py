# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Literal, Sequence
import numpy as np

TrajectoryType = Literal["linear", "cubic", "quintic"]


class JointTrajectory:
    def __init__(self, q_start: Sequence[float], q_goal: Sequence[float], duration: float, method: TrajectoryType = "quintic"):
        self.q_start = np.asarray(q_start, dtype=float)
        self.q_goal = np.asarray(q_goal, dtype=float)
        self.duration = float(duration)
        self.method = method

    def sample(self, t: float):
        u = np.clip(t / self.duration, 0.0, 1.0)
        if self.method == "linear":
            s = u
        elif self.method == "cubic":
            s = 3*u*u - 2*u*u*u
        elif self.method == "quintic":
            s = 10*u**3 - 15*u**4 + 6*u**5
        else:
            raise ValueError(f"Unknown trajectory method: {self.method}")
        return self.q_start + s * (self.q_goal - self.q_start)
