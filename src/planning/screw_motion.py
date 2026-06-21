# -*- coding: utf-8 -*-
from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class ScrewMotionCommand:
    insertion_depth: float
    wiggle_offset: np.ndarray
    screw_angle: float
    progress: float


class ScrewMotion:
    """Insertion command generator with small lateral wiggle.

    The returned insertion depth is positive downward. The task layer maps it
    to world z by subtracting it from the hole-entry height.
    """

    def __init__(
        self,
        push_depth: float = 0.03,
        screw_amplitude: float = 0.2,
        duration: float = 5.0,
        wiggle_amplitude: float = 0.002,
        screw_turns: float = 3.0,
    ):
        self.push_depth = max(0.0, float(push_depth))
        self.screw_amplitude = float(screw_amplitude)
        self.duration = max(1e-9, float(duration))
        self.wiggle_amplitude = max(0.0, float(wiggle_amplitude))
        self.screw_turns = float(screw_turns)

    def sample(self, t: float):
        cmd = self.sample_command(t)
        return cmd.insertion_depth, cmd.screw_angle

    def sample_command(self, t: float) -> ScrewMotionCommand:
        u = float(np.clip(float(t) / self.duration, 0.0, 1.0))
        depth = self.push_depth * u
        phase = 2.0 * np.pi * u
        screw = 2.0 * np.pi * self.screw_turns * u + self.screw_amplitude * np.sin(phase)
        wiggle = self.wiggle_amplitude * np.array([np.cos(phase), np.sin(phase)], dtype=float)
        return ScrewMotionCommand(
            insertion_depth=float(depth),
            wiggle_offset=wiggle,
            screw_angle=float(screw),
            progress=u,
        )
