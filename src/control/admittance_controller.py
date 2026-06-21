# -*- coding: utf-8 -*-
"""
src/control/admittance_controller.py

Joint4 admittance force controller.

This is an outer-loop force controller for a position-actuated MuJoCo robot.
It does not output motor torque. It converts force error into a joint4 position
command correction.

Control idea:

    force_error = F_des - F_meas

If force is too small, joint4 moves further downward.
If force is too large, joint4 retreats upward.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class AdmittanceState:
    command: float
    force_filtered: float
    velocity_command: float
    force_error: float
    saturated_low: bool
    saturated_high: bool


class Joint4AdmittanceController:
    def __init__(
        self,
        initial_command: float,
        down_sign: float,
        command_min: float,
        command_max: float,
        target_force: float = 5.0,
        k_force: float = 0.004,
        max_speed: float = 0.015,
        filter_alpha: float = 0.15,
        deadband: float = 0.10,
    ) -> None:
        self.command_min = float(command_min)
        self.command_max = float(command_max)
        self.command = float(np.clip(initial_command, self.command_min, self.command_max))

        self.down_sign = 1.0 if down_sign >= 0.0 else -1.0
        self.target_force = float(target_force)
        self.k_force = float(k_force)
        self.max_speed = abs(float(max_speed))
        self.filter_alpha = float(np.clip(filter_alpha, 0.0, 1.0))
        self.deadband = abs(float(deadband))

        self.force_filtered = 0.0
        self.velocity_command = 0.0
        self.force_error = 0.0

    def reset(self, command: float | None = None, measured_force: float = 0.0) -> None:
        if command is not None:
            self.command = float(np.clip(command, self.command_min, self.command_max))
        self.force_filtered = max(0.0, float(measured_force))
        self.velocity_command = 0.0
        self.force_error = self.target_force - self.force_filtered

    def update(self, measured_force: float, dt: float) -> AdmittanceState:
        measured_force = max(0.0, float(measured_force))
        dt = max(float(dt), 1e-9)

        self.force_filtered = (
            (1.0 - self.filter_alpha) * self.force_filtered
            + self.filter_alpha * measured_force
        )

        self.force_error = self.target_force - self.force_filtered

        if abs(self.force_error) <= self.deadband:
            speed_along_down_direction = 0.0
        else:
            speed_along_down_direction = self.k_force * self.force_error
            speed_along_down_direction = float(
                np.clip(speed_along_down_direction, -self.max_speed, self.max_speed)
            )

        self.velocity_command = self.down_sign * speed_along_down_direction
        self.command += self.velocity_command * dt

        saturated_low = self.command <= self.command_min
        saturated_high = self.command >= self.command_max
        self.command = float(np.clip(self.command, self.command_min, self.command_max))

        return AdmittanceState(
            command=self.command,
            force_filtered=self.force_filtered,
            velocity_command=self.velocity_command,
            force_error=self.force_error,
            saturated_low=bool(saturated_low),
            saturated_high=bool(saturated_high),
        )
