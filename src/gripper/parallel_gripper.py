# -*- coding: utf-8 -*-
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class ParallelGripper:
    """High-level gripper interface.

    The rest of the project should call open(), close(), or set_opening().
    Nobody outside this module should directly write the gripper master/mimic
    joints. In v7 these are joint71 / joint72 with symmetric targets.
    """

    controller: object
    open_command: float = -0.05
    close_command: float = 0.004
    mimic_multiplier: Optional[float] = -1.0
    mimic_offset: float = 0.0

    def set_opening(self, value: float):
        if self.mimic_multiplier is None or len(self.controller.joint_names) == 1:
            return self.controller.set_target([value])
        mimic_value = self.mimic_multiplier * float(value) + self.mimic_offset
        return self.controller.set_target([value, mimic_value])

    def open(self):
        return self.set_opening(self.open_command)

    def close(self):
        return self.set_opening(self.close_command)

    def get_opening(self) -> float:
        return float(self.controller.get_q()[0])
