# -*- coding: utf-8 -*-
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence
import numpy as np
import mujoco


@dataclass
class JointRegistry:
    model: mujoco.MjModel
    joint_names: list[str]

    @classmethod
    def from_names(cls, model: mujoco.MjModel, joint_names: Sequence[str]):
        return cls(model=model, joint_names=list(joint_names))

    def joint_ids(self) -> list[int]:
        ids = []
        for name in self.joint_names:
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if jid < 0:
                raise ValueError(f"Joint not found: {name}")
            ids.append(jid)
        return ids

    def qpos_indices(self) -> np.ndarray:
        return np.array([self.model.jnt_qposadr[jid] for jid in self.joint_ids()], dtype=int)

    def qvel_indices(self) -> np.ndarray:
        return np.array([self.model.jnt_dofadr[jid] for jid in self.joint_ids()], dtype=int)
