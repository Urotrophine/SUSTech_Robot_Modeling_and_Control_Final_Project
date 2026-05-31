# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Sequence
import numpy as np
import mujoco


class JointSpaceController:
    """Generic position controller wrapper.

    This class is intentionally independent of the arm/gripper meaning.
    It only maps ordered joint targets to ordered MuJoCo actuators.
    """

    def __init__(self, model, data, joint_names: Sequence[str], actuator_names: Sequence[str] | None = None):
        self.model = model
        self.data = data
        self.joint_names = list(joint_names)

        if actuator_names is None:
            actuator_names = [self._find_actuator_for_joint(j) for j in self.joint_names]
        self.actuator_names = list(actuator_names)

        self.joint_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, j) for j in self.joint_names]
        self.actuator_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, a) for a in self.actuator_names]

        for j, jid in zip(self.joint_names, self.joint_ids):
            if jid < 0:
                raise ValueError(f"Joint not found: {j}")
        for a, aid in zip(self.actuator_names, self.actuator_ids):
            if aid < 0:
                raise ValueError(f"Actuator not found: {a}")

    def _find_actuator_for_joint(self, joint_name: str) -> str:
        candidates = [f"{joint_name}_pos", joint_name, f"{joint_name}_motor"]
        if joint_name == "joint71":
            candidates.insert(0, "gripper_opening_left")
        elif joint_name == "joint72":
            candidates.insert(0, "gripper_opening_right")
        for name in candidates:
            aid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            if aid >= 0:
                return name
        raise ValueError(f"No actuator found for joint: {joint_name}")

    def get_q(self) -> np.ndarray:
        q = []
        for jid in self.joint_ids:
            q.append(self.data.qpos[self.model.jnt_qposadr[jid]])
        return np.array(q, dtype=float)

    def get_qvel(self) -> np.ndarray:
        qv = []
        for jid in self.joint_ids:
            qv.append(self.data.qvel[self.model.jnt_dofadr[jid]])
        return np.array(qv, dtype=float)

    def limits(self):
        low = []
        high = []
        for aid in self.actuator_ids:
            lo, hi = self.model.actuator_ctrlrange[aid]
            low.append(lo)
            high.append(hi)
        return np.array(low, dtype=float), np.array(high, dtype=float)

    def set_target(self, q_des: Sequence[float], clamp: bool = True) -> np.ndarray:
        q_des = np.asarray(q_des, dtype=float)
        if clamp:
            lo, hi = self.limits()
            q_des = np.minimum(np.maximum(q_des, lo), hi)
        for value, aid in zip(q_des, self.actuator_ids):
            self.data.ctrl[aid] = value
        return q_des
