# -*- coding: utf-8 -*-
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence
import numpy as np
import mujoco


@dataclass
class TorqueCommand:
    tau: np.ndarray


class ComputedTorqueController:
    """Future dynamics-control placeholder.

    Current project uses MuJoCo position actuators for easy platform testing.

    When the project moves to torque-level control, the recommended workflow is:

        1. Replace position actuators in MJCF with motor actuators.
        2. Use this module to compute desired generalized torques.
        3. Write torque commands to data.ctrl.

    MuJoCo already computes system dynamics internally. This class provides a
    place for controller logic such as:
        - gravity compensation,
        - inverse dynamics feedforward,
        - computed torque control,
        - impedance control,
        - admittance control.
    """

    def __init__(self, model, data, arm_joint_names: Sequence[str]):
        self.model = model
        self.data = data
        self.arm_joint_names = list(arm_joint_names)
        self.joint_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, j) for j in self.arm_joint_names]
        self.qpos_idx = np.array([model.jnt_qposadr[jid] for jid in self.joint_ids], dtype=int)
        self.qvel_idx = np.array([model.jnt_dofadr[jid] for jid in self.joint_ids], dtype=int)

    def gravity_compensation(self) -> TorqueCommand:
        """Return gravity compensation torque for controlled joints.

        This uses MuJoCo bias force qfrc_bias. For static poses, qfrc_bias
        includes gravity-related generalized forces.
        """
        mujoco.mj_forward(self.model, self.data)
        tau = self.data.qfrc_bias[self.qvel_idx].copy()
        return TorqueCommand(tau=tau)

    def inverse_dynamics(self, qacc_des: Sequence[float]) -> TorqueCommand:
        """Compute inverse dynamics torque for desired joint acceleration.

        This is a template. For full use, set qpos, qvel and qacc consistently,
        call mj_inverse, then read data.qfrc_inverse.
        """
        qacc_des = np.asarray(qacc_des, dtype=float).reshape(len(self.arm_joint_names))
        old_qacc = self.data.qacc.copy()
        self.data.qacc[:] = 0.0
        self.data.qacc[self.qvel_idx] = qacc_des
        mujoco.mj_inverse(self.model, self.data)
        tau = self.data.qfrc_inverse[self.qvel_idx].copy()
        self.data.qacc[:] = old_qacc
        return TorqueCommand(tau=tau)
