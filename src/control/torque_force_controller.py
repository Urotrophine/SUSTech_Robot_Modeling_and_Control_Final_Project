# -*- coding: utf-8 -*-
"""
src/control/torque_force_controller.py

Torque-level hybrid position-force controller for the MuJoCo arm.

Stage 3 v2 controller.

Compared with v1:
    v1 used only a proportional force term:
        F_task = Kf * (F_des - F_meas)

    In a torque-level controller with posture PD, a pure proportional force term
    may settle below the target force because the joint-space PD stiffness and
    the environment contact stiffness reach a static equilibrium.

    v2 adds an integral force term:
        F_task = Kf * e_F + Ki * integral(e_F dt)

This gives the force loop enough authority to remove steady-state force error.

Control law:
    tau = qfrc_bias + Kp(q_ref - q) - Kd*dq + J(q)^T F_task
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence
import numpy as np
import mujoco


@dataclass
class TorqueControlState:
    tau: np.ndarray
    tau_bias: np.ndarray
    tau_pd: np.ndarray
    tau_force: np.ndarray
    task_force_cmd: np.ndarray
    force_error: float
    force_filtered: float
    force_integral: float
    contact_force: float
    saturated: bool
    task_force_saturated: bool


class TorqueHybridForceController:
    """Torque-level hybrid position-force controller.

    The arm uses motor actuators. The gripper may remain a position actuator.

    Parameters
    ----------
    force_gain:
        Proportional force-loop gain.
    force_integral_gain:
        Integral force-loop gain. This is important for removing steady-state
        force error in torque-level contact control.
    force_integral_limit:
        Anti-windup limit for the force-error integral.
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        arm_joint_names: Sequence[str] = ("joint1", "joint2", "joint3", "joint4"),
        arm_motor_names: Sequence[str] = ("joint1_motor", "joint2_motor", "joint3_motor", "joint4_motor"),
        site_name: str = "ee_site",
        kp: Sequence[float] = (120.0, 120.0, 80.0, 1200.0),
        kd: Sequence[float] = (18.0, 18.0, 12.0, 90.0),
        torque_limits: Sequence[float] = (80.0, 80.0, 60.0, 300.0),
        target_force: float = 5.0,
        force_gain: float = 1.0,
        force_integral_gain: float = 2.0,
        force_integral_limit: float = 25.0,
        max_task_force: float = 60.0,
        filter_alpha: float = 0.15,
        deadband: float = 0.10,
    ) -> None:
        self.model = model
        self.data = data
        self.arm_joint_names = list(arm_joint_names)
        self.arm_motor_names = list(arm_motor_names)

        self.arm_joint_ids = [
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            for name in self.arm_joint_names
        ]
        if any(jid < 0 for jid in self.arm_joint_ids):
            raise ValueError(f"Cannot find all arm joints: {self.arm_joint_names}")

        self.qpos_idx = np.array([model.jnt_qposadr[jid] for jid in self.arm_joint_ids], dtype=int)
        self.qvel_idx = np.array([model.jnt_dofadr[jid] for jid in self.arm_joint_ids], dtype=int)

        self.arm_motor_ids = [
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            for name in self.arm_motor_names
        ]
        if any(aid < 0 for aid in self.arm_motor_ids):
            raise ValueError(f"Cannot find all arm motor actuators: {self.arm_motor_names}")

        self.site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
        if self.site_id < 0:
            raise ValueError(f"Site not found: {site_name}")

        self.kp = np.asarray(kp, dtype=float)
        self.kd = np.asarray(kd, dtype=float)
        self.torque_limits = np.asarray(torque_limits, dtype=float)

        self.target_force = float(target_force)
        self.force_gain = float(force_gain)
        self.force_integral_gain = float(force_integral_gain)
        self.force_integral_limit = abs(float(force_integral_limit))
        self.max_task_force = abs(float(max_task_force))
        self.filter_alpha = float(np.clip(filter_alpha, 0.0, 1.0))
        self.deadband = abs(float(deadband))

        self.force_filtered = 0.0
        self.force_integral = 0.0

    def get_q(self) -> np.ndarray:
        return self.data.qpos[self.qpos_idx].copy()

    def get_dq(self) -> np.ndarray:
        return self.data.qvel[self.qvel_idx].copy()

    def reset_filter(self, measured_force: float = 0.0) -> None:
        self.force_filtered = max(0.0, float(measured_force))
        self.force_integral = 0.0

    def compute(
        self,
        q_ref: Sequence[float],
        measured_force: float,
        down_axis_world: Sequence[float],
        enable_force_term: bool = True,
        dt: float | None = None,
    ) -> TorqueControlState:
        q_ref = np.asarray(q_ref, dtype=float).reshape(len(self.arm_joint_names))
        down_axis = np.asarray(down_axis_world, dtype=float).reshape(3)
        down_axis_norm = np.linalg.norm(down_axis)
        if down_axis_norm < 1e-12:
            down_axis = np.array([0.0, 0.0, -1.0], dtype=float)
        else:
            down_axis = down_axis / down_axis_norm

        if dt is None:
            dt = float(self.model.opt.timestep)
        dt = max(float(dt), 1e-9)

        measured_force = max(0.0, float(measured_force))
        self.force_filtered = (
            (1.0 - self.filter_alpha) * self.force_filtered
            + self.filter_alpha * measured_force
        )
        force_error = self.target_force - self.force_filtered

        mujoco.mj_forward(self.model, self.data)

        q = self.get_q()
        dq = self.get_dq()

        # MuJoCo generalized bias terms: gravity + Coriolis + centrifugal.
        tau_bias = self.data.qfrc_bias[self.qvel_idx].copy()

        # Joint-space posture stabilization.
        tau_pd = self.kp * (q_ref - q) - self.kd * dq

        # Cartesian force term.
        jacp = np.zeros((3, self.model.nv), dtype=float)
        jacr = np.zeros((3, self.model.nv), dtype=float)
        mujoco.mj_jacSite(self.model, self.data, jacp, jacr, self.site_id)
        J = jacp[:, self.qvel_idx]

        if enable_force_term:
            if abs(force_error) > self.deadband:
                self.force_integral += force_error * dt
                self.force_integral = float(
                    np.clip(self.force_integral, -self.force_integral_limit, self.force_integral_limit)
                )
            # If inside deadband, keep integral as-is to hold the achieved force.
            force_mag_raw = self.force_gain * force_error + self.force_integral_gain * self.force_integral
            force_mag = float(np.clip(force_mag_raw, -self.max_task_force, self.max_task_force))
            task_force_saturated = abs(force_mag_raw) > self.max_task_force + 1e-9
            task_force_cmd = down_axis * force_mag
        else:
            task_force_cmd = np.zeros(3, dtype=float)
            task_force_saturated = False

        tau_force = J.T @ task_force_cmd

        tau_raw = tau_bias + tau_pd + tau_force
        tau = np.clip(tau_raw, -self.torque_limits, self.torque_limits)
        saturated = bool(np.any(np.abs(tau_raw) > self.torque_limits + 1e-9))

        for aid, value in zip(self.arm_motor_ids, tau):
            self.data.ctrl[aid] = float(value)

        return TorqueControlState(
            tau=tau.copy(),
            tau_bias=tau_bias.copy(),
            tau_pd=tau_pd.copy(),
            tau_force=tau_force.copy(),
            task_force_cmd=task_force_cmd.copy(),
            force_error=float(force_error),
            force_filtered=float(self.force_filtered),
            force_integral=float(self.force_integral),
            contact_force=float(measured_force),
            saturated=saturated,
            task_force_saturated=task_force_saturated,
        )
