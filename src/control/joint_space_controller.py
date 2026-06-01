# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Sequence

import mujoco
import numpy as np


class JointSpaceController:
    """Torque/force motor controller with optional paper-style JTC/TCC/FSC mode.

    Without ``task_site_name`` this behaves as a joint-space impedance
    controller for simple joints such as the gripper. With ``task_site_name``
    enabled, the commanded joint target is converted to a target end-effector
    pose, a task-space PID generates a wrench, and the Jacobian transpose maps
    that wrench into joint torques:

        tau = -C qdot + J.T @ [f, m] + tau_g + tau_fric + tau_posture

    ``[f, m]`` is split by FSC masks: motion axes use PID feedback, while force
    axes use feed-forward wrench terms. ``tau_posture`` is a low-level joint PID
    used to keep the redundant/prismatic joints near their planned posture.
    """

    def __init__(
        self,
        model,
        data,
        joint_names: Sequence[str],
        actuator_names: Sequence[str] | None = None,
        kp: Sequence[float] | float | None = None,
        kd: Sequence[float] | float | None = None,
        bias_compensation: bool = True,
        task_site_name: str | None = None,
        task_kp_pos: Sequence[float] | float | None = None,
        task_ki_pos: Sequence[float] | float | None = None,
        task_kd_pos: Sequence[float] | float | None = None,
        task_kp_rot: Sequence[float] | float | None = None,
        task_ki_rot: Sequence[float] | float | None = None,
        task_kd_rot: Sequence[float] | float | None = None,
        task_integral_limit: Sequence[float] | float = 0.025,
        joint_ki: Sequence[float] | float | None = None,
        joint_integral_limit: Sequence[float] | float = 0.08,
        damping_shape: Sequence[float] | float | None = None,
        friction_coeff: Sequence[float] | float | None = None,
        friction_velocity: float = 0.015,
    ):
        self.model = model
        self.data = data
        self.joint_names = list(joint_names)
        self.bias_compensation = bool(bias_compensation)
        self.task_site_name = task_site_name
        self.task_site_id = -1
        if task_site_name is not None:
            self.task_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, task_site_name)
            if self.task_site_id < 0:
                raise ValueError(f"Task site not found: {task_site_name}")

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

        self.qpos_idx = np.array([model.jnt_qposadr[jid] for jid in self.joint_ids], dtype=int)
        self.qvel_idx = np.array([model.jnt_dofadr[jid] for jid in self.joint_ids], dtype=int)
        self.q_target = self.get_q()
        self.qd_target = np.zeros(len(self.joint_names), dtype=float)

        self.kp = self._as_gain(kp, self._default_kp())
        self.kd = self._as_gain(kd, 2.0 * np.sqrt(np.maximum(self.kp, 1e-9)))
        self.ki = self._as_gain(joint_ki, np.zeros(len(self.joint_names), dtype=float))
        self.joint_integral = np.zeros(len(self.joint_names), dtype=float)
        self.joint_integral_limit = self._as_gain(joint_integral_limit, np.full(len(self.joint_names), 0.08))

        self.damping_shape = self._as_gain(damping_shape, np.zeros(len(self.joint_names), dtype=float))
        self.friction_coeff = self._as_gain(friction_coeff, np.zeros(len(self.joint_names), dtype=float))
        self.friction_velocity = max(float(friction_velocity), 1e-6)

        self.task_kp_pos = self._as_task_gain(task_kp_pos, [260.0, 260.0, 420.0])
        self.task_ki_pos = self._as_task_gain(task_ki_pos, [4.0, 4.0, 14.0])
        self.task_kd_pos = self._as_task_gain(task_kd_pos, [38.0, 38.0, 60.0])
        self.task_kp_rot = self._as_task_gain(task_kp_rot, [70.0, 70.0, 42.0])
        self.task_ki_rot = self._as_task_gain(task_ki_rot, [0.8, 0.8, 0.35])
        self.task_kd_rot = self._as_task_gain(task_kd_rot, [10.0, 10.0, 7.0])
        self.task_integral_limit = self._as_task_gain(task_integral_limit, [0.025, 0.025, 0.025])
        self.task_pos_integral = np.zeros(3, dtype=float)
        self.task_rot_integral = np.zeros(3, dtype=float)

        self.motion_axis_mask = np.ones(6, dtype=float)
        self.force_axis_mask = np.zeros(6, dtype=float)
        self.feedforward_wrench = np.zeros(6, dtype=float)

        self._target_pose_q: np.ndarray | None = None
        self._target_pos = np.zeros(3, dtype=float)
        self._target_mat = np.eye(3, dtype=float)
        self._last_time = float(data.time)
        self.last_tau = np.zeros(len(self.joint_names), dtype=float)
        self.last_wrench = np.zeros(6, dtype=float)
        self.last_components: dict[str, np.ndarray] = {}

    def _find_actuator_for_joint(self, joint_name: str) -> str:
        candidates = [f"{joint_name}_motor", joint_name, f"{joint_name}_pos"]
        if joint_name == "joint71":
            candidates.insert(0, "gripper_left_motor")
            candidates.append("gripper_opening_left")
        elif joint_name == "joint72":
            candidates.insert(0, "gripper_right_motor")
            candidates.append("gripper_opening_right")
        for name in candidates:
            aid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            if aid >= 0:
                return name
        raise ValueError(f"No actuator found for joint: {joint_name}")

    def _default_kp(self) -> np.ndarray:
        gains = []
        for jid in self.joint_ids:
            if self.model.jnt_type[jid] == mujoco.mjtJoint.mjJNT_SLIDE:
                gains.append(3500.0)
            else:
                gains.append(260.0)
        return np.array(gains, dtype=float)

    def _as_gain(self, value: Sequence[float] | float | None, default: np.ndarray | Sequence[float]) -> np.ndarray:
        default_arr = np.asarray(default, dtype=float)
        if value is None:
            arr = default_arr.astype(float)
        else:
            arr = np.asarray(value, dtype=float)
        if arr.ndim == 0:
            arr = np.full(len(self.joint_names), float(arr), dtype=float)
        return arr.reshape(len(self.joint_names))

    def _as_task_gain(self, value: Sequence[float] | float | None, default: Sequence[float]) -> np.ndarray:
        if value is None:
            arr = np.asarray(default, dtype=float)
        else:
            arr = np.asarray(value, dtype=float)
        if arr.ndim == 0:
            arr = np.full(3, float(arr), dtype=float)
        return arr.reshape(3)

    def get_q(self) -> np.ndarray:
        return self.data.qpos[self.qpos_idx].copy()

    def get_qvel(self) -> np.ndarray:
        return self.data.qvel[self.qvel_idx].copy()

    def limits(self):
        low = []
        high = []
        for jid in self.joint_ids:
            if bool(self.model.jnt_limited[jid]):
                lo, hi = self.model.jnt_range[jid]
            else:
                lo, hi = -np.inf, np.inf
            low.append(float(lo))
            high.append(float(hi))
        return np.array(low, dtype=float), np.array(high, dtype=float)

    def actuator_limits(self):
        low = []
        high = []
        for aid in self.actuator_ids:
            if bool(self.model.actuator_ctrllimited[aid]):
                lo, hi = self.model.actuator_ctrlrange[aid]
            else:
                lo, hi = -np.inf, np.inf
            low.append(float(lo))
            high.append(float(hi))
        return np.array(low, dtype=float), np.array(high, dtype=float)

    def set_motion_axis_mask(self, mask: Sequence[float] | None) -> None:
        self.motion_axis_mask = np.ones(6, dtype=float) if mask is None else np.asarray(mask, dtype=float).reshape(6)
        self.task_pos_integral[:] = 0.0
        self.task_rot_integral[:] = 0.0

    def set_task_feedforward_wrench(
        self,
        wrench: Sequence[float] | None,
        force_axis_mask: Sequence[float] | None = None,
    ) -> None:
        self.feedforward_wrench = np.zeros(6, dtype=float) if wrench is None else np.asarray(wrench, dtype=float).reshape(6)
        if force_axis_mask is None:
            self.force_axis_mask = (np.abs(self.feedforward_wrench) > 1e-12).astype(float)
        else:
            self.force_axis_mask = np.asarray(force_axis_mask, dtype=float).reshape(6)

    def clear_task_force(self) -> None:
        self.feedforward_wrench[:] = 0.0
        self.force_axis_mask[:] = 0.0
        self.motion_axis_mask[:] = 1.0
        self.task_pos_integral[:] = 0.0
        self.task_rot_integral[:] = 0.0

    def set_target(
        self,
        q_des: Sequence[float],
        qd_des: Sequence[float] | None = None,
        clamp: bool = True,
    ) -> np.ndarray:
        q_des = np.asarray(q_des, dtype=float)
        if clamp:
            lo, hi = self.limits()
            q_des = np.minimum(np.maximum(q_des, lo), hi)
        if q_des.shape != self.q_target.shape or not np.allclose(q_des, self.q_target, atol=1e-10, rtol=0.0):
            self._target_pose_q = None
        self.q_target = q_des.copy()
        self.qd_target = np.zeros_like(self.q_target) if qd_des is None else np.asarray(qd_des, dtype=float).copy()
        self.update()
        return q_des

    def update(self) -> np.ndarray:
        mujoco.mj_forward(self.model, self.data)
        if self.task_site_id < 0:
            tau = self._joint_space_tau()
        else:
            tau = self._task_space_tau()

        lo, hi = self.actuator_limits()
        tau = np.minimum(np.maximum(tau, lo), hi)
        for value, aid in zip(tau, self.actuator_ids):
            self.data.ctrl[aid] = value
        self.last_tau = tau.copy()
        return tau

    def _joint_space_tau(self) -> np.ndarray:
        q = self.get_q()
        qd = self.get_qvel()
        tau = self.kp * (self.q_target - q) + self.kd * (self.qd_target - qd)
        if self.bias_compensation:
            tau = tau + self.data.qfrc_bias[self.qvel_idx]
        self.last_wrench[:] = 0.0
        self.last_components = {"joint_pid": tau.copy()}
        return tau

    def _cache_target_pose(self) -> None:
        if self._target_pose_q is not None and np.allclose(self._target_pose_q, self.q_target, atol=1e-10, rtol=0.0):
            return

        qpos_saved = self.data.qpos.copy()
        qvel_saved = self.data.qvel.copy()
        ctrl_saved = self.data.ctrl.copy()
        self.data.qpos[self.qpos_idx] = self.q_target
        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)
        self._target_pos = self.data.site_xpos[self.task_site_id].copy()
        self._target_mat = self.data.site_xmat[self.task_site_id].reshape(3, 3).copy()
        self._target_pose_q = self.q_target.copy()
        self.data.qpos[:] = qpos_saved
        self.data.qvel[:] = qvel_saved
        self.data.ctrl[:] = ctrl_saved
        mujoco.mj_forward(self.model, self.data)

    @staticmethod
    def _rot_error(target_mat: np.ndarray, current_mat: np.ndarray) -> np.ndarray:
        err_mat = target_mat @ current_mat.T
        return 0.5 * np.array(
            [
                err_mat[2, 1] - err_mat[1, 2],
                err_mat[0, 2] - err_mat[2, 0],
                err_mat[1, 0] - err_mat[0, 1],
            ],
            dtype=float,
        )

    def _task_space_tau(self) -> np.ndarray:
        q = self.get_q()
        qd = self.get_qvel()
        self._cache_target_pose()

        dt = max(float(self.data.time) - self._last_time, float(self.model.opt.timestep))
        self._last_time = float(self.data.time)

        jacp_full = np.zeros((3, self.model.nv), dtype=float)
        jacr_full = np.zeros((3, self.model.nv), dtype=float)
        mujoco.mj_jacSite(self.model, self.data, jacp_full, jacr_full, self.task_site_id)
        jacp = jacp_full[:, self.qvel_idx]
        jacr = jacr_full[:, self.qvel_idx]
        jac = np.vstack((jacp, jacr))

        current_pos = self.data.site_xpos[self.task_site_id].copy()
        current_mat = self.data.site_xmat[self.task_site_id].reshape(3, 3).copy()
        xdot = jacp @ qd
        wdot = jacr @ qd
        xdot_des = jacp @ self.qd_target
        wdot_des = jacr @ self.qd_target

        pos_error = self._target_pos - current_pos
        rot_error = self._rot_error(self._target_mat, current_mat)
        active_pos = self.motion_axis_mask[:3]
        active_rot = self.motion_axis_mask[3:]
        self.task_pos_integral = np.clip(
            self.task_pos_integral + active_pos * pos_error * dt,
            -self.task_integral_limit,
            self.task_integral_limit,
        )
        self.task_rot_integral = np.clip(
            self.task_rot_integral + active_rot * rot_error * dt,
            -self.task_integral_limit,
            self.task_integral_limit,
        )

        force_pid = (
            self.task_kp_pos * pos_error
            + self.task_ki_pos * self.task_pos_integral
            + self.task_kd_pos * (xdot_des - xdot)
        )
        moment_pid = (
            self.task_kp_rot * rot_error
            + self.task_ki_rot * self.task_rot_integral
            + self.task_kd_rot * (wdot_des - wdot)
        )
        feedback_wrench = np.concatenate((force_pid, moment_pid))
        wrench = self.motion_axis_mask * feedback_wrench + self.force_axis_mask * self.feedforward_wrench

        q_error = self.q_target - q
        self.joint_integral = np.clip(
            self.joint_integral + q_error * dt,
            -self.joint_integral_limit,
            self.joint_integral_limit,
        )
        tau_posture = self.kp * q_error + self.ki * self.joint_integral + self.kd * (self.qd_target - qd)
        tau_task = jac.T @ wrench
        tau_damping = -self.damping_shape * qd
        tau_fric = self.friction_coeff * np.tanh(qd / self.friction_velocity)
        tau_bias = self.data.qfrc_bias[self.qvel_idx] if self.bias_compensation else 0.0
        tau = tau_bias + tau_task + tau_posture + tau_damping + tau_fric

        self.last_wrench = wrench.copy()
        self.last_components = {
            "bias": np.asarray(tau_bias, dtype=float).copy(),
            "task": tau_task.copy(),
            "posture": tau_posture.copy(),
            "damping": tau_damping.copy(),
            "friction": tau_fric.copy(),
            "pos_error": pos_error.copy(),
            "rot_error": rot_error.copy(),
            "wrench": wrench.copy(),
        }
        return tau
