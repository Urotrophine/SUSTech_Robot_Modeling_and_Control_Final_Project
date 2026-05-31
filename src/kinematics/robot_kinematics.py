# -*- coding: utf-8 -*-
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Optional, Dict, Tuple

import numpy as np
import mujoco


def clamp(x: np.ndarray, low: np.ndarray, high: np.ndarray) -> np.ndarray:
    return np.minimum(np.maximum(x, low), high)


def quat_wxyz_to_rot(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=float)
    q = q / max(np.linalg.norm(q), 1e-12)
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y*y + z*z), 2 * (x*y - z*w),     2 * (x*z + y*w)],
        [2 * (x*y + z*w),     1 - 2 * (x*x + z*z), 2 * (y*z - x*w)],
        [2 * (x*z - y*w),     2 * (y*z + x*w),     1 - 2 * (x*x + y*y)],
    ], dtype=float)


def rot_error_so3(R_current: np.ndarray, R_target: np.ndarray) -> np.ndarray:
    """Small-angle orientation error from current to target."""
    R_err = R_target @ R_current.T
    return 0.5 * np.array([
        R_err[2, 1] - R_err[1, 2],
        R_err[0, 2] - R_err[2, 0],
        R_err[1, 0] - R_err[0, 1],
    ], dtype=float)


@dataclass
class KinematicsResult:
    q: np.ndarray
    success: bool
    error_norm: float
    iterations: int


class RobotKinematics:
    """Forward/inverse kinematics wrapper for algorithm users.

    This module is the main interface for the algorithm group.

    Arm joint convention for v7:
        q_arm = [q1, q2, q3, q4, d5, q6]

    Units:
        q1, q2, q3, q4, q6: rad
        d5: m

    Default end-effector:
        ee_site

    Notes:
        The gripper joints are deliberately excluded from IK. Gripper opening
        does not define the arm pose and should be controlled by the gripper
        module.
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        arm_joint_names: Sequence[str] = ("joint1", "joint2", "joint3", "joint4", "joint5", "joint6"),
        ee_site_name: str = "ee_site",
    ) -> None:
        self.model = model
        self.data = data
        self.arm_joint_names = list(arm_joint_names)
        self.ee_site_name = ee_site_name

        self.arm_joint_ids = [
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            for name in self.arm_joint_names
        ]
        for name, jid in zip(self.arm_joint_names, self.arm_joint_ids):
            if jid < 0:
                raise ValueError(f"Arm joint not found: {name}")

        self.qpos_idx = np.array([model.jnt_qposadr[jid] for jid in self.arm_joint_ids], dtype=int)
        self.qvel_idx = np.array([model.jnt_dofadr[jid] for jid in self.arm_joint_ids], dtype=int)

        self.ee_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, ee_site_name)
        if self.ee_site_id < 0:
            raise ValueError(f"End-effector site not found: {ee_site_name}")
        self._jacp = np.zeros((3, self.model.nv), dtype=float)
        self._jacr = np.zeros((3, self.model.nv), dtype=float)

    def get_q_arm(self) -> np.ndarray:
        return self.data.qpos[self.qpos_idx].copy()

    def set_q_arm(self, q_arm: Sequence[float], forward: bool = True) -> None:
        q_arm = np.asarray(q_arm, dtype=float).reshape(len(self.arm_joint_names))
        self.data.qpos[self.qpos_idx] = q_arm
        if forward:
            mujoco.mj_forward(self.model, self.data)

    def get_qvel_arm(self) -> np.ndarray:
        return self.data.qvel[self.qvel_idx].copy()

    def get_joint_limits(self) -> Tuple[np.ndarray, np.ndarray]:
        low = []
        high = []
        for jid, name in zip(self.arm_joint_ids, self.arm_joint_names):
            if bool(self.model.jnt_limited[jid]):
                lo, hi = self.model.jnt_range[jid]
            else:
                if name == "joint5":
                    lo, hi = 0.0, 0.2
                else:
                    lo, hi = -3.14, 3.14
            low.append(float(lo))
            high.append(float(hi))
        return np.array(low, dtype=float), np.array(high, dtype=float)

    def forward_kinematics(self, q_arm: Optional[Sequence[float]] = None, site_name: Optional[str] = None) -> Dict[str, np.ndarray]:
        """Return world position and rotation matrix of a site."""
        old_q = None
        if q_arm is not None:
            old_q = self.get_q_arm()
            self.set_q_arm(q_arm, forward=True)
        else:
            mujoco.mj_forward(self.model, self.data)

        sid = self.ee_site_id
        if site_name is not None:
            sid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, site_name)
            if sid < 0:
                raise ValueError(f"Site not found: {site_name}")

        pos = self.data.site_xpos[sid].copy()
        rot = self.data.site_xmat[sid].reshape(3, 3).copy()

        if old_q is not None:
            self.set_q_arm(old_q, forward=True)

        return {"position": pos, "rotation": rot}

    def site_jacobian(self, site_name: Optional[str] = None) -> Tuple[np.ndarray, np.ndarray]:
        sid = self.ee_site_id
        if site_name is not None:
            sid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, site_name)
            if sid < 0:
                raise ValueError(f"Site not found: {site_name}")

        self._jacp.fill(0.0)
        self._jacr.fill(0.0)
        mujoco.mj_jacSite(self.model, self.data, self._jacp, self._jacr, sid)
        return self._jacp[:, self.qvel_idx].copy(), self._jacr[:, self.qvel_idx].copy()

    def solve_ik_position(
        self,
        target_pos: Sequence[float],
        q_init: Optional[Sequence[float]] = None,
        site_name: Optional[str] = None,
        max_iter: int = 250,
        tol: float = 1e-4,
        damping: float = 1e-3,
        step_scale: float = 0.7,
        max_step: float = 0.12,
    ) -> KinematicsResult:
        """Damped least-squares position IK."""
        target_pos = np.asarray(target_pos, dtype=float).reshape(3)
        old_q = self.get_q_arm()

        q = old_q.copy() if q_init is None else np.asarray(q_init, dtype=float).reshape(len(self.arm_joint_names))
        q_low, q_high = self.get_joint_limits()
        q = clamp(q, q_low, q_high)

        sid = self.ee_site_id
        if site_name is not None:
            sid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, site_name)
            if sid < 0:
                raise ValueError(f"Site not found: {site_name}")

        err_norm = np.inf
        for it in range(max_iter):
            self.data.qpos[self.qpos_idx] = q
            self.data.qvel[:] = 0.0
            mujoco.mj_forward(self.model, self.data)

            pos = self.data.site_xpos[sid].copy()
            err = target_pos - pos
            err_norm = float(np.linalg.norm(err))
            if err_norm < tol:
                self.set_q_arm(old_q, forward=True)
                return KinematicsResult(q=q.copy(), success=True, error_norm=err_norm, iterations=it)

            self._jacp.fill(0.0)
            self._jacr.fill(0.0)
            mujoco.mj_jacSite(self.model, self.data, self._jacp, self._jacr, sid)
            J = self._jacp[:, self.qvel_idx]

            A = J @ J.T + (damping ** 2) * np.eye(3)
            dq = J.T @ np.linalg.solve(A, err)

            dq_norm = float(np.linalg.norm(dq))
            if dq_norm > max_step:
                dq *= max_step / dq_norm

            q = q + step_scale * dq
            q = clamp(q, q_low, q_high)

        self.data.qpos[self.qpos_idx] = q
        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)
        final_pos = self.data.site_xpos[sid].copy()
        err_norm = float(np.linalg.norm(target_pos - final_pos))

        self.set_q_arm(old_q, forward=True)
        return KinematicsResult(q=q.copy(), success=False, error_norm=err_norm, iterations=max_iter)

    def solve_ik_pose(
        self,
        target_pos: Sequence[float],
        target_quat_wxyz: Sequence[float],
        q_init: Optional[Sequence[float]] = None,
        site_name: Optional[str] = None,
        pos_weight: float = 1.0,
        rot_weight: float = 0.15,
        max_iter: int = 300,
        tol: float = 1e-4,
        damping: float = 1e-3,
        step_scale: float = 0.5,
        max_step: float = 0.1,
    ) -> KinematicsResult:
        """Optional 6D pose IK.

        Warning:
            The v7 arm has 6 controlled DoF, so pose IK is now feasible, but
            position IK remains the recommended first step for task debugging.
        """
        target_pos = np.asarray(target_pos, dtype=float).reshape(3)
        R_target = quat_wxyz_to_rot(np.asarray(target_quat_wxyz, dtype=float).reshape(4))
        old_q = self.get_q_arm()

        q = old_q.copy() if q_init is None else np.asarray(q_init, dtype=float).reshape(len(self.arm_joint_names))
        q_low, q_high = self.get_joint_limits()
        q = clamp(q, q_low, q_high)

        sid = self.ee_site_id
        if site_name is not None:
            sid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, site_name)
            if sid < 0:
                raise ValueError(f"Site not found: {site_name}")

        err_norm = np.inf
        for it in range(max_iter):
            self.data.qpos[self.qpos_idx] = q
            self.data.qvel[:] = 0.0
            mujoco.mj_forward(self.model, self.data)

            pos = self.data.site_xpos[sid].copy()
            R_current = self.data.site_xmat[sid].reshape(3, 3).copy()

            err_pos = target_pos - pos
            err_rot = rot_error_so3(R_current, R_target)
            err = np.concatenate([pos_weight * err_pos, rot_weight * err_rot])
            err_norm = float(np.linalg.norm(err))
            if err_norm < tol:
                self.set_q_arm(old_q, forward=True)
                return KinematicsResult(q=q.copy(), success=True, error_norm=err_norm, iterations=it)

            self._jacp.fill(0.0)
            self._jacr.fill(0.0)
            mujoco.mj_jacSite(self.model, self.data, self._jacp, self._jacr, sid)
            J = np.vstack([pos_weight * self._jacp[:, self.qvel_idx], rot_weight * self._jacr[:, self.qvel_idx]])

            A = J @ J.T + (damping ** 2) * np.eye(J.shape[0])
            dq = J.T @ np.linalg.solve(A, err)

            dq_norm = float(np.linalg.norm(dq))
            if dq_norm > max_step:
                dq *= max_step / dq_norm

            q = q + step_scale * dq
            q = clamp(q, q_low, q_high)

        self.set_q_arm(old_q, forward=True)
        return KinematicsResult(q=q.copy(), success=False, error_norm=err_norm, iterations=max_iter)
