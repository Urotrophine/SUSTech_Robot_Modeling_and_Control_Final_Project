# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Sequence
import numpy as np
import mujoco


class NumericalIK:
    """Damped least-squares position IK for arm joints only."""

    def __init__(self, model, data, arm_joint_names: Sequence[str], ee_site_name: str = "ee_site"):
        self.model = model
        self.data = data
        self.arm_joint_names = list(arm_joint_names)
        self.ee_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, ee_site_name)
        if self.ee_site_id < 0:
            raise ValueError(f"Site not found: {ee_site_name}")
        self.joint_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, j) for j in self.arm_joint_names]
        self.qpos_idx = np.array([model.jnt_qposadr[jid] for jid in self.joint_ids], dtype=int)
        self.qvel_idx = np.array([model.jnt_dofadr[jid] for jid in self.joint_ids], dtype=int)

    def current_q(self):
        return self.data.qpos[self.qpos_idx].copy()

    def solve(self, target_pos, q_init=None, max_iter=250, tol=1e-4, damping=1e-3):
        target_pos = np.asarray(target_pos, dtype=float).reshape(3)
        q = self.current_q() if q_init is None else np.asarray(q_init, dtype=float).copy()

        for it in range(max_iter):
            self.data.qpos[self.qpos_idx] = q
            self.data.qvel[:] = 0
            mujoco.mj_forward(self.model, self.data)
            pos = self.data.site_xpos[self.ee_site_id].copy()
            err = target_pos - pos
            if np.linalg.norm(err) < tol:
                return q, True, {"iterations": it, "error_norm": float(np.linalg.norm(err))}

            jacp = np.zeros((3, self.model.nv))
            jacr = np.zeros((3, self.model.nv))
            mujoco.mj_jacSite(self.model, self.data, jacp, jacr, self.ee_site_id)
            J = jacp[:, self.qvel_idx]
            A = J @ J.T + damping**2 * np.eye(3)
            dq = J.T @ np.linalg.solve(A, err)
            step_norm = np.linalg.norm(dq)
            if step_norm > 0.12:
                dq = dq / step_norm * 0.12
            q = q + 0.7 * dq

        return q, False, {"iterations": max_iter, "error_norm": float(np.linalg.norm(err))}
