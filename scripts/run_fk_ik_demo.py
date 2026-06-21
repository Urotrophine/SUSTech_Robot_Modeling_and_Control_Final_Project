# -*- coding: utf-8 -*-
from __future__ import annotations
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np
import mujoco
import mujoco.viewer

from api.arm_platform_api import ArmPlatformAPI

api = ArmPlatformAPI(PROJECT_ROOT / "models" / "robot_with_gripper.xml")

state = api.get_state()
print("[Initial state]")
print("q_arm =", state.q_arm)
print("ee_pos =", state.ee_pos)

fk = api.fk(state.q_arm)
print("\n[FK]")
print("position =", fk["position"])
print("rotation =")
print(fk["rotation"])

target = state.ee_pos + np.array([0.03, 0.02, 0.02])
result = api.ik_position(target)
print("\n[IK position]")
print("target =", target)
print("success =", result.success)
print("q =", result.q)
print("error_norm =", result.error_norm)
print("iterations =", result.iterations)

api.plan_joint_trajectory(result.q, duration=4.0, method="quintic")

with mujoco.viewer.launch_passive(api.model, api.data) as viewer:
    while viewer.is_running():
        api.step()
        viewer.sync()
