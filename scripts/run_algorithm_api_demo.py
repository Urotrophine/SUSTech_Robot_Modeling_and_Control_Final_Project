# -*- coding: utf-8 -*-
from __future__ import annotations
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import time
import numpy as np
import mujoco.viewer

from api.arm_platform_api import ArmPlatformAPI

api = ArmPlatformAPI(PROJECT_ROOT / "models" / "robot_with_gripper.xml")

print("API loaded.")
print("Initial state:", api.get_state())

q_goal = np.array([0.25, -0.25, 0.2, 0.02])
api.plan_joint_trajectory(q_goal, duration=3.0, method="quintic")

with mujoco.viewer.launch_passive(api.model, api.data) as viewer:
    while viewer.is_running():
        t = api.data.time

        if 4.0 < t < 6.0:
            api.open_gripper()
        elif 6.0 <= t < 8.0:
            api.close_gripper()

        api.step()
        viewer.sync()

        if int(t * 2) != int((t - api.model.opt.timestep) * 2):
            s = api.get_state()
            print(f"t={s.time:.2f}, q={s.q_arm}, g={s.gripper_opening:.4f}, ee={s.ee_pos}")
