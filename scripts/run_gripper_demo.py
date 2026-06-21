# -*- coding: utf-8 -*-
from __future__ import annotations
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import time
import numpy as np
import mujoco
import mujoco.viewer
from control.joint_space_controller import JointSpaceController
from gripper.parallel_gripper import ParallelGripper

# Uses the corrected parallel topology model: fin1 and fin2 are sibling bodies under link4.
xml_path = PROJECT_ROOT / "models" / "robot_with_gripper.xml"
model = mujoco.MjModel.from_xml_path(str(xml_path))
data = mujoco.MjData(model)

arm = JointSpaceController(model, data, ["joint1", "joint2", "joint3", "joint4"])
gripper_ctrl = JointSpaceController(model, data, ["joint5"], ["gripper_opening"])
gripper = ParallelGripper(gripper_ctrl)

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        t0 = time.time()
        t = data.time

        # Keep arm near a mild pose
        arm.set_target([0.2*np.sin(0.4*t), -0.2, 0.15, 0.0])

        # Open/close the gripper by commanding the master joint only.
        g = 0.03 * np.sin(1.0 * t)
        gripper.set_opening(g)

        mujoco.mj_step(model, data)
        viewer.sync()

        dt = model.opt.timestep
        elapsed = time.time() - t0
        if elapsed < dt:
            time.sleep(dt - elapsed)
