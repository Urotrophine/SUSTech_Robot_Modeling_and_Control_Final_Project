# -*- coding: utf-8 -*-
from __future__ import annotations
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import mujoco
from control.joint_space_controller import JointSpaceController

xml_path = PROJECT_ROOT / "models" / "robot_with_gripper.xml"
model = mujoco.MjModel.from_xml_path(str(xml_path))
data = mujoco.MjData(model)

print("Loaded XML:", xml_path)
print("Bodies:", model.nbody, "Joints:", model.njnt, "Actuators:", model.nu)

arm = JointSpaceController(model, data, ["joint1", "joint2", "joint3", "joint4"])
gripper = JointSpaceController(model, data, ["joint5"], ["gripper_opening"])

arm.set_target([0.0, 0.0, 0.0, 0.0])
gripper.set_target([0.0])
mujoco.mj_forward(model, data)
print("Smoke test passed.")
