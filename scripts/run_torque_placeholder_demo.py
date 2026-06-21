# -*- coding: utf-8 -*-
from __future__ import annotations
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import mujoco
from dynamics_control.torque_controller import ComputedTorqueController

from api.arm_platform_api import ArmPlatformAPI

api = ArmPlatformAPI(PROJECT_ROOT / "models" / "robot_with_gripper.xml")
ctrl = ComputedTorqueController(api.model, api.data, ["joint1", "joint2", "joint3", "joint4"])

gc = ctrl.gravity_compensation()
print("Gravity compensation torque placeholder:")
print(gc.tau)

id_tau = ctrl.inverse_dynamics([0.0, 0.0, 0.0, 0.0])
print("Inverse dynamics torque placeholder:")
print(id_tau.tau)

print("""
Note:
Current MJCF uses position actuators, so these torque values are not written
to data.ctrl yet. When switching to torque control, replace position actuators
with motor actuators and route these tau values to data.ctrl.
""")
