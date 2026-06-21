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

from api.arm_platform_api import ArmPlatformAPI
from environment.simple_grasp_environment import SimpleGraspEnvironment

api = ArmPlatformAPI(PROJECT_ROOT / "models" / "simple_grasp_scene.xml")
env = SimpleGraspEnvironment(api.model, api.data)

print("Simple grasp scene loaded.")
print("This demo checks gripper-object contact. It does not perform peg-in-hole insertion.")

# Keep the arm in a stable pose. You can tune this target after visual inspection.
api.set_arm_target([0.0, 0.0, 0.0, 0.0])

with mujoco.viewer.launch_passive(api.model, api.data) as viewer:
    viewer.cam.distance = 1.2

    while viewer.is_running():
        t = api.data.time

        if t < 1.5:
            # Open gripper
            api.set_gripper(-0.03)
        elif t < 4.0:
            # Close gripper
            api.set_gripper(0.03)
        else:
            # Hold
            api.set_gripper(0.03)

        api.step()
        viewer.sync()

        if int(t * 2) != int((t - api.model.opt.timestep) * 2):
            print(
                f"t={t:6.3f}, "
                f"object={env.object_position()}, "
                f"finger_contacts={env.count_finger_contacts()}, "
                f"g={api.get_state().gripper_opening:+.4f}"
            )

        time.sleep(max(0.0, api.model.opt.timestep))
