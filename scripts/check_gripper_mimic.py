# -*- coding: utf-8 -*-
from __future__ import annotations

import sys
import time
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import mujoco
import mujoco.viewer


def main():
    xml_path = PROJECT_ROOT / "models" / "robot_with_gripper.xml"
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)

    j5 = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "joint5")
    j6 = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "joint6")
    a = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "gripper_opening")

    if j5 < 0 or j6 < 0 or a < 0:
        raise RuntimeError("Cannot find joint5, joint6, or gripper_opening actuator.")

    q5adr = model.jnt_qposadr[j5]
    q6adr = model.jnt_qposadr[j6]

    print("Expected mimic relation: joint6 = -joint5")
    print("Only actuator should be gripper_opening -> joint5")
    print("Move gripper in viewer and watch both fingers.")
    print()

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            t0 = time.time()
            t = data.time

            data.ctrl[a] = 0.03 * np.sin(1.0 * t)

            mujoco.mj_step(model, data)
            viewer.sync()

            if int(data.time * 2) != int((data.time - model.opt.timestep) * 2):
                q5 = data.qpos[q5adr]
                q6 = data.qpos[q6adr]
                print(f"t={data.time:6.3f}, joint5={q5:+.5f}, joint6={q6:+.5f}, sum={q5+q6:+.3e}")

            dt = model.opt.timestep
            elapsed = time.time() - t0
            if elapsed < dt:
                time.sleep(dt - elapsed)


if __name__ == "__main__":
    main()
