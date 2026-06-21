# -*- coding: utf-8 -*-
from __future__ import annotations
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import mujoco
import mujoco.viewer

xml_path = PROJECT_ROOT / "models" / "robot_with_gripper.xml"
model = mujoco.MjModel.from_xml_path(str(xml_path))
data = mujoco.MjData(model)
print("Loaded:", xml_path)
print("nq =", model.nq, "nv =", model.nv, "nu =", model.nu)
mujoco.viewer.launch(model, data)
