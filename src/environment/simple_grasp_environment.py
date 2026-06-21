# -*- coding: utf-8 -*-
from __future__ import annotations

from pathlib import Path
import numpy as np
import mujoco


class SimpleGraspEnvironment:
    """Utility wrapper for the first simple grasp environment.

    The goal of this environment is not peg-in-hole insertion yet. It only
    verifies that:
        1. the gripper can open/close,
        2. simple finger collision exists,
        3. a lightweight object can contact the fingers,
        4. future grasp logic has a stable place to live.
    """

    def __init__(self, model, data, object_body_name: str = "grasp_object"):
        self.model = model
        self.data = data
        self.object_body_name = object_body_name
        self.object_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, object_body_name)
        if self.object_body_id < 0:
            raise ValueError(f"Object body not found: {object_body_name}")

    def object_position(self) -> np.ndarray:
        mujoco.mj_forward(self.model, self.data)
        return self.data.xpos[self.object_body_id].copy()

    def count_finger_contacts(self) -> int:
        count = 0
        for i in range(self.data.ncon):
            con = self.data.contact[i]
            g1 = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, con.geom1) or ""
            g2 = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, con.geom2) or ""
            names = {g1, g2}
            if "grasp_object_collision" in names and (
                "fin1_finger_collision" in names or "fin2_finger_collision" in names
            ):
                count += 1
        return count
