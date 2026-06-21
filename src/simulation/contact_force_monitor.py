# -*- coding: utf-8 -*-
"""
src/simulation/contact_force_monitor.py

Read MuJoCo normal contact force between selected geom sets.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable
import numpy as np
import mujoco


@dataclass
class ContactForceReading:
    normal_force: float
    contact_count: int
    max_single_contact_force: float


class ContactForceMonitor:
    def __init__(
        self,
        model: mujoco.MjModel,
        object_geom_names: Iterable[str],
        environment_geom_names: Iterable[str],
    ) -> None:
        self.model = model
        self.object_geom_names = set(object_geom_names)
        self.environment_geom_names = set(environment_geom_names)

    def _geom_name(self, geom_id: int) -> str:
        return mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or ""

    def read(self, data: mujoco.MjData) -> ContactForceReading:
        total_normal = 0.0
        count = 0
        max_single = 0.0
        force6 = np.zeros(6, dtype=float)

        for i in range(data.ncon):
            con = data.contact[i]
            g1 = self._geom_name(con.geom1)
            g2 = self._geom_name(con.geom2)
            names = {g1, g2}

            if not (names & self.object_geom_names and names & self.environment_geom_names):
                continue

            force6[:] = 0.0
            mujoco.mj_contactForce(self.model, data, i, force6)
            normal_force = abs(float(force6[0]))

            total_normal += normal_force
            max_single = max(max_single, normal_force)
            count += 1

        return ContactForceReading(
            normal_force=float(total_normal),
            contact_count=int(count),
            max_single_contact_force=float(max_single),
        )
