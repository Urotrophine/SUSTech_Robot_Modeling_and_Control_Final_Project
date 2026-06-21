# -*- coding: utf-8 -*-
from __future__ import annotations

import time
import mujoco
import mujoco.viewer


def run_passive_viewer(model, data, step_callback=None):
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            t0 = time.time()
            if step_callback is not None:
                step_callback(model, data)
            mujoco.mj_step(model, data)
            viewer.sync()
            dt = model.opt.timestep
            elapsed = time.time() - t0
            if elapsed < dt:
                time.sleep(dt - elapsed)
