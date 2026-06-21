# -*- coding: utf-8 -*-
from __future__ import annotations

from pathlib import Path
import mujoco


def load_model_and_data(xml_path: str | Path):
    xml_path = Path(xml_path)
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)
    return model, data
