# -*- coding: utf-8 -*-
"""
MuJoCo GUI demo: v7 arm peg-in-hole trial with Archimedean spiral search.

After grasping the peg, the arm intentionally keeps a small lateral alignment
error near the hole, lowers the peg to a constant top-contact height, and then
uses only joint1/joint2 to move the carried peg center along an Archimedean
spiral. When the peg projection is inside the hole projection, joint5 performs
the final insertion.

Run:
    python scripts/run_v6_peg_in_hole_spiral_search_demo.py

Headless check:
    python scripts/run_v6_peg_in_hole_spiral_search_demo.py --headless
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from time import perf_counter

import mujoco
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from api.arm_platform_api import ArmPlatformAPI  # noqa: E402


BASE_MODEL_PATH = PROJECT_ROOT / "models" / "simple_grasp_scene.xml"
DEMO_MODEL_PATH = PROJECT_ROOT / "models" / f"_v7_peg_insert_spiral_search_scene_{os.getpid()}.xml"
CONFIG_PATH = PROJECT_ROOT / "configs" / "robot_description.yaml"

CYL_RADIUS = 0.035
CYL_HALF_LENGTH = 0.055
HOLE_RADIUS = 0.039
HOLE_DEPTH = 0.080
TABLE_TOP_Z = 0.0
TABLE_HALF_Z = 0.010

FIXED_Q3 = 0.0
FIXED_Q4 = 0.0
FIXED_Q6 = 0.0
LIFT_JOINT5 = 0.0
OPEN_CMD = -0.05
CLOSE_CMD = 0.014

DEFAULT_OBJECT_POS = np.array([-0.280, -0.620, TABLE_TOP_Z + CYL_HALF_LENGTH], dtype=float)
OBJECT_POS = DEFAULT_OBJECT_POS.copy()
OBJECT_QUAT = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
HOLE_CENTER = np.array([-0.374, -0.631, TABLE_TOP_Z + HOLE_DEPTH / 2.0], dtype=float)
HOME_Q = np.array([0.0, 0.0, FIXED_Q3, FIXED_Q4, LIFT_JOINT5, FIXED_Q6], dtype=float)
OBJECT_SPAWN_XY_MIN = np.array([-0.760, -0.880], dtype=float)
OBJECT_SPAWN_XY_MAX = np.array([0.040, -0.300], dtype=float)
HOLE_BLOCK_HALF_WIDTH = 0.145
OBJECT_HOLE_BLOCK_CLEARANCE = CYL_RADIUS + 0.035
MIN_OBJECT_HOLE_DISTANCE = HOLE_BLOCK_HALF_WIDTH + OBJECT_HOLE_BLOCK_CLEARANCE
GRASP_HEIGHT_ABOVE_CENTER = CYL_HALF_LENGTH * 0.95
INSERT_DEPTH_TARGET = min(0.070, HOLE_DEPTH - 0.010)
INSERT_JOINT5_MARGIN = 0.004
CONTACT_GAP = 0.0015
SEARCH_ALIGN_ERROR = np.array([0.024, -0.018], dtype=float)
SPIRAL_PITCH = 0.012
SPIRAL_THETA_STEP = 0.24
SPIRAL_MAX_RADIUS = 0.065
SPIRAL_STEP_DURATION = 0.035
SPIRAL_SUCCESS_RADIUS = max(0.003, HOLE_RADIUS - CYL_RADIUS + 0.001)
SCREW_TURNS = 2.0
SCREW_INSERT_DURATION = 1.10
CONTACT_DOWN_FORCE = 1.2
INSERT_DOWN_FORCE = 2.0


def name_id(model: mujoco.MjModel, obj_type: mujoco.mjtObj, name: str) -> int:
    oid = mujoco.mj_name2id(model, obj_type, name)
    if oid < 0:
        raise ValueError(f"Missing {obj_type.name}: {name}")
    return oid


def add_box(parent: ET.Element, name: str, pos, size, rgba: str, contact: dict[str, str]) -> None:
    attrs = {
        "name": name,
        "type": "box",
        "pos": f"{pos[0]:.6f} {pos[1]:.6f} {pos[2]:.6f}",
        "size": f"{size[0]:.6f} {size[1]:.6f} {size[2]:.6f}",
        "rgba": rgba,
    }
    attrs.update(contact)
    ET.SubElement(parent, "geom", attrs)


def create_demo_scene() -> Path:
    tree = ET.parse(BASE_MODEL_PATH)
    root = tree.getroot()
    root.set("model", "robotic_arm_v7_peg_insert_physical_half_height")
    world = root.find("worldbody")
    if world is None:
        raise RuntimeError("MJCF worldbody not found")

    for body in list(world.findall("body")):
        if body.get("name") in {"fixed_hole_block", "raised_work_table"}:
            world.remove(body)

    for geom in root.findall(".//geom"):
        if geom.get("name") == "grasp_object_collision":
            geom.set("type", "cylinder")
            geom.set("size", f"{CYL_RADIUS} {CYL_HALF_LENGTH}")
            geom.set("mass", "0.00005")
            geom.set("friction", "600 80 5")
            geom.set("condim", "6")
            geom.set("contype", "1")
            geom.set("conaffinity", "7")
            geom.set("solref", "0.001 1")
            geom.set("solimp", "0.995 0.999 0.0001")
        if geom.get("name") in {"fin1_visual", "fin2_visual"}:
            geom.set("friction", "600 80 5")
            geom.set("condim", "6")
            geom.set("solref", "0.001 1")
            geom.set("solimp", "0.995 0.999 0.0001")

    finger_pad_contact = {
        "type": "box",
        "size": "0.030 0.010 0.050",
        "rgba": "0 0.8 0.2 0.20",
        "contype": "2",
        "conaffinity": "0",
        "group": "3",
        "density": "0",
        "condim": "6",
        "friction": "600 80 5",
        "solref": "0.001 1",
        "solimp": "0.995 0.999 0.0001",
    }
    for body in root.findall(".//body"):
        if body.get("name") == "fin1":
            attrs = dict(finger_pad_contact)
            attrs.update({"name": "fin1_pad_collision", "pos": "0.04 -0.046 0.0575"})
            ET.SubElement(body, "geom", attrs)
        elif body.get("name") == "fin2":
            attrs = dict(finger_pad_contact)
            attrs.update({"name": "fin2_pad_collision", "pos": "0.04 -0.046 0.1425"})
            ET.SubElement(body, "geom", attrs)

    table_contact = {
        "contype": "4",
        "conaffinity": "1",
        "friction": "3 0.5 0.05",
        "condim": "6",
        "solref": "0.002 1",
        "solimp": "0.98 0.995 0.0005",
    }
    table = ET.SubElement(world, "body", {"name": "raised_work_table", "pos": "0 0 0"})
    add_box(
        table,
        "raised_work_table_collision",
        [-0.40, -0.56, TABLE_TOP_Z - TABLE_HALF_Z],
        [0.62, 0.36, TABLE_HALF_Z],
        "0.38 0.38 0.36 1",
        table_contact,
    )

    block_contact = {
        "contype": "4",
        "conaffinity": "1",
        "friction": "5 0.8 0.08",
        "condim": "6",
        "solref": "0.002 1",
        "solimp": "0.98 0.995 0.0005",
    }
    block = ET.SubElement(world, "body", {"name": "fixed_hole_block", "pos": "0 0 0"})
    cx, cy, cz = HOLE_CENTER
    z_half = HOLE_DEPTH / 2.0
    block_half = HOLE_BLOCK_HALF_WIDTH
    wall = 0.040

    # Full fixed block volume around the circular opening.
    side = (block_half - HOLE_RADIUS) / 2.0
    add_box(block, "hole_block_left", [cx - HOLE_RADIUS - side, cy, cz], [side, block_half, z_half], "0.13 0.18 0.23 1", block_contact)
    add_box(block, "hole_block_right", [cx + HOLE_RADIUS + side, cy, cz], [side, block_half, z_half], "0.13 0.18 0.23 1", block_contact)
    add_box(block, "hole_block_front", [cx, cy - HOLE_RADIUS - side, cz], [HOLE_RADIUS, side, z_half], "0.13 0.18 0.23 1", block_contact)
    add_box(block, "hole_block_back", [cx, cy + HOLE_RADIUS + side, cz], [HOLE_RADIUS, side, z_half], "0.13 0.18 0.23 1", block_contact)

    # Circular inner wall approximation. These geoms provide the round hole.
    nseg = 96
    outer_radius = HOLE_RADIUS + wall
    seg_half_len = outer_radius * math.pi / nseg * 1.08
    for i in range(nseg):
        theta = 2.0 * math.pi * i / nseg
        px = cx + (HOLE_RADIUS + wall / 2.0) * math.cos(theta)
        py = cy + (HOLE_RADIUS + wall / 2.0) * math.sin(theta)
        attrs = {
            "name": f"hole_wall_{i:02d}",
            "type": "box",
            "pos": f"{px:.6f} {py:.6f} {cz:.6f}",
            "euler": f"0 0 {theta:.6f}",
            "size": f"{wall / 2.0:.6f} {seg_half_len:.6f} {z_half:.6f}",
            "rgba": "0.05 0.26 0.75 0.55",
        }
        attrs.update(block_contact)
        ET.SubElement(block, "geom", attrs)

    ET.SubElement(
        block,
        "site",
        {
            "name": "hole_entry_site",
            "pos": f"{cx:.6f} {cy:.6f} {TABLE_TOP_Z + HOLE_DEPTH:.6f}",
            "size": f"{HOLE_RADIUS:.6f}",
            "rgba": "0 0.35 1 0.45",
        },
    )
    ET.SubElement(
        block,
        "camera",
        {
            "name": "hole_top_camera",
            "pos": f"{cx:.6f} {cy:.6f} {TABLE_TOP_Z + 0.55:.6f}",
            "euler": "0 0 0",
        },
    )

    ET.indent(tree, space="  ")
    tree.write(DEMO_MODEL_PATH, encoding="utf-8", xml_declaration=True)
    return DEMO_MODEL_PATH


def reset_object(api: ArmPlatformAPI) -> None:
    joint_id = name_id(api.model, mujoco.mjtObj.mjOBJ_JOINT, "grasp_object_freejoint")
    qadr = api.model.jnt_qposadr[joint_id]
    vadr = api.model.jnt_dofadr[joint_id]
    api.data.qpos[qadr:qadr + 3] = OBJECT_POS
    api.data.qpos[qadr + 3:qadr + 7] = OBJECT_QUAT
    api.data.qvel[vadr:vadr + 6] = 0.0
    mujoco.mj_forward(api.model, api.data)
    print(f"object reset pos={OBJECT_POS.tolist()}")


def finger_midpoint(api: ArmPlatformAPI) -> np.ndarray:
    left = api.data.site_xpos[name_id(api.model, mujoco.mjtObj.mjOBJ_SITE, "left_finger_tip_site")]
    right = api.data.site_xpos[name_id(api.model, mujoco.mjtObj.mjOBJ_SITE, "right_finger_tip_site")]
    return 0.5 * (left + right)


def joint_range(api: ArmPlatformAPI, joint_name: str) -> tuple[float, float]:
    jid = name_id(api.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    return float(api.model.jnt_range[jid, 0]), float(api.model.jnt_range[jid, 1])


def q_with_joint5(q_arm: np.ndarray, joint5_value: float) -> np.ndarray:
    q = np.asarray(q_arm, dtype=float).copy()
    q[4] = float(joint5_value)
    return q


def finger_z_at_joint5(api: ArmPlatformAPI, q_arm: np.ndarray, joint5_value: float) -> float:
    saved_qpos = api.data.qpos.copy()
    saved_qvel = api.data.qvel.copy()
    set_arm_qpos(api, q_with_joint5(q_arm, joint5_value))
    mujoco.mj_forward(api.model, api.data)
    z = float(finger_midpoint(api)[2])
    api.data.qpos[:] = saved_qpos
    api.data.qvel[:] = saved_qvel
    mujoco.mj_forward(api.model, api.data)
    return z


def object_pos_with_lock_at_joint5(
    api: ArmPlatformAPI,
    q_arm: np.ndarray,
    joint5_value: float,
    grasp_lock: tuple[np.ndarray, np.ndarray],
) -> np.ndarray:
    saved_qpos = api.data.qpos.copy()
    saved_qvel = api.data.qvel.copy()
    set_arm_qpos(api, q_with_joint5(q_arm, joint5_value))
    apply_grasp_lock(api, grasp_lock)
    mujoco.mj_forward(api.model, api.data)
    pos, _ = object_pose(api)
    api.data.qpos[:] = saved_qpos
    api.data.qvel[:] = saved_qvel
    mujoco.mj_forward(api.model, api.data)
    return pos


def solve_joint5_for_finger_z(api: ArmPlatformAPI, q_arm: np.ndarray, target_z: float) -> float:
    low, high = joint_range(api, "joint5")
    z_low = finger_z_at_joint5(api, q_arm, low)
    z_high = finger_z_at_joint5(api, q_arm, high)
    if abs(z_high - z_low) < 1e-9:
        return high
    s = (float(target_z) - z_low) / (z_high - z_low)
    return float(np.clip(low + s * (high - low), low, high))


def solve_joint5_for_object_z(
    api: ArmPlatformAPI,
    q_arm: np.ndarray,
    grasp_lock: tuple[np.ndarray, np.ndarray],
    target_z: float,
) -> float:
    low, high = joint_range(api, "joint5")
    z_low = object_pos_with_lock_at_joint5(api, q_arm, low, grasp_lock)[2]
    z_high = object_pos_with_lock_at_joint5(api, q_arm, high, grasp_lock)[2]
    if abs(z_high - z_low) < 1e-9:
        return high
    s = (float(target_z) - z_low) / (z_high - z_low)
    return float(np.clip(low + s * (high - low), low, high))


def compute_object_spawn(api: ArmPlatformAPI, rng: np.random.Generator) -> np.ndarray:
    q_probe = HOME_Q.copy()
    low5, high5 = joint_range(api, "joint5")
    q_probe[4] = high5
    for _ in range(240):
        xy = rng.uniform(OBJECT_SPAWN_XY_MIN, OBJECT_SPAWN_XY_MAX)
        if np.all(np.abs(xy - HOLE_CENTER[:2]) < MIN_OBJECT_HOLE_DISTANCE):
            continue
        try:
            solve_revolute12_xy(api, xy, q_probe, "spawn_probe", verbose=False)
        except RuntimeError:
            continue
        return np.array([xy[0], xy[1], TABLE_TOP_Z + CYL_HALF_LENGTH], dtype=float)
    print("  random spawn sampling fell back to default reachable object position")
    return DEFAULT_OBJECT_POS.copy()


def joint12_only_target(q_current: np.ndarray, q_xy: np.ndarray) -> np.ndarray:
    q = np.asarray(q_current, dtype=float).copy()
    q[:2] = np.asarray(q_xy, dtype=float)[:2]
    q[2] = FIXED_Q3
    q[3] = FIXED_Q4
    q[5] = FIXED_Q6
    return q


def joint5_only_target(q_current: np.ndarray, joint5_value: float) -> np.ndarray:
    q = np.asarray(q_current, dtype=float).copy()
    q[4] = float(joint5_value)
    q[2] = FIXED_Q3
    q[3] = FIXED_Q4
    q[5] = FIXED_Q6
    return q


def archimedean_spiral_offsets() -> list[np.ndarray]:
    offsets: list[np.ndarray] = [np.zeros(2, dtype=float)]
    theta = SPIRAL_THETA_STEP
    while True:
        radius = SPIRAL_PITCH * theta / (2.0 * math.pi)
        if radius > SPIRAL_MAX_RADIUS:
            break
        offsets.append(radius * np.array([math.cos(theta), math.sin(theta)], dtype=float))
        theta += SPIRAL_THETA_STEP
    return offsets


def peg_inserted_depth(pos: np.ndarray) -> float:
    peg_bottom_z = float(pos[2] - CYL_HALF_LENGTH)
    return float(np.clip(TABLE_TOP_Z + HOLE_DEPTH - peg_bottom_z, 0.0, HOLE_DEPTH))


def peg_center_xy(api: ArmPlatformAPI) -> np.ndarray:
    pos, _ = object_pose(api)
    return pos[:2]


def rotmat_to_quat_wxyz(rot: np.ndarray) -> np.ndarray:
    quat = np.empty(4, dtype=float)
    mujoco.mju_mat2Quat(quat, np.asarray(rot, dtype=float).reshape(9))
    if quat[0] < 0.0:
        quat *= -1.0
    return quat / max(np.linalg.norm(quat), 1e-12)


def body_world_pose(api: ArmPlatformAPI, body_name: str) -> tuple[np.ndarray, np.ndarray]:
    bid = name_id(api.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    return api.data.xpos[bid].copy(), api.data.xmat[bid].reshape(3, 3).copy()


def body_world_pose_at_arm_q(api: ArmPlatformAPI, body_name: str, q_arm: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    saved_qpos = api.data.qpos.copy()
    saved_qvel = api.data.qvel.copy()
    api.data.qpos[api.kin.qpos_idx] = np.asarray(q_arm, dtype=float)
    api.data.qvel[api.kin.qvel_idx] = 0.0
    mujoco.mj_forward(api.model, api.data)
    pose = body_world_pose(api, body_name)
    api.data.qpos[:] = saved_qpos
    api.data.qvel[:] = saved_qvel
    mujoco.mj_forward(api.model, api.data)
    return pose


def object_pose_from_lock_at_arm_q(
    api: ArmPlatformAPI,
    q_arm: np.ndarray,
    grasp_lock: tuple[np.ndarray, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    rel_pos, rel_rot = grasp_lock
    link_pos, link_rot = body_world_pose_at_arm_q(api, "link6", q_arm)
    return link_pos + link_rot @ rel_pos, link_rot @ rel_rot


def capture_grasp_lock(api: ArmPlatformAPI) -> tuple[np.ndarray, np.ndarray]:
    mujoco.mj_forward(api.model, api.data)
    link_pos, link_rot = body_world_pose(api, "link6")
    obj_pos, obj_rot = body_world_pose(api, "grasp_object")
    rel_pos = link_rot.T @ (obj_pos - link_pos)
    rel_rot = link_rot.T @ obj_rot
    return rel_pos, rel_rot


def apply_grasp_lock(
    api: ArmPlatformAPI,
    grasp_lock: tuple[np.ndarray, np.ndarray] | None,
    position_q_arm: np.ndarray | None = None,
) -> None:
    if grasp_lock is None:
        return
    mujoco.mj_forward(api.model, api.data)
    rel_pos, rel_rot = grasp_lock
    link_pos, link_rot = body_world_pose(api, "link6")
    pos_link_pos, pos_link_rot = (link_pos, link_rot)
    if position_q_arm is not None:
        pos_link_pos, pos_link_rot = body_world_pose_at_arm_q(api, "link6", position_q_arm)
    obj_pos = pos_link_pos + pos_link_rot @ rel_pos
    obj_rot = pos_link_rot @ rel_rot
    joint_id = name_id(api.model, mujoco.mjtObj.mjOBJ_JOINT, "grasp_object_freejoint")
    qadr = api.model.jnt_qposadr[joint_id]
    vadr = api.model.jnt_dofadr[joint_id]
    api.data.qpos[qadr:qadr + 3] = obj_pos
    api.data.qpos[qadr + 3:qadr + 7] = rotmat_to_quat_wxyz(obj_rot)
    api.data.qvel[vadr:vadr + 6] = 0.0


def solve_revolute12_xy(
    api: ArmPlatformAPI,
    target_xy: np.ndarray,
    q_init: np.ndarray,
    label: str,
    verbose: bool = True,
) -> np.ndarray:
    start = perf_counter()
    model = api.model
    data = api.data
    saved_qpos = data.qpos.copy()
    saved_qvel = data.qvel.copy()
    saved_ctrl = data.ctrl.copy()
    q = np.asarray(q_init, dtype=float).copy()
    q[2] = FIXED_Q3
    q[3] = FIXED_Q4
    q[5] = FIXED_Q6

    arm_idx = api.kin.qpos_idx
    joint_ids = [name_id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in ("joint1", "joint2")]
    qvel_idx = np.array([model.jnt_dofadr[jid] for jid in joint_ids], dtype=int)
    q_low = np.array([model.jnt_range[jid, 0] for jid in joint_ids], dtype=float)
    q_high = np.array([model.jnt_range[jid, 1] for jid in joint_ids], dtype=float)
    site_ids = [
        name_id(model, mujoco.mjtObj.mjOBJ_SITE, "left_finger_tip_site"),
        name_id(model, mujoco.mjtObj.mjOBJ_SITE, "right_finger_tip_site"),
    ]
    jacp = np.zeros((3, model.nv), dtype=float)
    jacr = np.zeros((3, model.nv), dtype=float)

    err_norm = float("inf")
    iterations = 0
    for iterations in range(80):
        data.qpos[arm_idx] = q
        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)
        err = np.asarray(target_xy, dtype=float) - finger_midpoint(api)[:2]
        err_norm = float(np.linalg.norm(err))
        if err_norm < 0.004:
            break

        J = np.zeros((2, 2), dtype=float)
        for sid in site_ids:
            mujoco.mj_jacSite(model, data, jacp, jacr, sid)
            J += jacp[:2, qvel_idx] * 0.5
        dq = J.T @ np.linalg.solve(J @ J.T + 1e-5 * np.eye(2), err)
        dq_norm = float(np.linalg.norm(dq))
        if dq_norm > 0.12:
            dq *= 0.12 / dq_norm
        q[:2] = np.clip(q[:2] + 0.9 * dq, q_low, q_high)
        q[2] = FIXED_Q3
        q[3] = FIXED_Q4
        q[5] = FIXED_Q6

    try:
        data.qpos[arm_idx] = q
        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)
        final_err = float(np.linalg.norm(np.asarray(target_xy) - finger_midpoint(api)[:2]))
        if verbose:
            print(f"IK {label:14s}: iter={iterations:3d} time={(perf_counter() - start) * 1000.0:5.1f} ms xy_err={final_err:.4f}")
        if final_err > 0.025:
            raise RuntimeError(f"XY target {label} is too far: {final_err:.4f} m")
        return q.copy()
    finally:
        data.qpos[:] = saved_qpos
        data.qvel[:] = saved_qvel
        data.ctrl[:] = saved_ctrl
        mujoco.mj_forward(model, data)


def solve_revolute12_for_locked_object_xy(
    api: ArmPlatformAPI,
    target_xy: np.ndarray,
    q_init: np.ndarray,
    grasp_lock: tuple[np.ndarray, np.ndarray],
    label: str,
    verbose: bool = False,
) -> np.ndarray:
    start = perf_counter()
    q = np.asarray(q_init, dtype=float).copy()
    q[2] = FIXED_Q3
    q[3] = FIXED_Q4

    joint_ids = [name_id(api.model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in ("joint1", "joint2")]
    q_low = np.array([api.model.jnt_range[jid, 0] for jid in joint_ids], dtype=float)
    q_high = np.array([api.model.jnt_range[jid, 1] for jid in joint_ids], dtype=float)
    target_xy = np.asarray(target_xy, dtype=float)

    final_err = float("inf")
    for iterations in range(18):
        obj_pos, _ = object_pose_from_lock_at_arm_q(api, q, grasp_lock)
        err = target_xy - obj_pos[:2]
        final_err = float(np.linalg.norm(err))
        if final_err < 0.0015:
            break

        jac = np.zeros((2, 2), dtype=float)
        eps = 1e-4
        for col in range(2):
            q_eps = q.copy()
            q_eps[col] += eps
            pos_eps, _ = object_pose_from_lock_at_arm_q(api, q_eps, grasp_lock)
            jac[:, col] = (pos_eps[:2] - obj_pos[:2]) / eps

        dq = jac.T @ np.linalg.solve(jac @ jac.T + 2e-5 * np.eye(2), err)
        dq_norm = float(np.linalg.norm(dq))
        if dq_norm > 0.075:
            dq *= 0.075 / dq_norm
        q[:2] = np.clip(q[:2] + 0.85 * dq, q_low, q_high)
        q[2] = FIXED_Q3
        q[3] = FIXED_Q4

    if verbose:
        print(f"IK {label:14s}: time={(perf_counter() - start) * 1000.0:5.1f} ms xy_err={final_err:.4f}")
    if final_err > 0.012:
        raise RuntimeError(f"Locked-object XY target {label} is too far: {final_err:.4f} m")
    return q.copy()


def smoothstep5(s: float) -> float:
    s = float(np.clip(s, 0.0, 1.0))
    return 10.0 * s**3 - 15.0 * s**4 + 6.0 * s**5


def set_arm_qpos(api: ArmPlatformAPI, q_arm: np.ndarray, qvel_arm: np.ndarray | None = None) -> None:
    q = np.asarray(q_arm, dtype=float).copy()
    q[2] = FIXED_Q3
    q[3] = FIXED_Q4
    q[5] = FIXED_Q6
    api.data.qpos[api.kin.qpos_idx] = q
    api.data.qvel[api.kin.qvel_idx] = 0.0 if qvel_arm is None else qvel_arm
    api.data.qvel[api.kin.qvel_idx[2]] = 0.0
    api.data.qvel[api.kin.qvel_idx[3]] = 0.0
    api.data.qvel[api.kin.qvel_idx[5]] = 0.0
    api.arm_target = q.copy()
    api.arm_controller.set_target(q)


def command_arm_target(
    api: ArmPlatformAPI,
    q_arm: np.ndarray,
    qvel_arm: np.ndarray | None = None,
    lock_joint6: bool = True,
    joint34_target: np.ndarray | None = None,
) -> None:
    q = np.asarray(q_arm, dtype=float).copy()
    if joint34_target is None:
        q[2] = FIXED_Q3
        q[3] = FIXED_Q4
    else:
        q[2:4] = np.asarray(joint34_target, dtype=float)
    if lock_joint6:
        q[5] = FIXED_Q6
    qd = None if qvel_arm is None else np.asarray(qvel_arm, dtype=float).copy()
    if qd is not None and lock_joint6:
        qd[5] = 0.0
    api.arm_target = q.copy()
    api.arm_controller.set_target(q, qd_des=qd)


def set_gripper_qpos(api: ArmPlatformAPI, opening: float) -> None:
    for name, value in (("joint71", opening), ("joint72", -opening)):
        jid = name_id(api.model, mujoco.mjtObj.mjOBJ_JOINT, name)
        api.data.qpos[api.model.jnt_qposadr[jid]] = float(value)
        api.data.qvel[api.model.jnt_dofadr[jid]] = 0.0
    api.set_gripper(opening)


def hold_gripper_target(api: ArmPlatformAPI, opening: float) -> None:
    api.set_gripper(opening)


def set_fsc_down_force(api: ArmPlatformAPI, force_n: float) -> None:
    """Use FSC: z force is mostly feed-forward, x/y and orientation remain PID controlled."""
    if hasattr(api.arm_controller, "set_motion_axis_mask"):
        api.arm_controller.set_motion_axis_mask([1.0, 1.0, 0.35, 1.0, 1.0, 1.0])
        api.arm_controller.set_task_feedforward_wrench(
            [0.0, 0.0, -abs(float(force_n)), 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        )


def clear_fsc_force(api: ArmPlatformAPI) -> None:
    if hasattr(api.arm_controller, "clear_task_force"):
        api.arm_controller.clear_task_force()


def step(api: ArmPlatformAPI, viewer=None) -> None:
    api.update_control()
    mujoco.mj_step(api.model, api.data)
    if viewer is not None:
        viewer.sync()
        time.sleep(max(0.0, api.model.opt.timestep))


def hold_current(
    api: ArmPlatformAPI,
    seconds: float,
    viewer=None,
    gripper_closed: bool = False,
    grasp_lock=None,
    lock_joint6: bool = True,
    decouple_screw_position: bool = False,
    position_joint6: float | None = None,
    position_q_arm: np.ndarray | None = None,
    preserve_joint34: bool = False,
) -> None:
    q = api.arm_controller.get_q()
    joint34_target = q[2:4].copy() if preserve_joint34 else None
    steps = max(1, int(seconds / api.model.opt.timestep))
    for _ in range(steps):
        command_arm_target(api, q, lock_joint6=lock_joint6, joint34_target=joint34_target)
        if gripper_closed:
            hold_gripper_target(api, CLOSE_CMD)
        position_q = None
        if decouple_screw_position:
            position_q = q.copy() if position_q_arm is None else np.asarray(position_q_arm, dtype=float).copy()
            position_q[5] = q[5] if position_joint6 is None else float(position_joint6)
        apply_grasp_lock(api, grasp_lock, position_q_arm=position_q)
        step(api, viewer)


def move_arm(
    api: ArmPlatformAPI,
    q_goal: np.ndarray,
    duration: float,
    viewer=None,
    gripper_closed: bool = False,
    grasp_lock=None,
    stop_on_object_collision: bool = False,
    lock_joint6: bool = True,
    decouple_screw_position: bool = False,
    preserve_joint34: bool = False,
) -> None:
    q_start = api.arm_controller.get_q()
    q_goal = np.asarray(q_goal, dtype=float).copy()
    joint34_target = q_start[2:4].copy() if preserve_joint34 else None
    if preserve_joint34:
        q_goal[2:4] = joint34_target
    else:
        q_goal[2] = FIXED_Q3
        q_goal[3] = FIXED_Q4
    if lock_joint6:
        q_goal[5] = FIXED_Q6
    steps = max(1, int(duration / api.model.opt.timestep))
    prev_q = q_start.copy()
    final_q = q_start.copy()
    for i in range(steps):
        s = smoothstep5((i + 1) / steps)
        q = q_start + s * (q_goal - q_start)
        if preserve_joint34:
            q[2:4] = joint34_target
        else:
            q[2] = FIXED_Q3
            q[3] = FIXED_Q4
        if lock_joint6:
            q[5] = FIXED_Q6
        qvel = (q - prev_q) / api.model.opt.timestep
        qvel[2] = 0.0
        qvel[3] = 0.0
        if lock_joint6:
            qvel[5] = 0.0
        command_arm_target(api, q, qvel, lock_joint6=lock_joint6, joint34_target=joint34_target)
        if gripper_closed:
            hold_gripper_target(api, CLOSE_CMD)
        position_q = None
        if decouple_screw_position:
            position_q = q.copy()
            position_q[5] = q_start[5]
        apply_grasp_lock(api, grasp_lock, position_q_arm=position_q)
        if stop_on_object_collision:
            mujoco.mj_forward(api.model, api.data)
            if count_bad_robot_object_contacts(api) > 0:
                print("  insertion stopped: predicted robot-object collision")
                final_q = prev_q.copy()
                command_arm_target(api, final_q, lock_joint6=lock_joint6, joint34_target=joint34_target)
                if gripper_closed:
                    hold_gripper_target(api, CLOSE_CMD)
                apply_grasp_lock(api, grasp_lock, position_q_arm=position_q)
                break
        prev_q = q.copy()
        final_q = q.copy()
        step(api, viewer)
    else:
        final_q = q_goal.copy()
    command_arm_target(api, final_q, lock_joint6=lock_joint6, joint34_target=joint34_target)
    if gripper_closed:
        hold_gripper_target(api, CLOSE_CMD)
    position_q = None
    if decouple_screw_position:
        position_q = final_q.copy()
        position_q[5] = q_start[5]
    apply_grasp_lock(api, grasp_lock, position_q_arm=position_q)


def joint5_insertion_target(q_current: np.ndarray, joint5_value: float) -> np.ndarray:
    q = np.asarray(q_current, dtype=float).copy()
    q[4] = float(joint5_value)
    q[5] = FIXED_Q6
    return q


def joint5_screwing_target(q_current: np.ndarray, joint5_value: float, turns: float) -> np.ndarray:
    q = np.asarray(q_current, dtype=float).copy()
    q[4] = float(joint5_value)
    q[5] = float(q_current[5] + turns * 2.0 * math.pi)
    return q


def move_arm_screwing_compensated(
    api: ArmPlatformAPI,
    q_goal: np.ndarray,
    duration: float,
    grasp_lock: tuple[np.ndarray, np.ndarray],
    target_peg_xy: np.ndarray,
    viewer=None,
) -> None:
    q_start = api.arm_controller.get_q()
    q_goal = np.asarray(q_goal, dtype=float).copy()
    q_goal[2] = q_start[2]
    q_goal[3] = q_start[3]
    target_peg_xy = np.asarray(target_peg_xy, dtype=float)
    steps = max(1, int(duration / api.model.opt.timestep))
    prev_q = q_start.copy()
    final_q = q_start.copy()

    for i in range(steps):
        s = smoothstep5((i + 1) / steps)
        q_nom = q_start + s * (q_goal - q_start)
        q_nom[2] = q_start[2]
        q_nom[3] = q_start[3]
        try:
            q = solve_revolute12_for_locked_object_xy(
                api,
                target_peg_xy,
                q_nom,
                grasp_lock,
                f"screw_{i:03d}",
                verbose=False,
            )
        except RuntimeError:
            q = q_nom.copy()
        q[4] = q_nom[4]
        q[5] = q_nom[5]
        q[2] = q_start[2]
        q[3] = q_start[3]
        qvel = (q - prev_q) / api.model.opt.timestep
        qvel[2] = 0.0
        qvel[3] = 0.0
        command_arm_target(api, q, qvel, lock_joint6=False, joint34_target=q_start[2:4])
        hold_gripper_target(api, CLOSE_CMD)
        apply_grasp_lock(api, grasp_lock)
        prev_q = q.copy()
        final_q = q.copy()
        step(api, viewer)

    command_arm_target(api, final_q, lock_joint6=False, joint34_target=q_start[2:4])
    hold_gripper_target(api, CLOSE_CMD)
    apply_grasp_lock(api, grasp_lock)


def spiral_search(
    api: ArmPlatformAPI,
    q_start: np.ndarray,
    grasp_lock: tuple[np.ndarray, np.ndarray],
    object_from_finger_xy: np.ndarray,
    viewer=None,
) -> tuple[np.ndarray, bool]:
    search_center_xy = peg_center_xy(api)
    q_current = np.asarray(q_start, dtype=float).copy()
    offsets = archimedean_spiral_offsets()
    print(
        "  spiral search: "
        f"start_center={np.round(search_center_xy, 4)} "
        f"hole_center={np.round(HOLE_CENTER[:2], 4)} "
        f"points={len(offsets)} success_radius={SPIRAL_SUCCESS_RADIUS:.4f}"
    )

    for idx, offset in enumerate(offsets):
        target_peg_xy = search_center_xy + offset
        target_finger_xy = target_peg_xy - object_from_finger_xy
        try:
            q_xy = solve_revolute12_xy(api, target_finger_xy, q_current, f"spiral_{idx:03d}", verbose=False)
        except RuntimeError:
            continue
        q_next = joint12_only_target(q_current, q_xy)
        q_next[4] = q_start[4]
        move_arm(
            api,
            q_next,
            SPIRAL_STEP_DURATION,
            viewer,
            gripper_closed=True,
            grasp_lock=grasp_lock,
            stop_on_object_collision=True,
        )
        q_current = api.arm_controller.get_q().copy()
        center_xy = peg_center_xy(api)
        center_err = float(np.linalg.norm(center_xy - HOLE_CENTER[:2]))
        if idx % 12 == 0 or center_err <= SPIRAL_SUCCESS_RADIUS:
            pos, _ = object_pose(api)
            print(
                f"    spiral idx={idx:03d} r={np.linalg.norm(offset):.4f} "
                f"center_err={center_err:.4f} z={pos[2]:.4f}"
            )
        if center_err <= SPIRAL_SUCCESS_RADIUS:
            print(f"  spiral found hole projection at idx={idx}, center_err={center_err:.4f}")
            return q_current, True

    print("  spiral search reached max radius without exact projection containment")
    return q_current, False


def object_pose(api: ArmPlatformAPI) -> tuple[np.ndarray, np.ndarray]:
    joint_id = name_id(api.model, mujoco.mjtObj.mjOBJ_JOINT, "grasp_object_freejoint")
    qadr = api.model.jnt_qposadr[joint_id]
    return api.data.qpos[qadr:qadr + 3].copy(), api.data.qpos[qadr + 3:qadr + 7].copy()


def count_contacts(api: ArmPlatformAPI, candidates: set[str]) -> int:
    count = 0
    for i in range(api.data.ncon):
        con = api.data.contact[i]
        names = {
            mujoco.mj_id2name(api.model, mujoco.mjtObj.mjOBJ_GEOM, con.geom1) or "",
            mujoco.mj_id2name(api.model, mujoco.mjtObj.mjOBJ_GEOM, con.geom2) or "",
        }
        if "grasp_object_collision" in names and names.intersection(candidates):
            count += 1
    return count


def count_bad_robot_object_contacts(api: ArmPlatformAPI) -> int:
    allowed = {"link6_visual", "fin1_visual", "fin2_visual", "fin1_pad_collision", "fin2_pad_collision"}
    count = 0
    for i in range(api.data.ncon):
        con = api.data.contact[i]
        names = {
            mujoco.mj_id2name(api.model, mujoco.mjtObj.mjOBJ_GEOM, con.geom1) or "",
            mujoco.mj_id2name(api.model, mujoco.mjtObj.mjOBJ_GEOM, con.geom2) or "",
        }
        if "grasp_object_collision" in names:
            other = next((name for name in names if name != "grasp_object_collision"), "")
            if other.endswith("_visual") and other not in allowed:
                count += 1
    return count


def max_contact_force(api: ArmPlatformAPI) -> float:
    force = np.zeros(6, dtype=float)
    max_force = 0.0
    for i in range(api.data.ncon):
        mujoco.mj_contactForce(api.model, api.data, i, force)
        max_force = max(max_force, float(np.linalg.norm(force[:3])))
    return max_force


def run_demo(headless: bool = False, seed: int | None = None) -> None:
    global OBJECT_POS

    xml_path = create_demo_scene()
    api = ArmPlatformAPI(xml_path, CONFIG_PATH)
    api.reset()
    rng = np.random.default_rng(seed)
    OBJECT_POS = compute_object_spawn(api, rng)
    reset_object(api)
    set_arm_qpos(api, HOME_Q)
    set_gripper_qpos(api, OPEN_CMD)
    mujoco.mj_forward(api.model, api.data)

    q_grasp = HOME_Q.copy()
    grasp_target_z = OBJECT_POS[2] + GRASP_HEIGHT_ABOVE_CENTER
    q_grasp[4] = solve_joint5_for_finger_z(api, q_grasp, grasp_target_z)
    q_grasp = solve_revolute12_xy(api, OBJECT_POS[:2], q_grasp, "object_xy")
    q_hover = joint12_only_target(HOME_Q, q_grasp)
    q_descend = joint5_only_target(q_hover, q_grasp[4])
    q_lift = joint5_only_target(q_descend, LIFT_JOINT5)

    set_arm_qpos(api, HOME_Q)
    set_gripper_qpos(api, OPEN_CMD)
    reset_object(api)

    viewer_cm = None
    viewer = None
    if not headless:
        from mujoco import viewer as mujoco_viewer

        viewer_cm = mujoco_viewer.launch_passive(api.model, api.data)
        viewer = viewer_cm.__enter__()
        viewer.cam.distance = 1.45
        viewer.cam.azimuth = -50
        viewer.cam.elevation = -22
        viewer.cam.lookat[:] = np.array([-0.34, -0.75, 0.28], dtype=float)

    try:
        print("Phase: open gripper to maximum")
        set_gripper_qpos(api, OPEN_CMD)
        hold_current(api, 0.20, viewer)
        print("Phase: settle on raised table")
        hold_current(api, 0.35, viewer)
        print("Phase: align gripper center to object center using joint1/joint2 only")
        print(f"  object q1/q2 target: {np.round(q_hover[:2], 4)}")
        print(f"  object spawn: {np.round(OBJECT_POS, 4)} grasp_joint5={q_descend[4]:.4f}")
        move_arm(api, q_hover, 1.00, viewer)
        print("Phase: descend to grasp height using joint5 only")
        move_arm(api, q_descend, 0.65, viewer)
        print("Phase: close gripper physically")
        api.set_gripper(CLOSE_CMD)
        hold_current(api, 0.60, viewer)
        set_gripper_qpos(api, CLOSE_CMD)
        hold_current(api, 0.50, viewer, gripper_closed=True)
        grasp_lock = capture_grasp_lock(api)
        print("Phase: lift with joint5")
        move_arm(api, q_lift, 1.00, viewer, gripper_closed=True, grasp_lock=grasp_lock)
        saved_qpos = api.data.qpos.copy()
        saved_qvel = api.data.qvel.copy()
        lifted_object_pos, _ = object_pose(api)
        object_from_finger_xy = lifted_object_pos[:2] - finger_midpoint(api)[:2]
        search_start_peg_xy = HOLE_CENTER[:2] + SEARCH_ALIGN_ERROR
        search_start_finger_xy = search_start_peg_xy - object_from_finger_xy
        q_search_xy = solve_revolute12_xy(api, search_start_finger_xy, q_lift, "search_start_xy")
        q_search_xy = joint12_only_target(q_lift, q_search_xy)
        print(
            "  search start intentionally offset: "
            f"peg_xy={np.round(search_start_peg_xy, 4)} "
            f"offset={np.round(SEARCH_ALIGN_ERROR, 4)} "
            f"q1/q2={np.round(q_search_xy[:2], 4)}"
        )
        hole_top_z = TABLE_TOP_Z + HOLE_DEPTH
        target_contact_object_z = hole_top_z + CYL_HALF_LENGTH + CONTACT_GAP
        raw_contact_joint5 = solve_joint5_for_object_z(api, q_search_xy, grasp_lock, target_contact_object_z)
        low5, high5 = joint_range(api, "joint5")
        contact_joint5 = float(np.clip(raw_contact_joint5, low5, high5))
        q_contact = joint5_only_target(q_search_xy, contact_joint5)
        target_insert_object_z = hole_top_z + CYL_HALF_LENGTH - INSERT_DEPTH_TARGET
        raw_insert_joint5 = solve_joint5_for_object_z(api, q_search_xy, grasp_lock, target_insert_object_z)
        insert_joint5 = float(np.clip(raw_insert_joint5 - INSERT_JOINT5_MARGIN, low5, high5))
        if insert_joint5 < contact_joint5:
            insert_joint5 = contact_joint5
        predicted_contact_pos = object_pos_with_lock_at_joint5(api, q_search_xy, contact_joint5, grasp_lock)
        predicted_insert_pos = object_pos_with_lock_at_joint5(api, q_search_xy, insert_joint5, grasp_lock)
        print(
            "  search/insert height plan: "
            f"contact_obj_z={target_contact_object_z:.4f} "
            f"contact_joint5={contact_joint5:.4f} "
            f"target_obj_z={target_insert_object_z:.4f} "
            f"raw_joint5={raw_insert_joint5:.4f} "
            f"safe_joint5={insert_joint5:.4f} "
            f"pred_contact={np.round(predicted_contact_pos, 4)} "
            f"pred_obj={np.round(predicted_insert_pos, 4)}"
        )
        api.data.qpos[:] = saved_qpos
        api.data.qvel[:] = saved_qvel
        set_arm_qpos(api, q_lift)
        hold_gripper_target(api, CLOSE_CMD)
        mujoco.mj_forward(api.model, api.data)
        print("Phase: move to offset search start using joint1/joint2 only")
        move_arm(api, q_search_xy, 1.20, viewer, gripper_closed=True, grasp_lock=grasp_lock)
        print("Phase: lower to constant contact height using joint5 only")
        clear_fsc_force(api)
        move_arm(api, q_contact, 0.45, viewer, gripper_closed=True, grasp_lock=grasp_lock, stop_on_object_collision=True)
        print("Phase: Archimedean spiral search using joint1/joint2 only")
        set_fsc_down_force(api, CONTACT_DOWN_FORCE)
        q_found, found = spiral_search(api, api.arm_controller.get_q(), grasp_lock, object_from_finger_xy, viewer)
        print("Phase: final insertion using joint5 with compensated joint6 screwing")
        if found:
            target_insert_object_z = hole_top_z + CYL_HALF_LENGTH - INSERT_DEPTH_TARGET
            raw_insert_joint5 = solve_joint5_for_object_z(api, q_found, grasp_lock, target_insert_object_z)
            insert_joint5 = float(np.clip(raw_insert_joint5 - INSERT_JOINT5_MARGIN, low5, high5))
            if insert_joint5 < q_found[4]:
                insert_joint5 = q_found[4]
            q_insert = joint5_screwing_target(q_found, insert_joint5, SCREW_TURNS)
            print(
                "  recomputed insertion height and compensated screwing: "
                f"current_joint5={q_found[4]:.4f} raw_joint5={raw_insert_joint5:.4f} "
                f"safe_joint5={insert_joint5:.4f} "
                f"joint6_turns={SCREW_TURNS:.1f}"
            )
            set_fsc_down_force(api, INSERT_DOWN_FORCE)
            move_arm_screwing_compensated(
                api,
                q_insert,
                SCREW_INSERT_DURATION,
                grasp_lock=grasp_lock,
                target_peg_xy=HOLE_CENTER[:2],
                viewer=viewer,
            )
            clear_fsc_force(api)
        else:
            print("  final insertion skipped because spiral did not find containment")
            clear_fsc_force(api)
        hold_current(
            api,
            0.80,
            viewer,
            gripper_closed=True,
            grasp_lock=grasp_lock,
            lock_joint6=False,
            decouple_screw_position=False,
            position_joint6=q_found[5] if found else None,
            preserve_joint34=found,
        )

        pos, quat = object_pose(api)
        inserted_depth = peg_inserted_depth(pos)
        hole_names = {f"hole_wall_{i:02d}" for i in range(96)} | {
            "hole_block_left",
            "hole_block_right",
            "hole_block_front",
            "hole_block_back",
        }
        print(f"Final object pos: {np.round(pos, 4)}")
        print(f"Final object quat: {np.round(quat, 4)}")
        print(f"Peg inserted depth: {inserted_depth:.4f} m / hole depth {HOLE_DEPTH:.4f} m")
        print(f"Finger contacts: {count_contacts(api, {'fin1_visual', 'fin2_visual', 'fin1_pad_collision', 'fin2_pad_collision'})}")
        print(f"Hole/block contacts: {count_contacts(api, hole_names)}")
        print(f"Max instantaneous contact force: {max_contact_force(api):.2f} N")
        print(f"Arm q: {np.round(api.arm_controller.get_q(), 4)}")
    finally:
        if viewer_cm is not None:
            viewer_cm.__exit__(None, None, None)
        try:
            xml_path.unlink(missing_ok=True)
        except OSError:
            pass


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()
    run_demo(headless=args.headless, seed=args.seed)


if __name__ == "__main__":
    main()
