# -*- coding: utf-8 -*-
"""
scripts/run_joint4_admittance_force_control_demo.py

Joint4 admittance force-control demo v2.

Fix compared with the previous version:
    The previous version started force control with joint4 already at its
    mechanical limit q4 = 0.2. Therefore the controller wanted to push down,
    but q4_cmd was saturated and the force stayed around 0.95 N.

    This version reserves a downward control margin before contact. By default,
    force control starts at q_contact = q_down_limit - force_margin, so joint4
    can still move further downward during the admittance loop.

Run from project root:
    python scripts/run_joint4_admittance_force_control_demo.py

Useful tuning:
    python scripts/run_joint4_admittance_force_control_demo.py --target-force 5
    python scripts/run_joint4_admittance_force_control_demo.py --force-margin 0.05
    python scripts/run_joint4_admittance_force_control_demo.py --target-force 8 --k-force 0.002
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

import mujoco
import mujoco.viewer

from api.arm_platform_api import ArmPlatformAPI
from control.admittance_controller import Joint4AdmittanceController
from simulation.contact_force_monitor import ContactForceMonitor


def array_str(x: np.ndarray, precision: int = 6) -> str:
    return np.array2string(np.asarray(x), precision=precision, suppress_small=False)


def get_id(model, objtype, name: str) -> int:
    oid = mujoco.mj_name2id(model, objtype, name)
    if oid < 0:
        raise ValueError(f"Object not found: {name}")
    return oid


def get_site_pos(model, data, name: str) -> np.ndarray:
    sid = get_id(model, mujoco.mjtObj.mjOBJ_SITE, name)
    mujoco.mj_forward(model, data)
    return data.site_xpos[sid].copy()


def set_q_arm_kinematic(api: ArmPlatformAPI, q_arm: np.ndarray) -> None:
    api.data.qpos[api.kin.qpos_idx] = q_arm
    api.data.qvel[:] = 0.0
    mujoco.mj_forward(api.model, api.data)


def finger_midpoint(model, data) -> np.ndarray:
    left = get_site_pos(model, data, "left_finger_tip_site")
    right = get_site_pos(model, data, "right_finger_tip_site")
    return 0.5 * (left + right)


def finger_distance(model, data) -> float:
    left = get_site_pos(model, data, "left_finger_tip_site")
    right = get_site_pos(model, data, "right_finger_tip_site")
    return float(np.linalg.norm(left - right))


def set_slide_pair_direct(model, data, joint5_value: float) -> None:
    j5 = get_id(model, mujoco.mjtObj.mjOBJ_JOINT, "joint5")
    j6 = get_id(model, mujoco.mjtObj.mjOBJ_JOINT, "joint6")
    data.qpos[model.jnt_qposadr[j5]] = joint5_value
    data.qpos[model.jnt_qposadr[j6]] = -joint5_value
    mujoco.mj_forward(model, data)


def detect_open_close_commands(api: ArmPlatformAPI, a: float = 0.03) -> Tuple[float, float, float, float]:
    old_qpos = api.data.qpos.copy()
    old_qvel = api.data.qvel.copy()

    set_slide_pair_direct(api.model, api.data, -a)
    d_minus = finger_distance(api.model, api.data)

    set_slide_pair_direct(api.model, api.data, +a)
    d_plus = finger_distance(api.model, api.data)

    api.data.qpos[:] = old_qpos
    api.data.qvel[:] = old_qvel
    mujoco.mj_forward(api.model, api.data)

    if d_minus >= d_plus:
        return -a, +a, d_minus, d_plus
    return +a, -a, d_plus, d_minus


def remove_extra_finger_collision(root: ET.Element) -> None:
    for parent in root.iter():
        for child in list(parent):
            if child.tag == "geom" and child.attrib.get("name", "") in (
                "fin1_finger_collision",
                "fin2_finger_collision",
            ):
                parent.remove(child)


def patch_mesh_contacts(root: ET.Element) -> None:
    contact_visuals = {"fin1_visual", "fin2_visual", "link4_visual"}

    for geom in root.findall(".//geom"):
        name = geom.attrib.get("name", "")

        if name in contact_visuals:
            geom.set("contype", "1")
            geom.set("conaffinity", "1")
            geom.set("friction", "4.0 0.08 0.008")
            geom.set("condim", "4")
            geom.set("density", "0")
        elif name.endswith("_visual"):
            geom.set("contype", "0")
            geom.set("conaffinity", "0")


def write_force_control_scene(
    base_model: Path,
    output_scene: Path,
    shaft_pos_at_pregrasp: np.ndarray,
    force_plate_top_z: float,
    shaft_radius: float,
    shaft_half_length: float,
    plate_stiffness: float,
) -> None:
    tree = ET.parse(base_model)
    root = tree.getroot()
    root.set("model", "joint4_admittance_force_control_scene_v2")

    remove_extra_finger_collision(root)
    patch_mesh_contacts(root)

    actuator = root.find("actuator")
    if actuator is not None:
        for act in actuator:
            name = act.attrib.get("name", "")
            joint = act.attrib.get("joint", "")
            if name in ("joint4_pos", "gripper_opening") or joint in ("joint4", "joint5"):
                act.set("kp", "2200")
                act.set("kv", "120")
                act.set("forcelimited", "true")
                act.set("forcerange", "-1200 1200")

    world = root.find("worldbody")
    if world is None:
        raise RuntimeError("worldbody not found")

    for body_name in ("grasp_object", "force_plate"):
        old = world.find(f".//body[@name='{body_name}']")
        if old is not None:
            world.remove(old)

    shaft_body = ET.SubElement(world, "body", {
        "name": "grasp_object",
        "pos": f"{shaft_pos_at_pregrasp[0]:.10g} {shaft_pos_at_pregrasp[1]:.10g} {shaft_pos_at_pregrasp[2]:.10g}",
    })
    ET.SubElement(shaft_body, "freejoint", {"name": "grasp_object_freejoint"})
    ET.SubElement(shaft_body, "geom", {
        "name": "grasp_object_collision",
        "type": "cylinder",
        "size": f"{shaft_radius:.10g} {shaft_half_length:.10g}",
        "mass": "0.035",
        "rgba": "0.95 0.55 0.15 1",
        "friction": "4.0 0.08 0.008",
        "condim": "4",
    })

    plate_thickness = 0.02
    plate_body = ET.SubElement(world, "body", {
        "name": "force_plate",
        "pos": f"{shaft_pos_at_pregrasp[0]:.10g} {shaft_pos_at_pregrasp[1]:.10g} {force_plate_top_z - plate_thickness / 2.0:.10g}",
    })
    plate_solref_time = max(0.002, min(0.03, 1.0 / max(plate_stiffness, 1.0)))
    ET.SubElement(plate_body, "geom", {
        "name": "force_plate_geom",
        "type": "box",
        "size": "0.13 0.13 0.01",
        "rgba": "0.20 0.30 0.90 1",
        "friction": "3.0 0.05 0.005",
        "condim": "4",
        "solref": f"{plate_solref_time:.6f} 1",
        "solimp": "0.95 0.99 0.001",
    })

    ET.indent(tree, space="  ")
    tree.write(output_scene, encoding="utf-8", xml_declaration=True)


def set_freejoint_pose(model, data, joint_name: str, pos: np.ndarray, quat=None) -> None:
    if quat is None:
        quat = np.array([1.0, 0.0, 0.0, 0.0])
    jid = get_id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    qadr = int(model.jnt_qposadr[jid])
    vadr = int(model.jnt_dofadr[jid])
    data.qpos[qadr:qadr + 3] = pos
    data.qpos[qadr + 3:qadr + 7] = quat
    data.qvel[vadr:vadr + 6] = 0.0
    mujoco.mj_forward(model, data)


def object_body_pos(model, data) -> np.ndarray:
    bid = get_id(model, mujoco.mjtObj.mjOBJ_BODY, "grasp_object")
    mujoco.mj_forward(model, data)
    return data.xpos[bid].copy()


def qpos_addr_freejoint(model, joint_name: str) -> int:
    jid = get_id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    return int(model.jnt_qposadr[jid])


def apply_grasp_lock(api: ArmPlatformAPI, object_offset_from_mid: np.ndarray) -> None:
    mid = finger_midpoint(api.model, api.data)
    new_pos = mid + object_offset_from_mid
    qadr = qpos_addr_freejoint(api.model, "grasp_object_freejoint")
    api.data.qpos[qadr:qadr + 3] = new_pos
    api.data.qpos[qadr + 3:qadr + 7] = np.array([1.0, 0.0, 0.0, 0.0])
    mujoco.mj_forward(api.model, api.data)


def clamp_joint4(api: ArmPlatformAPI, d4: float) -> float:
    low, high = api.arm_controller.limits()
    return float(np.clip(d4, low[3], high[3]))


def finger_mid_for_q(api: ArmPlatformAPI, q_arm: np.ndarray) -> np.ndarray:
    old = api.get_state().q_arm.copy()
    set_q_arm_kinematic(api, q_arm)
    mid = finger_midpoint(api.model, api.data)
    set_q_arm_kinematic(api, old)
    return mid


def vertical_ratio_for_motion(api: ArmPlatformAPI, q_pre: np.ndarray, q_final: np.ndarray):
    mid_pre = finger_mid_for_q(api, q_pre)
    mid_final = finger_mid_for_q(api, q_final)
    disp = mid_final - mid_pre
    norm = float(np.linalg.norm(disp))
    if norm < 1e-12:
        return 0.0, 1e9, 0.0, disp
    xy = float(np.linalg.norm(disp[:2]))
    z = float(abs(disp[2]))
    return z / norm, xy, z, disp


def search_vertical_joint4_force_pose(
    api: ArmPlatformAPI,
    approach_stroke: float,
    force_margin: float,
    samples_per_joint: int = 9,
):
    """Search a near-vertical pose while reserving joint4 travel for force control.

    If moving downward requires increasing q4, q_contact is set to high - force_margin
    instead of high. If moving downward requires decreasing q4, q_contact is set to
    low + force_margin instead of low.
    """
    q0 = api.get_state().q_arm.copy()
    low, high = api.arm_controller.limits()

    q1_vals = np.linspace(max(low[0], q0[0] - 0.8), min(high[0], q0[0] + 0.8), samples_per_joint)
    q2_vals = np.linspace(max(low[1], q0[1] - 0.8), min(high[1], q0[1] + 0.8), samples_per_joint)
    q3_vals = np.linspace(max(low[2], q0[2] - 0.8), min(high[2], q0[2] + 0.8), samples_per_joint)

    best = None

    for q1 in q1_vals:
        for q2 in q2_vals:
            for q3 in q3_vals:
                for down_sign in (+1.0, -1.0):
                    if down_sign > 0:
                        q_contact_d4 = high[3] - force_margin
                    else:
                        q_contact_d4 = low[3] + force_margin

                    q_contact_d4 = clamp_joint4(api, q_contact_d4)
                    q_contact = np.array([q1, q2, q3, q_contact_d4], dtype=float)
                    q_pre = q_contact.copy()
                    q_pre[3] = clamp_joint4(api, q_contact_d4 - down_sign * approach_stroke)

                    actual_stroke = abs(q_contact[3] - q_pre[3])
                    if actual_stroke < 0.65 * approach_stroke:
                        continue

                    ratio, xy, z, disp = vertical_ratio_for_motion(api, q_pre, q_contact)

                    # q_contact should be lower than q_pre in world z.
                    if disp[2] >= -0.02:
                        continue

                    score = ratio - 0.5 * xy + 0.05 * z
                    if best is None or score > best[0]:
                        best = (
                            score,
                            q_pre.copy(),
                            q_contact.copy(),
                            down_sign,
                            ratio,
                            xy,
                            z,
                            disp.copy(),
                        )

    if best is None:
        q_contact = q0.copy()
        q_contact[3] = high[3] - force_margin
        q_pre = q_contact.copy()
        q_pre[3] = clamp_joint4(api, q_contact[3] - approach_stroke)
        down_sign = 1.0
        ratio, xy, z, disp = vertical_ratio_for_motion(api, q_pre, q_contact)
        return q_pre, q_contact, down_sign, ratio, xy, z, disp

    _, q_pre, q_contact, down_sign, ratio, xy, z, disp = best
    return q_pre, q_contact, down_sign, ratio, xy, z, disp


def move_joint4_only(
    api: ArmPlatformAPI,
    viewer,
    q_start: np.ndarray,
    q_goal: np.ndarray,
    duration: float,
    lock_offset: np.ndarray | None = None,
    use_lock: bool = True,
):
    assert np.allclose(q_start[:3], q_goal[:3], atol=1e-10)

    steps = max(1, int(duration / api.model.opt.timestep))
    for k in range(steps):
        if not viewer.is_running():
            return
        u = k / max(1, steps - 1)
        s = 10*u**3 - 15*u**4 + 6*u**5
        q = q_start + s * (q_goal - q_start)

        api.set_arm_target(q)
        api.step()

        if use_lock and lock_offset is not None:
            apply_grasp_lock(api, lock_offset)

        viewer.sync()
        time.sleep(max(0.0, api.model.opt.timestep))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", type=str, default="models/robot_with_gripper.xml")
    parser.add_argument("--target-force", type=float, default=5.0)
    parser.add_argument("--k-force", type=float, default=0.004)
    parser.add_argument("--max-speed", type=float, default=0.015)
    parser.add_argument("--filter-alpha", type=float, default=0.15)
    parser.add_argument("--deadband", type=float, default=0.10)
    parser.add_argument("--approach-stroke", type=float, default=0.10)
    parser.add_argument("--force-margin", type=float, default=0.045, help="reserved joint4 downward travel during force control")
    parser.add_argument("--initial-penetration", type=float, default=0.0015, help="initial shaft/plate penetration at q_contact")
    parser.add_argument("--control-time", type=float, default=8.0)
    parser.add_argument("--samples-per-joint", type=int, default=9)
    parser.add_argument("--plate-stiffness", type=float, default=350.0)
    parser.add_argument("--no-grasp-lock", action="store_true")
    args = parser.parse_args()

    base_model = (PROJECT_ROOT / args.base_model).resolve()
    log_dir = PROJECT_ROOT / "logs"
    log_dir.mkdir(exist_ok=True)

    temp_scene = PROJECT_ROOT / "models" / "_admittance_force_control_scene_v2.xml"
    csv_path = log_dir / "admittance_force_control_log.csv"
    report_path = log_dir / "admittance_force_control_report.md"

    api0 = ArmPlatformAPI(base_model)
    open_cmd, close_cmd, open_dist, close_dist = detect_open_close_commands(api0, a=0.03)

    api0.reset()
    api0.set_gripper(close_cmd)
    for _ in range(int(0.5 / api0.model.opt.timestep)):
        api0.step()

    q_pre, q_contact, down_sign, vertical_ratio, xy_motion, z_motion, disp = search_vertical_joint4_force_pose(
        api0,
        approach_stroke=args.approach_stroke,
        force_margin=args.force_margin,
        samples_per_joint=args.samples_per_joint,
    )

    shaft_radius = min(max(close_dist * 0.5 + 0.006, 0.014), open_dist * 0.42)
    shaft_half_length = 0.075

    mid_pre = finger_mid_for_q(api0, q_pre)
    mid_contact = finger_mid_for_q(api0, q_contact)

    top_overlap = 0.04
    shaft_offset_from_mid = np.array([0.0, 0.0, top_overlap - shaft_half_length], dtype=float)
    shaft_pos_at_pregrasp = mid_pre + shaft_offset_from_mid
    shaft_pos_at_contact = mid_contact + shaft_offset_from_mid

    shaft_bottom_at_contact = shaft_pos_at_contact[2] - shaft_half_length
    force_plate_top_z = shaft_bottom_at_contact + args.initial_penetration

    write_force_control_scene(
        base_model=base_model,
        output_scene=temp_scene,
        shaft_pos_at_pregrasp=shaft_pos_at_pregrasp,
        force_plate_top_z=force_plate_top_z,
        shaft_radius=shaft_radius,
        shaft_half_length=shaft_half_length,
        plate_stiffness=args.plate_stiffness,
    )

    api = ArmPlatformAPI(temp_scene)
    api.reset()

    force_monitor = ContactForceMonitor(
        api.model,
        object_geom_names=["grasp_object_collision"],
        environment_geom_names=["force_plate_geom"],
    )

    low, high = api.arm_controller.limits()

    if down_sign > 0:
        cmd_min = float(low[3])
        cmd_max = float(high[3])
        q_down_limit = high[3]
    else:
        cmd_min = float(low[3])
        cmd_max = float(high[3])
        q_down_limit = low[3]

    admittance = Joint4AdmittanceController(
        initial_command=float(q_contact[3]),
        down_sign=down_sign,
        command_min=cmd_min,
        command_max=cmd_max,
        target_force=args.target_force,
        k_force=args.k_force,
        max_speed=args.max_speed,
        filter_alpha=args.filter_alpha,
        deadband=args.deadband,
    )

    print("=" * 80)
    print("Joint4 Admittance Force-Control Demo v2")
    print("=" * 80)
    print(f"Base model: {base_model}")
    print(f"Temp scene: {temp_scene}")
    print("Control type: admittance outer loop + position actuator inner loop")
    print(f"target_force = {args.target_force:.3f} N")
    print(f"k_force = {args.k_force:.6f} m/(N*s)")
    print(f"max_speed = {args.max_speed:.6f} m/s")
    print(f"vertical_ratio = {vertical_ratio:.6f}")
    print(f"q_pre     = {array_str(q_pre)}")
    print(f"q_contact = {array_str(q_contact)}")
    print(f"down_sign = {down_sign:+.1f}")
    print(f"force_margin = {args.force_margin:.6f}")
    print(f"remaining downward margin = {abs(q_down_limit - q_contact[3]):.6f}")
    print(f"shaft_radius = {shaft_radius:.6f}")
    print(f"force_plate_top_z = {force_plate_top_z:.6f}")
    print("=" * 80)

    force_history = []
    command_history = []
    time_history = []

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "time",
            "phase",
            "q1",
            "q2",
            "q3",
            "q4",
            "q4_cmd",
            "force_normal",
            "force_filtered",
            "force_error",
            "velocity_command",
            "saturated_low",
            "saturated_high",
            "contact_count",
            "shaft_x",
            "shaft_y",
            "shaft_z",
        ])

        with mujoco.viewer.launch_passive(api.model, api.data) as viewer:
            viewer.cam.distance = 1.2

            api.set_gripper(close_cmd)
            set_q_arm_kinematic(api, q_pre)
            api.set_arm_target(q_pre)

            set_freejoint_pose(api.model, api.data, "grasp_object_freejoint", shaft_pos_at_pregrasp)
            lock_offset = object_body_pos(api.model, api.data) - finger_midpoint(api.model, api.data)

            for _ in range(int(1.0 / api.model.opt.timestep)):
                if not viewer.is_running():
                    return
                api.step()
                if not args.no_grasp_lock:
                    apply_grasp_lock(api, lock_offset)
                viewer.sync()
                time.sleep(max(0.0, api.model.opt.timestep))

            move_joint4_only(
                api,
                viewer,
                q_start=q_pre,
                q_goal=q_contact,
                duration=3.0,
                lock_offset=lock_offset,
                use_lock=(not args.no_grasp_lock),
            )

            initial_force = force_monitor.read(api.data).normal_force
            admittance.reset(command=float(q_contact[3]), measured_force=initial_force)

            start_time = float(api.data.time)
            last_time = float(api.data.time)

            while viewer.is_running() and (api.data.time - start_time) < args.control_time:
                t_now = float(api.data.time)
                dt = max(t_now - last_time, api.model.opt.timestep)
                last_time = t_now

                reading = force_monitor.read(api.data)
                ctrl_state = admittance.update(reading.normal_force, dt)

                q_cmd = q_contact.copy()
                q_cmd[3] = ctrl_state.command
                api.set_arm_target(q_cmd)
                api.set_gripper(close_cmd)

                api.step()

                if not args.no_grasp_lock:
                    apply_grasp_lock(api, lock_offset)

                viewer.sync()

                shaft_pos = object_body_pos(api.model, api.data)
                state = api.get_state()

                writer.writerow([
                    api.data.time,
                    "admittance_force_control",
                    state.q_arm[0],
                    state.q_arm[1],
                    state.q_arm[2],
                    state.q_arm[3],
                    ctrl_state.command,
                    reading.normal_force,
                    ctrl_state.force_filtered,
                    ctrl_state.force_error,
                    ctrl_state.velocity_command,
                    ctrl_state.saturated_low,
                    ctrl_state.saturated_high,
                    reading.contact_count,
                    shaft_pos[0],
                    shaft_pos[1],
                    shaft_pos[2],
                ])

                force_history.append(ctrl_state.force_filtered)
                command_history.append(ctrl_state.command)
                time_history.append(api.data.time - start_time)

                if int((api.data.time - start_time) * 5) != int((api.data.time - start_time - api.model.opt.timestep) * 5):
                    print(
                        f"t={api.data.time - start_time:5.2f}s, "
                        f"F={reading.normal_force:7.3f} N, "
                        f"F_filt={ctrl_state.force_filtered:7.3f} N, "
                        f"err={ctrl_state.force_error:7.3f} N, "
                        f"q4_cmd={ctrl_state.command:+.5f}, "
                        f"contacts={reading.contact_count}, "
                        f"sat=({ctrl_state.saturated_low},{ctrl_state.saturated_high})"
                    )

                time.sleep(max(0.0, api.model.opt.timestep))

            final_lock_offset = object_body_pos(api.model, api.data) - finger_midpoint(api.model, api.data)
            for _ in range(int(2.0 / api.model.opt.timestep)):
                if not viewer.is_running():
                    break
                api.step()
                if not args.no_grasp_lock:
                    apply_grasp_lock(api, final_lock_offset)
                viewer.sync()
                time.sleep(max(0.0, api.model.opt.timestep))

    if force_history:
        final_force = float(force_history[-1])
        mean_tail_force = float(np.mean(force_history[max(0, len(force_history) - 100):]))
        final_error = float(args.target_force - final_force)
        mean_tail_error = float(args.target_force - mean_tail_force)
        final_command = float(command_history[-1])
        command_change = float(command_history[-1] - command_history[0])
    else:
        final_force = 0.0
        mean_tail_force = 0.0
        final_error = args.target_force
        mean_tail_error = args.target_force
        final_command = float(q_contact[3])
        command_change = 0.0

    passed = abs(mean_tail_error) <= max(1.0, 0.25 * args.target_force)

    lines = []
    lines.append("# Joint4 导纳力控仿真报告 v2\n")
    lines.append(f"- 控制方式：导纳外环 + 位置执行器内环")
    lines.append(f"- 目标接触力：`{args.target_force:.3f}` N")
    lines.append(f"- 最终滤波力：`{final_force:.3f}` N")
    lines.append(f"- 末段平均滤波力：`{mean_tail_force:.3f}` N")
    lines.append(f"- 末段平均误差：`{mean_tail_error:.3f}` N")
    lines.append(f"- 初始 q_contact：`{q_contact[3]:.6f}`")
    lines.append(f"- 最终 q4_cmd：`{final_command:.6f}`")
    lines.append(f"- q4_cmd 变化量：`{command_change:.6f}`")
    lines.append(f"- force_margin：`{args.force_margin:.6f}`")
    lines.append(f"- vertical_ratio：`{vertical_ratio:.6f}`")
    lines.append(f"- 结果：**{'通过' if passed else '未通过'}**\n")
    lines.append("## 说明\n")
    lines.append("上一版失败的原因是 force-control 开始时 joint4 已经在机械限位 q4=0.2，控制器无法继续下压。")
    lines.append("本版预留了 force_margin，使导纳控制阶段仍有向下运动空间。")
    lines.append("该 demo 不是力矩级控制，而是读取 MuJoCo 接触力并实时修改 joint4 位置目标。")
    report_path.write_text("\n".join(lines), encoding="utf-8")

    print()
    print("=" * 80)
    print("Admittance force-control demo finished.")
    print(f"Final filtered force: {final_force:.3f} N")
    print(f"Tail mean filtered force: {mean_tail_force:.3f} N")
    print(f"Tail mean error: {mean_tail_error:.3f} N")
    print(f"Final q4_cmd: {final_command:.6f}")
    print(f"q4_cmd change: {command_change:.6f}")
    print(f"Report: {report_path}")
    print(f"CSV log: {csv_path}")
    print(f"Result: {'PASSED' if passed else 'FAILED'}")
    print("=" * 80)


if __name__ == "__main__":
    main()
