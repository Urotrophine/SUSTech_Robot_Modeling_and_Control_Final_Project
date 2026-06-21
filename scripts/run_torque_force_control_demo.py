# -*- coding: utf-8 -*-
"""
scripts/run_torque_force_control_demo.py

Stage 3 torque-level force-control demo.

Compared with the admittance demo:
    Previous demo:
        force error -> joint4 position command q4_cmd
        low-level actuator = position actuator

    This demo:
        force error -> Cartesian task force F_task
        tau = qfrc_bias + joint PD + J^T F_task
        low-level actuator = motor actuator

This is closer to real robot motor torque control.

Run:
    python scripts/run_torque_force_control_demo.py

Useful tuning:
    python scripts/run_torque_force_control_demo.py --target-force 5
    python scripts/run_torque_force_control_demo.py --force-gain 0.8
    python scripts/run_torque_force_control_demo.py --kp-scale 0.7 --kd-scale 1.2
    python scripts/run_torque_force_control_demo.py --no-grasp-lock
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
from control.torque_force_controller import TorqueHybridForceController
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


def set_q_arm_kinematic_model(model, data, q_arm: np.ndarray, joint_names=("joint1", "joint2", "joint3", "joint4")) -> None:
    for value, name in zip(q_arm, joint_names):
        jid = get_id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        data.qpos[model.jnt_qposadr[jid]] = float(value)
        data.qvel[model.jnt_dofadr[jid]] = 0.0
    mujoco.mj_forward(model, data)


def set_q_arm_kinematic_api(api: ArmPlatformAPI, q_arm: np.ndarray) -> None:
    api.data.qpos[api.kin.qpos_idx] = q_arm
    api.data.qvel[:] = 0.0
    mujoco.mj_forward(api.model, api.data)


def get_q_arm(model, data, joint_names=("joint1", "joint2", "joint3", "joint4")) -> np.ndarray:
    values = []
    for name in joint_names:
        jid = get_id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        values.append(data.qpos[model.jnt_qposadr[jid]])
    return np.asarray(values, dtype=float)


def get_qvel_arm(model, data, joint_names=("joint1", "joint2", "joint3", "joint4")) -> np.ndarray:
    values = []
    for name in joint_names:
        jid = get_id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        values.append(data.qvel[model.jnt_dofadr[jid]])
    return np.asarray(values, dtype=float)


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


def replace_arm_position_actuators_with_motors(root: ET.Element) -> None:
    actuator = root.find("actuator")
    if actuator is None:
        actuator = ET.SubElement(root, "actuator")

    # Remove old position actuators for arm joints. Keep gripper_opening as position actuator.
    for child in list(actuator):
        joint = child.attrib.get("joint", "")
        name = child.attrib.get("name", "")
        if joint in ("joint1", "joint2", "joint3", "joint4") or name in (
            "joint1_pos",
            "joint2_pos",
            "joint3_pos",
            "joint4_pos",
        ):
            actuator.remove(child)

    # Motor torque/force limits. Hinge joints: Nm. Slide joint: N.
    motors = [
        ("joint1_motor", "joint1", "-80 80"),
        ("joint2_motor", "joint2", "-80 80"),
        ("joint3_motor", "joint3", "-60 60"),
        ("joint4_motor", "joint4", "-300 300"),
    ]

    # Insert motors before gripper position actuator for clear ordering.
    for name, joint, ctrlrange in motors:
        ET.SubElement(
            actuator,
            "motor",
            {
                "name": name,
                "joint": joint,
                "gear": "1",
                "ctrllimited": "true",
                "ctrlrange": ctrlrange,
            },
        )


def write_torque_force_scene(
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
    root.set("model", "torque_level_force_control_scene")

    option = root.find("option")
    if option is not None:
        option.set("integrator", "implicitfast")
        option.set("timestep", "0.002")

    remove_extra_finger_collision(root)
    patch_mesh_contacts(root)
    replace_arm_position_actuators_with_motors(root)

    actuator = root.find("actuator")
    if actuator is not None:
        for act in actuator:
            if act.attrib.get("name", "") == "gripper_opening":
                act.set("kp", "250")
                act.set("kv", "30")
                act.set("forcelimited", "true")
                act.set("forcerange", "-200 200")

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


def apply_grasp_lock(model, data, object_offset_from_mid: np.ndarray) -> None:
    mid = finger_midpoint(model, data)
    new_pos = mid + object_offset_from_mid
    qadr = qpos_addr_freejoint(model, "grasp_object_freejoint")
    data.qpos[qadr:qadr + 3] = new_pos
    data.qpos[qadr + 3:qadr + 7] = np.array([1.0, 0.0, 0.0, 0.0])
    mujoco.mj_forward(model, data)


def clamp_joint4(api: ArmPlatformAPI, d4: float) -> float:
    low, high = api.arm_controller.limits()
    return float(np.clip(d4, low[3], high[3]))


def finger_mid_for_q(api: ArmPlatformAPI, q_arm: np.ndarray) -> np.ndarray:
    old = api.get_state().q_arm.copy()
    set_q_arm_kinematic_api(api, q_arm)
    mid = finger_midpoint(api.model, api.data)
    set_q_arm_kinematic_api(api, old)
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

                    if abs(q_contact[3] - q_pre[3]) < 0.65 * approach_stroke:
                        continue

                    ratio, xy, z, disp = vertical_ratio_for_motion(api, q_pre, q_contact)

                    if disp[2] >= -0.02:
                        continue

                    score = ratio - 0.5 * xy + 0.05 * z
                    if best is None or score > best[0]:
                        best = (score, q_pre.copy(), q_contact.copy(), down_sign, ratio, xy, z, disp.copy())

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


def set_gripper_command(model, data, close_cmd: float) -> None:
    aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "gripper_opening")
    if aid >= 0:
        data.ctrl[aid] = float(close_cmd)


def drive_to_joint_target_torque(
    model,
    data,
    viewer,
    controller: TorqueHybridForceController,
    q_start: np.ndarray,
    q_goal: np.ndarray,
    duration: float,
    close_cmd: float,
    lock_offset: np.ndarray | None,
    use_lock: bool,
):
    steps = max(1, int(duration / model.opt.timestep))
    for k in range(steps):
        if not viewer.is_running():
            return

        u = k / max(1, steps - 1)
        s = 10*u**3 - 15*u**4 + 6*u**5
        q_ref = q_start + s * (q_goal - q_start)

        set_gripper_command(model, data, close_cmd)
        controller.compute(
            q_ref=q_ref,
            measured_force=0.0,
            down_axis_world=np.array([0.0, 0.0, -1.0]),
            enable_force_term=False,
            dt=model.opt.timestep,
        )
        mujoco.mj_step(model, data)

        if use_lock and lock_offset is not None:
            apply_grasp_lock(model, data, lock_offset)

        viewer.sync()
        time.sleep(max(0.0, model.opt.timestep))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", type=str, default="models/robot_with_gripper.xml")
    parser.add_argument("--target-force", type=float, default=5.0)
    parser.add_argument("--force-gain", type=float, default=1.0)
    parser.add_argument("--max-task-force", type=float, default=60.0)
    parser.add_argument("--force-integral-gain", type=float, default=2.0)
    parser.add_argument("--force-integral-limit", type=float, default=25.0)
    parser.add_argument("--kp-scale", type=float, default=1.0)
    parser.add_argument("--kd-scale", type=float, default=1.0)
    parser.add_argument("--approach-stroke", type=float, default=0.10)
    parser.add_argument("--force-margin", type=float, default=0.045)
    parser.add_argument("--initial-penetration", type=float, default=0.0015)
    parser.add_argument("--control-time", type=float, default=8.0)
    parser.add_argument("--samples-per-joint", type=int, default=9)
    parser.add_argument("--plate-stiffness", type=float, default=350.0)
    parser.add_argument("--filter-alpha", type=float, default=0.15)
    parser.add_argument("--deadband", type=float, default=0.10)
    parser.add_argument("--no-grasp-lock", action="store_true")
    args = parser.parse_args()

    base_model = (PROJECT_ROOT / args.base_model).resolve()
    log_dir = PROJECT_ROOT / "logs"
    log_dir.mkdir(exist_ok=True)

    temp_scene = PROJECT_ROOT / "models" / "_torque_force_control_scene.xml"
    csv_path = log_dir / "torque_force_control_log.csv"
    report_path = log_dir / "torque_force_control_report.md"

    # Use original position-actuated model only for geometry planning and pose search.
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

    down_axis_world = disp / max(np.linalg.norm(disp), 1e-12)

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

    write_torque_force_scene(
        base_model=base_model,
        output_scene=temp_scene,
        shaft_pos_at_pregrasp=shaft_pos_at_pregrasp,
        force_plate_top_z=force_plate_top_z,
        shaft_radius=shaft_radius,
        shaft_half_length=shaft_half_length,
        plate_stiffness=args.plate_stiffness,
    )

    model = mujoco.MjModel.from_xml_path(str(temp_scene))
    data = mujoco.MjData(model)

    force_monitor = ContactForceMonitor(
        model,
        object_geom_names=["grasp_object_collision"],
        environment_geom_names=["force_plate_geom"],
    )

    kp = np.array([120.0, 120.0, 80.0, 1200.0]) * args.kp_scale
    kd = np.array([18.0, 18.0, 12.0, 90.0]) * args.kd_scale

    controller = TorqueHybridForceController(
        model,
        data,
        kp=kp,
        kd=kd,
        target_force=args.target_force,
        force_gain=args.force_gain,
        force_integral_gain=args.force_integral_gain,
        force_integral_limit=args.force_integral_limit,
        max_task_force=args.max_task_force,
        filter_alpha=args.filter_alpha,
        deadband=args.deadband,
    )

    print("=" * 80)
    print("Stage 3 Torque-Level Hybrid Force-Control Demo")
    print("=" * 80)
    print(f"Base model: {base_model}")
    print(f"Motor scene: {temp_scene}")
    print("Arm actuator type: motor")
    print("Control law: tau = qfrc_bias + Kp(q_ref-q) - Kd*dq + J^T F_task")
    print("Force loop: F_task = Kf*e_F + Ki*integral(e_F dt)")
    print(f"target_force = {args.target_force:.3f} N")
    print(f"force_gain = {args.force_gain:.3f}")
    print(f"force_integral_gain = {args.force_integral_gain:.3f}")
    print(f"force_integral_limit = {args.force_integral_limit:.3f}")
    print(f"max_task_force = {args.max_task_force:.3f} N")
    print(f"vertical_ratio = {vertical_ratio:.6f}")
    print(f"q_pre     = {array_str(q_pre)}")
    print(f"q_contact = {array_str(q_contact)}")
    print(f"down_axis_world = {array_str(down_axis_world)}")
    print(f"force_plate_top_z = {force_plate_top_z:.6f}")
    print("=" * 80)

    force_history = []
    filtered_history = []
    tau_history = []

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "time",
            "phase",
            "q1",
            "q2",
            "q3",
            "q4",
            "dq1",
            "dq2",
            "dq3",
            "dq4",
            "force_normal",
            "force_filtered",
            "force_error",
            "force_integral",
            "task_force_saturated",
            "task_fx",
            "task_fy",
            "task_fz",
            "tau1",
            "tau2",
            "tau3",
            "tau4",
            "tau_bias1",
            "tau_bias2",
            "tau_bias3",
            "tau_bias4",
            "tau_pd1",
            "tau_pd2",
            "tau_pd3",
            "tau_pd4",
            "tau_force1",
            "tau_force2",
            "tau_force3",
            "tau_force4",
            "saturated",
            "contact_count",
            "shaft_x",
            "shaft_y",
            "shaft_z",
        ])

        with mujoco.viewer.launch_passive(model, data) as viewer:
            viewer.cam.distance = 1.2

            set_q_arm_kinematic_model(model, data, q_pre)
            set_gripper_command(model, data, close_cmd)
            set_freejoint_pose(model, data, "grasp_object_freejoint", shaft_pos_at_pregrasp)
            lock_offset = object_body_pos(model, data) - finger_midpoint(model, data)
            mujoco.mj_forward(model, data)

            controller.reset_filter(0.0)

            # Initial hold.
            for _ in range(int(1.0 / model.opt.timestep)):
                if not viewer.is_running():
                    return
                set_gripper_command(model, data, close_cmd)
                controller.compute(q_ref=q_pre, measured_force=0.0, down_axis_world=down_axis_world, enable_force_term=False)
                mujoco.mj_step(model, data)
                if not args.no_grasp_lock:
                    apply_grasp_lock(model, data, lock_offset)
                viewer.sync()
                time.sleep(max(0.0, model.opt.timestep))

            # Torque-driven approach.
            drive_to_joint_target_torque(
                model=model,
                data=data,
                viewer=viewer,
                controller=controller,
                q_start=q_pre,
                q_goal=q_contact,
                duration=3.0,
                close_cmd=close_cmd,
                lock_offset=lock_offset,
                use_lock=(not args.no_grasp_lock),
            )

            initial_force = force_monitor.read(data).normal_force
            controller.reset_filter(initial_force)

            start_time = float(data.time)
            last_print_bucket = -1

            while viewer.is_running() and (data.time - start_time) < args.control_time:
                reading = force_monitor.read(data)
                set_gripper_command(model, data, close_cmd)

                ctrl_state = controller.compute(
                    q_ref=q_contact,
                    measured_force=reading.normal_force,
                    down_axis_world=down_axis_world,
                    enable_force_term=True,
                    dt=model.opt.timestep,
                )

                mujoco.mj_step(model, data)

                if not args.no_grasp_lock:
                    apply_grasp_lock(model, data, lock_offset)

                viewer.sync()

                q = get_q_arm(model, data)
                dq = get_qvel_arm(model, data)
                shaft_pos = object_body_pos(model, data)

                writer.writerow([
                    data.time,
                    "torque_hybrid_force_control",
                    q[0],
                    q[1],
                    q[2],
                    q[3],
                    dq[0],
                    dq[1],
                    dq[2],
                    dq[3],
                    reading.normal_force,
                    ctrl_state.force_filtered,
                    ctrl_state.force_error,
                    ctrl_state.force_integral,
                    ctrl_state.task_force_saturated,
                    ctrl_state.task_force_cmd[0],
                    ctrl_state.task_force_cmd[1],
                    ctrl_state.task_force_cmd[2],
                    ctrl_state.tau[0],
                    ctrl_state.tau[1],
                    ctrl_state.tau[2],
                    ctrl_state.tau[3],
                    ctrl_state.tau_bias[0],
                    ctrl_state.tau_bias[1],
                    ctrl_state.tau_bias[2],
                    ctrl_state.tau_bias[3],
                    ctrl_state.tau_pd[0],
                    ctrl_state.tau_pd[1],
                    ctrl_state.tau_pd[2],
                    ctrl_state.tau_pd[3],
                    ctrl_state.tau_force[0],
                    ctrl_state.tau_force[1],
                    ctrl_state.tau_force[2],
                    ctrl_state.tau_force[3],
                    ctrl_state.saturated,
                    reading.contact_count,
                    shaft_pos[0],
                    shaft_pos[1],
                    shaft_pos[2],
                ])

                force_history.append(float(reading.normal_force))
                filtered_history.append(float(ctrl_state.force_filtered))
                tau_history.append(ctrl_state.tau.copy())

                print_bucket = int((data.time - start_time) * 5)
                if print_bucket != last_print_bucket:
                    last_print_bucket = print_bucket
                    print(
                        f"t={data.time - start_time:5.2f}s, "
                        f"F={reading.normal_force:7.3f} N, "
                        f"F_filt={ctrl_state.force_filtered:7.3f} N, "
                        f"err={ctrl_state.force_error:7.3f} N, "
                        f"int={ctrl_state.force_integral:7.3f}, "
                        f"Ftask={array_str(ctrl_state.task_force_cmd, precision=3)}, "
                        f"tau={array_str(ctrl_state.tau, precision=3)}, "
                        f"contacts={reading.contact_count}, "
                        f"sat={ctrl_state.saturated}, Fsat={ctrl_state.task_force_saturated}"
                    )

                time.sleep(max(0.0, model.opt.timestep))

            # Final hold with torque controller and grasp lock.
            final_lock_offset = object_body_pos(model, data) - finger_midpoint(model, data)
            for _ in range(int(2.0 / model.opt.timestep)):
                if not viewer.is_running():
                    break
                reading = force_monitor.read(data)
                set_gripper_command(model, data, close_cmd)
                controller.compute(
                    q_ref=q_contact,
                    measured_force=reading.normal_force,
                    down_axis_world=down_axis_world,
                    enable_force_term=True,
                    dt=model.opt.timestep,
                )
                mujoco.mj_step(model, data)
                if not args.no_grasp_lock:
                    apply_grasp_lock(model, data, final_lock_offset)
                viewer.sync()
                time.sleep(max(0.0, model.opt.timestep))

    if filtered_history:
        final_force = float(filtered_history[-1])
        mean_tail_force = float(np.mean(filtered_history[max(0, len(filtered_history) - 100):]))
        mean_tail_error = float(args.target_force - mean_tail_force)
        max_abs_tau = np.max(np.abs(np.asarray(tau_history)), axis=0).tolist()
    else:
        final_force = 0.0
        mean_tail_force = 0.0
        mean_tail_error = args.target_force
        max_abs_tau = [0.0, 0.0, 0.0, 0.0]

    passed = abs(mean_tail_error) <= max(1.5, 0.35 * args.target_force)

    lines = []
    lines.append("# Stage 3 torque-level force-control report\n")
    lines.append(f"- Controller: `tau = qfrc_bias + Kp(q_ref-q) - Kd*dq + J^T F_task`")
    lines.append(f"- Force loop: `F_task = Kf*e_F + Ki*integral(e_F dt)`")
    lines.append(f"- force_integral_gain: `{args.force_integral_gain:.3f}`")
    lines.append(f"- Target force: `{args.target_force:.3f}` N")
    lines.append(f"- Final filtered force: `{final_force:.3f}` N")
    lines.append(f"- Tail mean filtered force: `{mean_tail_force:.3f}` N")
    lines.append(f"- Tail mean error: `{mean_tail_error:.3f}` N")
    lines.append(f"- vertical_ratio: `{vertical_ratio:.6f}`")
    lines.append(f"- q_pre: `{array_str(q_pre)}`")
    lines.append(f"- q_contact: `{array_str(q_contact)}`")
    lines.append(f"- max_abs_tau: `{array_str(np.asarray(max_abs_tau))}`")
    lines.append(f"- Result: **{'PASSED' if passed else 'FAILED'}**\n")
    lines.append("## Notes\n")
    lines.append("This demo uses motor actuators for joint1--joint4. The gripper remains a position actuator.")
    lines.append("The force term is generated in Cartesian space and mapped to joint torques through `J^T F`.")
    lines.append("The script still uses logical grasp lock after the shaft is grasped, so the object attachment is a task-level constraint.")
    report_path.write_text("\n".join(lines), encoding="utf-8")

    print()
    print("=" * 80)
    print("Stage 3 torque-level force-control demo finished.")
    print(f"Final filtered force: {final_force:.3f} N")
    print(f"Tail mean filtered force: {mean_tail_force:.3f} N")
    print(f"Tail mean error: {mean_tail_error:.3f} N")
    print(f"Max |tau|: {array_str(np.asarray(max_abs_tau))}")
    print(f"Report: {report_path}")
    print(f"CSV log: {csv_path}")
    print(f"Result: {'PASSED' if passed else 'FAILED'}")
    print("=" * 80)


if __name__ == "__main__":
    main()
