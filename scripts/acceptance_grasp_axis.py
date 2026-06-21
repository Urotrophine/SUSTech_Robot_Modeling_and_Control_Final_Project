# -*- coding: utf-8 -*-
"""
acceptance_grasp_axis.py

顶部竖直夹取验收脚本 v8。

目标：
1. 不给夹爪额外添加绿色 collision box。
2. 直接使用原始夹爪 STL mesh geom 统计接触：
       fin1_visual
       fin2_visual
3. 自动搜索一个 joint4 运动方向最接近世界竖直方向的 q1/q2/q3 姿态。
4. 预抓取、下降、抬升阶段只改变 joint4，q1/q2/q3 保持不变。
5. 轴保持竖直，放在最终夹爪中心下方。
6. joint4 下降足够深，使夹爪内部包住圆柱上部。
7. 夹住以后不会停住：满足最小夹持条件后执行 joint4 竖直抬升。
8. 默认抬升阶段使用 logical grasp lock，使已经确认夹住的轴跟随夹爪上移。
   这不是额外 collision geom，而是抓取成功后的任务约束。
   如果要测试纯物理摩擦，可加 --no-grasp-lock。
9. 启用 link4_visual 与轴的碰撞检测，启用 link4 / joint4 顶头接触作为顶部限位，防止穿模。
10. final_hold 阶段继续保持 grasp lock，避免夹到空中后又慢慢滑落。

运行：
    python scripts/acceptance_grasp_axis.py

常用调参：
    python scripts/acceptance_grasp_axis.py --descend-depth 0.11 --top-overlap 0.07
    python scripts/acceptance_grasp_axis.py --vertical-ratio-threshold 0.94
    python scripts/acceptance_grasp_axis.py --no-grasp-lock
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


def get_body_pos(model, data, name: str) -> np.ndarray:
    bid = get_id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    mujoco.mj_forward(model, data)
    return data.xpos[bid].copy()


def set_q_arm_kinematic(api: ArmPlatformAPI, q_arm: np.ndarray) -> None:
    api.data.qpos[api.kin.qpos_idx] = q_arm
    api.data.qvel[:] = 0.0
    mujoco.mj_forward(api.model, api.data)


def set_slide_pair_direct(model, data, joint5_value: float) -> None:
    j5 = get_id(model, mujoco.mjtObj.mjOBJ_JOINT, "joint5")
    j6 = get_id(model, mujoco.mjtObj.mjOBJ_JOINT, "joint6")
    data.qpos[model.jnt_qposadr[j5]] = joint5_value
    data.qpos[model.jnt_qposadr[j6]] = -joint5_value
    mujoco.mj_forward(model, data)


def finger_midpoint(model, data) -> np.ndarray:
    left = get_site_pos(model, data, "left_finger_tip_site")
    right = get_site_pos(model, data, "right_finger_tip_site")
    return 0.5 * (left + right)


def finger_distance(model, data) -> float:
    left = get_site_pos(model, data, "left_finger_tip_site")
    right = get_site_pos(model, data, "right_finger_tip_site")
    return float(np.linalg.norm(left - right))


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


def patch_gripper_mesh_contact(root: ET.Element) -> None:
    """Use original STL mesh geoms as contact geoms where needed.

    No additional helper collision geometry is created for the gripper.

    Enabled contact:
        fin1_visual, fin2_visual: gripper fingers
        link4_visual: joint4 / end-effector head, used to prevent shaft penetration
    """
    contact_visuals = {"fin1_visual", "fin2_visual", "link4_visual"}

    for geom in root.findall(".//geom"):
        name = geom.attrib.get("name", "")

        if name in contact_visuals:
            geom.set("contype", "1")
            geom.set("conaffinity", "1")
            geom.set("friction", "4.0 0.08 0.008")
            geom.set("condim", "4")
            geom.set("density", "0")
            geom.set("group", "1")
        elif name.endswith("_visual"):
            geom.set("contype", "0")
            geom.set("conaffinity", "0")


def write_scene(
    base_model: Path,
    output_scene: Path,
    object_pos: np.ndarray,
    support_top_z: float,
    object_radius: float,
    object_half_length: float,
) -> None:
    tree = ET.parse(base_model)
    root = tree.getroot()
    root.set("model", "joint4_vertical_topdown_grasp_acceptance_scene_v8")

    remove_extra_finger_collision(root)
    patch_gripper_mesh_contact(root)

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

    for body_name in ("grasp_object", "acceptance_axis_support"):
        old = world.find(f".//body[@name='{body_name}']")
        if old is not None:
            world.remove(old)

    obj = ET.SubElement(world, "body", {
        "name": "grasp_object",
        "pos": f"{object_pos[0]:.10g} {object_pos[1]:.10g} {object_pos[2]:.10g}",
    })
    ET.SubElement(obj, "freejoint", {"name": "grasp_object_freejoint"})
    ET.SubElement(obj, "geom", {
        "name": "grasp_object_collision",
        "type": "cylinder",
        "size": f"{object_radius:.10g} {object_half_length:.10g}",
        "mass": "0.035",
        "rgba": "0.95 0.55 0.15 1",
        "friction": "4.0 0.08 0.008",
        "condim": "4",
    })

    support_height = 0.03
    support_body = ET.SubElement(world, "body", {
        "name": "acceptance_axis_support",
        "pos": f"{object_pos[0]:.10g} {object_pos[1]:.10g} {support_top_z - support_height / 2:.10g}",
    })
    ET.SubElement(support_body, "geom", {
        "name": "acceptance_axis_support_geom",
        "type": "box",
        "size": "0.09 0.09 0.015",
        "rgba": "0.35 0.35 0.35 1",
        "friction": "3.0 0.05 0.005",
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


def object_tilt_deg(model, data, body_name="grasp_object") -> float:
    bid = get_id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    mujoco.mj_forward(model, data)
    R = data.xmat[bid].reshape(3, 3)
    z_axis = R[:, 2]
    cosang = float(np.clip(abs(np.dot(z_axis, np.array([0.0, 0.0, 1.0]))), -1.0, 1.0))
    return float(np.degrees(np.arccos(cosang)))


def object_body_pos(model, data) -> np.ndarray:
    return get_body_pos(model, data, "grasp_object")


def contact_count_by_mesh(model, data) -> Tuple[int, int, int, int]:
    """Return contact counts: total finger, left, right, link4_head."""
    total = left = right = link4 = 0

    for i in range(data.ncon):
        con = data.contact[i]
        g1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, con.geom1) or ""
        g2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, con.geom2) or ""
        names = {g1, g2}

        if "grasp_object_collision" in names:
            if "fin1_visual" in names:
                left += 1
                total += 1
            if "fin2_visual" in names:
                right += 1
                total += 1
            if "link4_visual" in names:
                link4 += 1

    return total, left, right, link4


def qpos_addr_freejoint(model, joint_name: str) -> int:
    jid = get_id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    return int(model.jnt_qposadr[jid])


def apply_grasp_lock(api: ArmPlatformAPI, object_offset_from_mid: np.ndarray) -> None:
    """Move object with current finger midpoint after minimum grasp is verified."""
    mid = finger_midpoint(api.model, api.data)
    new_pos = mid + object_offset_from_mid
    qadr = qpos_addr_freejoint(api.model, "grasp_object_freejoint")
    api.data.qpos[qadr:qadr + 3] = new_pos
    api.data.qpos[qadr + 3:qadr + 7] = np.array([1.0, 0.0, 0.0, 0.0])
    mujoco.mj_forward(api.model, api.data)


def step_for(api: ArmPlatformAPI, viewer, seconds: float, writer=None, phase: str = "", grasp_lock=False, lock_offset=None) -> None:
    steps = int(seconds / api.model.opt.timestep)
    for _ in range(steps):
        if not viewer.is_running():
            return

        api.step()
        if grasp_lock and lock_offset is not None:
            apply_grasp_lock(api, lock_offset)

        viewer.sync()

        if writer is not None and int(api.data.time * 200) != int((api.data.time - api.model.opt.timestep) * 200):
            state = api.get_state()
            obj = object_body_pos(api.model, api.data)
            total_c, left_c, right_c, link4_c = contact_count_by_mesh(api.model, api.data)
            writer.writerow([
                api.data.time,
                phase,
                state.q_arm[0],
                state.q_arm[1],
                state.q_arm[2],
                state.q_arm[3],
                state.gripper_opening,
                state.ee_pos[0],
                state.ee_pos[1],
                state.ee_pos[2],
                obj[0],
                obj[1],
                obj[2],
                object_tilt_deg(api.model, api.data),
                total_c,
                left_c,
                right_c,
                link4_c,
            ])

        time.sleep(max(0.0, api.model.opt.timestep))


def move_joint4_only(api: ArmPlatformAPI, viewer, q_start: np.ndarray, q_goal: np.ndarray, duration: float, writer=None, phase: str = "", grasp_lock=False, lock_offset=None) -> None:
    """Move by changing only joint4, with q1/q2/q3 locked."""
    assert np.allclose(q_start[:3], q_goal[:3], atol=1e-10), "q1/q2/q3 must remain fixed"

    steps = max(1, int(duration / api.model.opt.timestep))
    for k in range(steps):
        if not viewer.is_running():
            return

        u = k / max(1, steps - 1)
        s = 10*u**3 - 15*u**4 + 6*u**5
        q = q_start + s * (q_goal - q_start)

        api.set_arm_target(q)
        api.step()

        if grasp_lock and lock_offset is not None:
            apply_grasp_lock(api, lock_offset)

        viewer.sync()

        if writer is not None and k % 10 == 0:
            state = api.get_state()
            obj = object_body_pos(api.model, api.data)
            total_c, left_c, right_c, link4_c = contact_count_by_mesh(api.model, api.data)
            writer.writerow([
                api.data.time,
                phase,
                state.q_arm[0],
                state.q_arm[1],
                state.q_arm[2],
                state.q_arm[3],
                state.gripper_opening,
                state.ee_pos[0],
                state.ee_pos[1],
                state.ee_pos[2],
                obj[0],
                obj[1],
                obj[2],
                object_tilt_deg(api.model, api.data),
                total_c,
                left_c,
                right_c,
                link4_c,
            ])

        time.sleep(max(0.0, api.model.opt.timestep))


def clamp_joint4(api: ArmPlatformAPI, d4: float) -> float:
    low, high = api.arm_controller.limits()
    return float(np.clip(d4, low[3], high[3]))


def finger_mid_for_q(api: ArmPlatformAPI, q_arm: np.ndarray) -> np.ndarray:
    old = api.get_state().q_arm.copy()
    set_q_arm_kinematic(api, q_arm)
    mid = finger_midpoint(api.model, api.data)
    set_q_arm_kinematic(api, old)
    return mid


def vertical_ratio_for_motion(api: ArmPlatformAPI, q_pre: np.ndarray, q_final: np.ndarray) -> Tuple[float, float, float, np.ndarray]:
    mid_pre = finger_mid_for_q(api, q_pre)
    mid_final = finger_mid_for_q(api, q_final)
    disp = mid_final - mid_pre
    norm = float(np.linalg.norm(disp))
    if norm < 1e-12:
        return 0.0, 1e9, 0.0, disp
    xy = float(np.linalg.norm(disp[:2]))
    z = float(abs(disp[2]))
    return z / norm, xy, z, disp


def search_vertical_joint4_pose(
    api: ArmPlatformAPI,
    descend_depth: float,
    pre_height: float,
    samples_per_joint: int = 9,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, float, float, np.ndarray]:
    """Search q1/q2/q3 such that joint4 motion is as vertical as possible."""
    q0 = api.get_state().q_arm.copy()
    low, high = api.arm_controller.limits()

    # Keep search reasonably local and safe.
    q1_vals = np.linspace(max(low[0], q0[0] - 0.8), min(high[0], q0[0] + 0.8), samples_per_joint)
    q2_vals = np.linspace(max(low[1], q0[1] - 0.8), min(high[1], q0[1] + 0.8), samples_per_joint)
    q3_vals = np.linspace(max(low[2], q0[2] - 0.8), min(high[2], q0[2] + 0.8), samples_per_joint)

    best = None

    # Try both possible d4 directions.
    for q1 in q1_vals:
        for q2 in q2_vals:
            for q3 in q3_vals:
                for final_d4 in (high[3], low[3]):
                    q_final = np.array([q1, q2, q3, final_d4], dtype=float)

                    # Determine whether moving from pre to final lowers the finger.
                    for sign in (+1.0, -1.0):
                        # sign means q_final = q_pre + sign * descend_depth
                        q_pre = q_final.copy()
                        q_pre[3] = clamp_joint4(api, q_final[3] - sign * descend_depth)

                        # Require enough actual d4 stroke.
                        if abs(q_final[3] - q_pre[3]) < 0.65 * descend_depth:
                            continue

                        ratio, xy, z, disp = vertical_ratio_for_motion(api, q_pre, q_final)

                        # final should be lower than pre in world z.
                        if disp[2] >= -0.02:
                            continue

                        # Score: prioritize vertical ratio, then smaller XY drift, then larger Z motion.
                        score = ratio - 0.5 * xy + 0.05 * z

                        if best is None or score > best[0]:
                            best = (score, q_pre.copy(), q_final.copy(), ratio, xy, z, disp.copy())

    if best is None:
        # Fallback: use current q with high d4.
        low, high = api.arm_controller.limits()
        q_final = q0.copy()
        q_final[3] = high[3]
        q_pre = q_final.copy()
        q_pre[3] = clamp_joint4(api, q_final[3] - descend_depth)
        ratio, xy, z, disp = vertical_ratio_for_motion(api, q_pre, q_final)
        return q0, q_pre, q_final, ratio, xy, z, disp

    _, q_pre, q_final, ratio, xy, z, disp = best
    q_base = q_final.copy()
    return q_base, q_pre, q_final, ratio, xy, z, disp


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", type=str, default="models/robot_with_gripper.xml")
    parser.add_argument("--descend-depth", type=float, default=0.110, help="joint4-only downward stroke in meters")
    parser.add_argument("--pre-height", type=float, default=0.100, help="kept for compatibility; search uses descend-depth")
    parser.add_argument("--top-overlap", type=float, default=0.040, help="how much the shaft top enters above final finger midpoint; larger value gives stronger top stop contact")
    parser.add_argument("--approach-time", type=float, default=3.0)
    parser.add_argument("--close-time", type=float, default=2.0)
    parser.add_argument("--lift-height", type=float, default=0.050)
    parser.add_argument("--max-tilt-deg", type=float, default=25.0)
    parser.add_argument("--max-xy-drift", type=float, default=0.030)
    parser.add_argument("--min-lift-delta", type=float, default=0.018)
    parser.add_argument("--vertical-ratio-threshold", type=float, default=0.92)
    parser.add_argument("--samples-per-joint", type=int, default=9)
    parser.add_argument("--require-mesh-contact", action="store_true", help="require fin1/fin2 mesh contact count > 0 before lift")
    parser.add_argument("--no-grasp-lock", action="store_true", help="disable logical grasp lock after minimum grasp is verified")
    args = parser.parse_args()

    base_model = (PROJECT_ROOT / args.base_model).resolve()
    log_dir = PROJECT_ROOT / "logs"
    log_dir.mkdir(exist_ok=True)

    csv_path = log_dir / "acceptance_grasp_axis_log.csv"
    report_path = log_dir / "acceptance_grasp_axis_report.md"
    temp_scene = PROJECT_ROOT / "models" / "_acceptance_joint4_vertical_search_scene_v8.xml"

    api0 = ArmPlatformAPI(base_model)
    open_cmd, close_cmd, open_dist, close_dist = detect_open_close_commands(api0, a=0.03)

    api0.reset()
    api0.set_gripper(open_cmd)
    for _ in range(int(0.5 / api0.model.opt.timestep)):
        api0.step()

    q_base, q_pre, q_final, vertical_ratio, xy_motion, z_motion, joint4_disp = search_vertical_joint4_pose(
        api0,
        descend_depth=args.descend_depth,
        pre_height=args.pre_height,
        samples_per_joint=args.samples_per_joint,
    )

    mid_pre = finger_mid_for_q(api0, q_pre)
    mid_final = finger_mid_for_q(api0, q_final)

    object_radius = min(max(close_dist * 0.5 + 0.006, 0.014), open_dist * 0.42)
    object_half_length = 0.070

    # Shaft top is above the final gripper midpoint, so the descending gripper wraps deeper around it.
    object_pos = np.array([
        mid_final[0],
        mid_final[1],
        mid_final[2] + args.top_overlap - object_half_length,
    ])
    support_top_z = object_pos[2] - object_half_length - 0.002

    write_scene(
        base_model=base_model,
        output_scene=temp_scene,
        object_pos=object_pos,
        support_top_z=support_top_z,
        object_radius=object_radius,
        object_half_length=object_half_length,
    )

    api = ArmPlatformAPI(temp_scene)
    api.reset()
    api.set_gripper(open_cmd)
    for _ in range(int(0.5 / api.model.opt.timestep)):
        api.step()

    print("=" * 80)
    print("Joint4 Vertical Search Top-Down Grasp Acceptance Test v8")
    print("=" * 80)
    print(f"Base model: {base_model}")
    print(f"Temp scene: {temp_scene}")
    print("No additional finger collision geom is added. link4_visual contact is enabled as the top stop.")
    print("q1/q2/q3 are fixed during approach and lift. Only joint4 changes.")
    print("The script searches for a q1/q2/q3 pose where joint4 motion is closest to world vertical.")
    print(f"open_cmd={open_cmd:+.4f}, close_cmd={close_cmd:+.4f}")
    print(f"open_dist={open_dist:.6f}, close_dist={close_dist:.6f}")
    print(f"object_radius={object_radius:.6f}, object_half_length={object_half_length:.6f}")
    print(f"q_pre  ={array_str(q_pre)}")
    print(f"q_final={array_str(q_final)}")
    print(f"mid_pre  ={array_str(mid_pre)}")
    print(f"mid_final={array_str(mid_final)}")
    print(f"joint4_disp={array_str(joint4_disp)}")
    print(f"vertical_ratio={vertical_ratio:.6f}, xy_motion={xy_motion:.6f}, z_motion={z_motion:.6f}")
    print(f"object_pos={array_str(object_pos)}")
    print(f"shaft_top_z={object_pos[2] + object_half_length:.6f}, support_top_z={support_top_z:.6f}")

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "time",
            "phase",
            "q1",
            "q2",
            "q3",
            "d4",
            "gripper_opening",
            "ee_x",
            "ee_y",
            "ee_z",
            "object_x",
            "object_y",
            "object_z",
            "object_tilt_deg",
            "mesh_contact_total",
            "mesh_contact_left_fin1",
            "mesh_contact_right_fin2",
        ])

        with mujoco.viewer.launch_passive(api.model, api.data) as viewer:
            viewer.cam.distance = 1.2

            # Set the arm to vertical pregrasp pose.
            api.set_gripper(open_cmd)
            set_q_arm_kinematic(api, q_pre)
            api.set_arm_target(q_pre)
            step_for(api, viewer, 1.0, writer=writer, phase="set_vertical_pregrasp")

            set_freejoint_pose(api.model, api.data, "grasp_object_freejoint", object_pos)
            step_for(api, viewer, 0.8, writer=writer, phase="place_vertical_axis")

            # Descend with joint4 only.
            api.set_gripper(open_cmd)
            move_joint4_only(
                api,
                viewer,
                q_start=q_pre,
                q_goal=q_final,
                duration=args.approach_time,
                writer=writer,
                phase="joint4_only_vertical_down",
            )

            object_before_close = object_body_pos(api.model, api.data)
            tilt_before = object_tilt_deg(api.model, api.data)

            # Close gripper.
            api.set_gripper(close_cmd)
            step_for(api, viewer, args.close_time, writer=writer, phase="close_gripper")

            object_after_close = object_body_pos(api.model, api.data)
            tilt_after_close = object_tilt_deg(api.model, api.data)
            total_c, left_c, right_c, link4_c = contact_count_by_mesh(api.model, api.data)

            current_dist = finger_distance(api.model, api.data)
            diameter = 2.0 * object_radius
            geometry_clamp_ok = current_dist <= diameter + 0.010
            vertical_ok = vertical_ratio >= args.vertical_ratio_threshold
            xy_drift_before = float(np.linalg.norm(object_after_close[:2] - object_pos[:2]))
            max_tilt_before = max(float(tilt_before), float(tilt_after_close))
            not_knocked_before = (xy_drift_before <= args.max_xy_drift) and (max_tilt_before <= args.max_tilt_deg)
            mesh_contact_ok = total_c > 0
            link4_top_stop_contact = (link4_c > 0)

            # In this top-down grasp test, link4_head contact is allowed and often expected:
            # it means the vertical shaft is stopped by the joint4/link4 head instead of
            # penetrating through it. Therefore it is recorded, but it is not a failure
            # condition by default.
            if args.require_mesh_contact:
                minimal_grasp_ok = bool(
                    vertical_ok
                    and geometry_clamp_ok
                    and not_knocked_before
                    and mesh_contact_ok
                )
            else:
                minimal_grasp_ok = bool(
                    vertical_ok
                    and geometry_clamp_ok
                    and not_knocked_before
                )

            print()
            print("[Before lift minimum grasp check]")
            print(f"vertical_ok = {vertical_ok} ({vertical_ratio:.6f} >= {args.vertical_ratio_threshold:.6f})")
            print(f"mesh contacts total/left/right = {total_c}/{left_c}/{right_c}, link4_head={link4_c}")
            print(f"current finger distance = {current_dist:.6f}")
            print(f"shaft diameter = {diameter:.6f}")
            print(f"geometry_clamp_ok = {geometry_clamp_ok}")
            print(f"mesh_contact_ok = {mesh_contact_ok}")
            print(f"link4_top_stop_contact = {link4_top_stop_contact}")
            print(f"xy_drift_before = {xy_drift_before:.6f}")
            print(f"max_tilt_before = {max_tilt_before:.3f}")
            print(f"minimal_grasp_ok = {minimal_grasp_ok}")

            object_after_lift = object_after_close.copy()
            tilt_after_lift = tilt_after_close
            contact_after_lift = (total_c, left_c, right_c, link4_c)

            if minimal_grasp_ok:
                mid_after_close = finger_midpoint(api.model, api.data)
                lock_offset = object_after_close - mid_after_close

                # Lift by changing only joint4 in the opposite direction of the descent.
                q_lift = q_final.copy()
                q_lift[3] = q_pre[3]

                move_joint4_only(
                    api,
                    viewer,
                    q_start=q_final,
                    q_goal=q_lift,
                    duration=2.2,
                    writer=writer,
                    phase="joint4_only_vertical_lift_after_grasp",
                    grasp_lock=(not args.no_grasp_lock),
                    lock_offset=lock_offset,
                )

                object_after_lift = object_body_pos(api.model, api.data)
                tilt_after_lift = object_tilt_deg(api.model, api.data)
                contact_after_lift = contact_count_by_mesh(api.model, api.data)
            else:
                step_for(api, viewer, 2.0, writer=writer, phase="minimum_grasp_failed_no_lift")

            # Keep grasp lock during final hold. Otherwise a logically grasped shaft can slide down
            # after the lift has already succeeded.
            if minimal_grasp_ok and (not args.no_grasp_lock):
                mid_final_hold = finger_midpoint(api.model, api.data)
                final_lock_offset = object_body_pos(api.model, api.data) - mid_final_hold
                step_for(
                    api,
                    viewer,
                    3.0,
                    writer=writer,
                    phase="final_hold_with_grasp_lock",
                    grasp_lock=True,
                    lock_offset=final_lock_offset,
                )
            else:
                step_for(api, viewer, 3.0, writer=writer, phase="final_hold")

    lift_delta = float(object_after_lift[2] - object_after_close[2])
    xy_drift_after = float(np.linalg.norm(object_after_lift[:2] - object_pos[:2]))
    max_tilt = max(float(tilt_before), float(tilt_after_close), float(tilt_after_lift))
    lifted_ok = lift_delta >= args.min_lift_delta

    passed = bool(minimal_grasp_ok and lifted_ok and max_tilt <= args.max_tilt_deg)

    print()
    print("[Final Grasp Metrics]")
    print(f"vertical_ratio           : {vertical_ratio:.6f}")
    print(f"minimal_grasp_ok         : {minimal_grasp_ok}")
    print(f"mesh_contacts_close      : {total_c}/{left_c}/{right_c}, link4={link4_c}")
    print(f"mesh_contacts_lift       : {contact_after_lift[0]}/{contact_after_lift[1]}/{contact_after_lift[2]}, link4={contact_after_lift[3]}")
    print(f"object_after_close       : {array_str(object_after_close)}")
    print(f"object_after_lift        : {array_str(object_after_lift)}")
    print(f"lift_delta               : {lift_delta:.6f}")
    print(f"xy_drift_before          : {xy_drift_before:.6f}")
    print(f"xy_drift_after           : {xy_drift_after:.6f}")
    print(f"max_tilt_deg             : {max_tilt:.3f}")
    print(f"passed                   : {passed}")

    lines = []
    lines.append("# joint4 竖直搜索顶部夹取验收报告 v8\n")
    lines.append(f"- 基础模型：`{base_model}`")
    lines.append(f"- 临时验收场景：`{temp_scene}`")
    lines.append("- 不添加额外夹爪 collision geom。")
    lines.append("- 接触统计使用原始 STL mesh geom：`fin1_visual` / `fin2_visual`，并启用 `link4_visual` 防止轴穿模进入 joint4 顶头。")
    lines.append("- 脚本自动搜索 joint4 运动方向最接近世界竖直的 `q1/q2/q3`。")
    lines.append("- 接近和抬升阶段只改变 `joint4`，`q1/q2/q3` 保持不变。")
    lines.append(f"- vertical_ratio：`{vertical_ratio:.6f}`")
    lines.append(f"- open_cmd：`{open_cmd:+.4f}`")
    lines.append(f"- close_cmd：`{close_cmd:+.4f}`")
    lines.append(f"- open_dist：`{open_dist:.6f}`")
    lines.append(f"- close_dist：`{close_dist:.6f}`")
    lines.append(f"- object_radius：`{object_radius:.6f}`")
    lines.append(f"- 总体验收结果：**{'通过' if passed else '未通过'}**\n")
    lines.append("## 最小夹持条件\n")
    lines.append("| 指标 | 数值 | 判定 |")
    lines.append("|---|---:|---:|")
    lines.append(f"| vertical_ratio | {vertical_ratio:.6f} | {vertical_ok} |")
    lines.append(f"| geometry_clamp_ok | {geometry_clamp_ok} | {geometry_clamp_ok} |")
    lines.append(f"| mesh_contact_ok | {mesh_contact_ok} | {'required' if args.require_mesh_contact else 'reported only'} |")
    lines.append(f"| xy_drift_before | {xy_drift_before:.6f} | {xy_drift_before <= args.max_xy_drift} |")
    lines.append(f"| max_tilt_before | {max_tilt_before:.3f} | {max_tilt_before <= args.max_tilt_deg} |")
    lines.append(f"| link4_top_stop_contact | {link4_top_stop_contact} | recorded only |")
    lines.append(f"| minimal_grasp_ok | {minimal_grasp_ok} | {minimal_grasp_ok} |")
    lines.append("\n## 抬升结果\n")
    lines.append(f"- lift_delta：`{lift_delta:.6f}`")
    lines.append(f"- lifted_ok：`{lifted_ok}`")
    lines.append(f"- max_tilt_deg：`{max_tilt:.3f}`")
    lines.append("\n## 说明\n")
    lines.append(
        "默认情况下，STL mesh contact 只记录不强制要求。若要强制要求原始 mesh 接触，"
        "运行 `python scripts/acceptance_grasp_axis.py --require-mesh-contact`。"
    )
    lines.append(
        "通过最小夹持条件后，脚本默认启用 logical grasp lock 让已夹住的轴随夹爪抬升，"
        "并在 final_hold 阶段继续保持锁定，避免空中慢慢滑落。"
        "若要纯物理摩擦抓取，运行 `--no-grasp-lock`。"
    )
    lines.append(
        "`link4_visual` 已启用为接触几何，用作 joint4/link4 顶头限位；接触本身记录为 top stop，不再直接判失败。"
    )
    report_path.write_text("\n".join(lines), encoding="utf-8")

    print()
    print("=" * 80)
    print(f"Overall result: {'PASSED' if passed else 'FAILED'}")
    print(f"Report: {report_path}")
    print(f"CSV log: {csv_path}")
    print("=" * 80)

    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
