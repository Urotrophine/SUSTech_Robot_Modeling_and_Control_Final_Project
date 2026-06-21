# -*- coding: utf-8 -*-
"""
Run an algorithm-level reproduction of the paper peg-in-hole strategy.

The paper workflow is represented as:
    Reaching  = pushing
    Searching = pushing + rubbing / Archimedes spiral
    Inserting = pushing + wiggling + screwing

This demo is task-level and intentionally keeps grasping stable with a logical
grasp lock after the shaft has been grasped. The search plane is world XY and
the insertion direction is world Z downward.

Run from project root:
    python scripts/run_peg_in_hole_demo.py
    python scripts/run_peg_in_hole_demo.py --headless --no-sleep
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

import mujoco
import mujoco.viewer

from api.arm_platform_api import ArmPlatformAPI
from control.admittance_controller import Joint4AdmittanceController
from planning.joint_trajectory import JointTrajectory
from planning.screw_motion import ScrewMotion
from planning.spiral_search import ArchimedesSpiralSearch
from planning.vision_error import sample_random_xy_offset
from task.peg_in_hole.paper_mapping import algorithm_reproduction_notes, classify_demo_phase
from task.peg_in_hole.spiral_trace import SpiralTraceSample, write_spiral_trace_svg
from task.peg_in_hole.state_machine import PegInHoleContext, PegInHoleState, PegInHoleStateMachine
from vision.mujoco_keypoint_vision import (
    KeypointVisionEstimate,
    estimate_mujoco_oracle,
    estimate_mujoco_unet,
    estimate_random_vision,
    render_camera_rgbd,
    write_keypoint_overlay_svg,
)


@dataclass
class DemoResult:
    passed: bool
    final_state: PegInHoleState
    final_alignment_error: float
    final_insertion_depth: float
    max_search_radius_used: float
    failure_reason: str
    report_path: Path
    csv_path: Path
    spiral_trace_path: Path
    vision_debug_path: Optional[Path]
    initial_offset_xy: np.ndarray
    offset_mode: str
    offset_seed: Optional[int]
    final_shaft_axis: np.ndarray
    final_tool_tilt_deg: float
    hole_xy_radius: float
    vision_estimate: KeypointVisionEstimate


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
    api.arm_target = np.asarray(q_arm, dtype=float).copy()
    mujoco.mj_forward(api.model, api.data)


def set_slide_pair_direct(model, data, joint5_value: float) -> None:
    j5 = get_id(model, mujoco.mjtObj.mjOBJ_JOINT, "joint5")
    j6 = get_id(model, mujoco.mjtObj.mjOBJ_JOINT, "joint6")
    data.qpos[model.jnt_qposadr[j5]] = joint5_value
    data.qpos[model.jnt_qposadr[j6]] = -joint5_value
    mujoco.mj_forward(model, data)


def finger_distance(model, data) -> float:
    left = get_site_pos(model, data, "left_finger_tip_site")
    right = get_site_pos(model, data, "right_finger_tip_site")
    return float(np.linalg.norm(left - right))


def detect_open_close_commands(api: ArmPlatformAPI, a: float = 0.03):
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
        return -a, +a
    return +a, -a


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


def rotation_z(angle: float) -> np.ndarray:
    c = float(np.cos(angle))
    s = float(np.sin(angle))
    return np.array(
        [
            [c, -s, 0.0],
            [s, c, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )


def quat_from_matrix(rot: np.ndarray) -> np.ndarray:
    m = np.asarray(rot, dtype=float)
    trace = float(np.trace(m))
    if trace > 0.0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (m[2, 1] - m[1, 2]) * s
        y = (m[0, 2] - m[2, 0]) * s
        z = (m[1, 0] - m[0, 1]) * s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = 2.0 * np.sqrt(max(0.0, 1.0 + m[0, 0] - m[1, 1] - m[2, 2]))
        w = (m[2, 1] - m[1, 2]) / s
        x = 0.25 * s
        y = (m[0, 1] + m[1, 0]) / s
        z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = 2.0 * np.sqrt(max(0.0, 1.0 + m[1, 1] - m[0, 0] - m[2, 2]))
        w = (m[0, 2] - m[2, 0]) / s
        x = (m[0, 1] + m[1, 0]) / s
        y = 0.25 * s
        z = (m[1, 2] + m[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(max(0.0, 1.0 + m[2, 2] - m[0, 0] - m[1, 1]))
        w = (m[1, 0] - m[0, 1]) / s
        x = (m[0, 2] + m[2, 0]) / s
        y = (m[1, 2] + m[2, 1]) / s
        z = 0.25 * s
    quat = np.array([w, x, y, z], dtype=float)
    return quat / max(1e-12, float(np.linalg.norm(quat)))


def site_rotation(api: ArmPlatformAPI, site_name: str) -> np.ndarray:
    sid = get_id(api.model, mujoco.mjtObj.mjOBJ_SITE, site_name)
    mujoco.mj_forward(api.model, api.data)
    return api.data.site_xmat[sid].reshape(3, 3).copy()


def grasp_site_pos(api: ArmPlatformAPI, peg_tip_site: str) -> np.ndarray:
    del peg_tip_site
    left = get_site_pos(api.model, api.data, "left_finger_tip_site")
    right = get_site_pos(api.model, api.data, "right_finger_tip_site")
    return 0.5 * (left + right)


def tool_shaft_axis(api: ArmPlatformAPI, peg_tip_site: str) -> np.ndarray:
    rot = site_rotation(api, peg_tip_site)
    axis = -rot[:, 1]
    return axis / max(1e-12, float(np.linalg.norm(axis)))


def hole_axis_vector(api: ArmPlatformAPI, hole_entry_site: str, hole_axis_site: str) -> np.ndarray:
    entry = get_site_pos(api.model, api.data, hole_entry_site)
    axis_point = get_site_pos(api.model, api.data, hole_axis_site)
    axis = axis_point - entry
    return axis / max(1e-12, float(np.linalg.norm(axis)))


def shaft_tip_pos(
    api: ArmPlatformAPI,
    peg_tip_site: str,
    grasp_to_tip: float,
    shaft_axis_world: np.ndarray,
) -> np.ndarray:
    return grasp_site_pos(api, peg_tip_site) + grasp_to_tip * shaft_axis_world


def apply_grasp_lock(
    api: ArmPlatformAPI,
    peg_tip_site: str,
    shaft_half_length: float,
    grasp_to_tip: float,
    shaft_axis_world: np.ndarray,
    screw_angle: float = 0.0,
) -> None:
    grasp_pos = grasp_site_pos(api, peg_tip_site)
    axis = shaft_axis_world / max(1e-12, float(np.linalg.norm(shaft_axis_world)))
    center = grasp_pos + (grasp_to_tip - shaft_half_length) * axis

    tool_rot = site_rotation(api, peg_tip_site)
    radial_x = tool_rot[:, 0] - axis * float(np.dot(tool_rot[:, 0], axis))
    if np.linalg.norm(radial_x) < 1e-9:
        radial_x = np.array([1.0, 0.0, 0.0], dtype=float) - axis * axis[0]
    radial_x = radial_x / max(1e-12, float(np.linalg.norm(radial_x)))
    radial_y = np.cross(axis, radial_x)
    radial_y = radial_y / max(1e-12, float(np.linalg.norm(radial_y)))
    base_rot = np.column_stack([radial_x, radial_y, axis])
    object_rot = base_rot @ rotation_z(screw_angle)
    set_freejoint_pose(
        api.model,
        api.data,
        "grasp_object_freejoint",
        center,
        quat=quat_from_matrix(object_rot),
    )


def pseudo_contact_force(peg_tip: np.ndarray, hole_entry: np.ndarray, stiffness: float, contact_tolerance: float) -> float:
    contact_plane_z = float(hole_entry[2] + contact_tolerance)
    penetration = max(0.0, contact_plane_z - float(peg_tip[2]))
    return float(stiffness * penetration)


def safe_log_stem(value: str) -> str:
    stem = str(value or "peg_in_hole").strip()
    cleaned = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in stem)
    return cleaned or "peg_in_hole"


def resolve_initial_offset(args, geom_cfg: dict, search_cfg: dict) -> tuple[np.ndarray, str, Optional[int], float]:
    configured_fixed = np.array(geom_cfg["initial_search_offset_xy"], dtype=float)
    mode = str(getattr(args, "offset_mode", "random")).lower()
    explicit_offset = getattr(args, "initial_offset_xy", None)
    if explicit_offset is not None:
        return np.array(explicit_offset, dtype=float), "fixed", None, float(getattr(args, "offset_radius", 0.0))

    offset_radius = float(getattr(args, "offset_radius", 0.0))
    if offset_radius <= 0.0:
        offset_radius = min(float(np.linalg.norm(configured_fixed)), float(search_cfg["radius_max"]))

    if mode == "fixed":
        return configured_fixed, "fixed", None, offset_radius

    seed = getattr(args, "seed", None)
    rng = np.random.default_rng(seed)
    return sample_random_xy_offset(offset_radius, rng=rng), "random", seed, offset_radius


def load_task_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)["peg_in_hole"]


def sync_viewer(viewer, no_sleep: bool, dt: float) -> bool:
    if viewer is None:
        if not no_sleep:
            time.sleep(dt)
        return True
    if not viewer.is_running():
        return False
    viewer.sync()
    if not no_sleep:
        time.sleep(dt)
    return True


def write_report(
    path: Path,
    result: DemoResult,
    target_force: float,
    alignment_tolerance: float,
    success_depth: float,
    initial_offset_xy: np.ndarray,
    search_radius: float,
    spiral_pitch: float,
    spiral_angular_speed: float,
    velocity_threshold: float,
    velocity_hold_time: float,
    wiggle_amplitude: float,
    screw_turns: float,
    offset_mode: str,
    offset_seed: Optional[int],
    offset_radius: float,
    spiral_trace_path: Path,
    final_shaft_axis: np.ndarray,
    final_tool_tilt_deg: float,
    hole_xy_radius: float,
) -> None:
    lines = []
    lines.append("# Peg-in-Hole Paper Algorithm Reproduction Report\n")
    lines.append("- Reference paper: `Compliance-Based Robotic Peg-in-Hole Assembly Strategy Without Force Feedback`")
    lines.append("- Reproduction level: `algorithm-level simulation`")
    lines.append(f"- Final state: `{result.final_state.name}`")
    lines.append(f"- Result: **{'PASSED' if result.passed else 'FAILED'}**")
    lines.append(f"- Failure reason: `{result.failure_reason or 'none'}`")
    lines.append(f"- Vision mode: `{result.vision_estimate.mode}`")
    lines.append(f"- Vision source: `{result.vision_estimate.source}`")
    lines.append(f"- Vision camera: `{result.vision_estimate.camera_name}`")
    lines.append(f"- Estimated hole world: `{array_str(result.vision_estimate.hole_world_est, precision=4)}` m")
    lines.append(f"- Estimated peg world: `{array_str(result.vision_estimate.peg_world_est, precision=4)}` m")
    lines.append(f"- Vision XY offset from true hole: `{array_str(initial_offset_xy, precision=4)}` m")
    lines.append(f"- Hole pixel estimate: `{array_str(result.vision_estimate.hole_pixel, precision=3)}`")
    lines.append(f"- Peg pixel estimate: `{array_str(result.vision_estimate.peg_pixel, precision=3)}`")
    lines.append(f"- Hole pixel error: `{result.vision_estimate.hole_pixel_error:.6f}` px")
    lines.append(f"- Peg pixel error: `{result.vision_estimate.peg_pixel_error:.6f}` px")
    lines.append(f"- Legacy offset mode: `{offset_mode}`")
    lines.append(f"- Random offset radius limit: `{offset_radius:.6f}` m")
    lines.append(f"- Random seed: `{offset_seed if offset_seed is not None else 'none'}`")
    lines.append(f"- Spiral search radius: `{search_radius:.6f}` m")
    lines.append(f"- Spiral pitch: `{spiral_pitch:.6f}` m/turn")
    lines.append(f"- Spiral angular speed: `{spiral_angular_speed:.6f}` rad/s")
    lines.append(f"- Target contact force: `{target_force:.3f}` N")
    lines.append(f"- Velocity/contact threshold: `{velocity_threshold:.6f}` m/s for `{velocity_hold_time:.3f}` s")
    lines.append(f"- Wiggle amplitude: `{wiggle_amplitude:.6f}` m")
    lines.append(f"- Screw turns: `{screw_turns:.3f}`")
    lines.append(f"- Final XY alignment error: `{result.final_alignment_error:.6f}` m")
    lines.append(f"- Alignment tolerance: `{alignment_tolerance:.6f}` m")
    lines.append(f"- Final insertion depth: `{result.final_insertion_depth:.6f}` m")
    lines.append(f"- Required insertion depth: `{success_depth:.6f}` m")
    lines.append(f"- Hole XY radius from base: `{hole_xy_radius:.6f}` m")
    lines.append(f"- Final shaft/tool axis: `{array_str(final_shaft_axis, precision=4)}`")
    lines.append(f"- Final tool tilt from world vertical: `{final_tool_tilt_deg:.3f}` deg")
    lines.append(f"- Max search radius used: `{result.max_search_radius_used:.6f}` m")
    lines.append(f"- CSV log: `{result.csv_path.name}`")
    lines.append(f"- Spiral trace image: `{spiral_trace_path.name}`")
    if result.vision_debug_path is not None:
        lines.append(f"- Vision keypoint overlay: `{result.vision_debug_path.name}`")
    if result.vision_debug_path is not None:
        lines.append("\n## Vision Keypoint Overlay\n")
        lines.append(f"![MuJoCo camera keypoint overlay]({result.vision_debug_path.name})")
    lines.append("\n## Spiral Trace\n")
    lines.append(f"![Archimedes spiral rubbing trace]({spiral_trace_path.name})")
    lines.append("\n## Paper Procedure Mapping\n")
    lines.append("| Paper step | Unit motion represented in this demo |")
    lines.append("| --- | --- |")
    lines.append("| Setup | initial gripper grasp, then carry peg to estimated hole pose |")
    lines.append("| Reaching | pushing |")
    lines.append("| Searching | pushing + rubbing / Archimedes spiral |")
    lines.append("| Inserting | pushing + wiggling + screwing |")
    lines.append("\n## Notes\n")
    lines.extend(algorithm_reproduction_notes())
    path.write_text("\n".join(lines), encoding="utf-8")


def run_demo(args, viewer=None, api: Optional[ArmPlatformAPI] = None) -> DemoResult:
    cfg = load_task_config(PROJECT_ROOT / "configs" / "peg_in_hole_task.yaml")
    geom_cfg = cfg["geometry"]
    search_cfg = cfg["search"]
    insertion_cfg = cfg["insertion"]
    force_cfg = cfg["force_control"]
    contact_cfg = cfg["contact_detection"]
    site_cfg = cfg["sites"]

    model_path = (PROJECT_ROOT / args.model).resolve()
    log_dir = PROJECT_ROOT / "logs"
    log_dir.mkdir(exist_ok=True)
    log_stem = safe_log_stem(getattr(args, "log_stem", "peg_in_hole"))
    csv_path = log_dir / f"{log_stem}_log.csv"
    report_path = log_dir / f"{log_stem}_report.md"
    spiral_trace_path = log_dir / f"{log_stem}_spiral_trace.svg"
    vision_debug_path = log_dir / f"{log_stem}_vision_keypoints.svg"

    if api is None:
        api = ArmPlatformAPI(model_path)
    open_cmd, close_cmd = detect_open_close_commands(api)
    close_cmd = 0.04 if close_cmd >= 0.0 else -0.04

    peg_tip_site = site_cfg["peg_tip"]
    hole_entry_site = site_cfg["hole_entry"]
    hole_axis_site = site_cfg["hole_axis"]

    hole_entry = get_site_pos(api.model, api.data, hole_entry_site)
    initial_offset, offset_mode, offset_seed, offset_radius = resolve_initial_offset(args, geom_cfg, search_cfg)
    shaft_axis_mode = str(geom_cfg.get("shaft_axis_mode", "hole")).lower()
    shaft_half_length = float(geom_cfg["shaft_half_length"])
    grasp_to_tip = float(geom_cfg["grasp_to_tip"])
    pre_insert_clearance = float(geom_cfg["pre_insert_clearance"])
    alignment_tolerance = float(args.alignment_tolerance)
    contact_tolerance = float(args.contact_tolerance)
    success_depth = float(args.success_depth)
    pseudo_stiffness = float(force_cfg["pseudo_contact_stiffness"])
    velocity_threshold = float(contact_cfg["velocity_threshold"])
    velocity_hold_required = float(contact_cfg["hold_time"])
    use_kinematic_velocity = bool(contact_cfg.get("use_kinematic_velocity", True))
    dt = float(args.dt)

    spiral = ArchimedesSpiralSearch(
        radius_max=float(args.search_radius),
        pitch=float(args.spiral_pitch),
        angular_speed=float(args.spiral_angular_speed),
    )
    screw = ScrewMotion(
        push_depth=float(args.insert_depth),
        screw_amplitude=float(args.screw_amplitude),
        duration=float(args.insert_duration),
        wiggle_amplitude=float(args.wiggle_amplitude),
        screw_turns=float(args.screw_turns),
    )

    machine = PegInHoleStateMachine()
    ctx = PegInHoleContext(
        alignment_tolerance=alignment_tolerance,
        insert_depth_target=success_depth,
        insert_timeout=float(insertion_cfg["timeout"]),
        search_timeout=float(args.max_time),
        max_search_radius=float(args.search_radius),
    )

    sim_time = 0.0
    state_enter_time = 0.0
    target_seed_q = np.array(geom_cfg.get("vertical_tool_seed_q", api.get_state().q_arm), dtype=float)
    initial_home_q = np.array(geom_cfg.get("initial_home_q", target_seed_q), dtype=float)
    initial_grasp_q = np.array(geom_cfg.get("initial_grasp_q", target_seed_q), dtype=float)
    set_q_arm_kinematic(api, target_seed_q)
    target_tool_quat = quat_from_matrix(site_rotation(api, peg_tip_site))
    set_q_arm_kinematic(api, initial_home_q)
    current_q = api.get_state().q_arm.copy()
    max_search_radius_used = 0.0
    last_spiral_offset = np.zeros(2, dtype=float)
    last_wiggle_offset = np.zeros(2, dtype=float)
    last_screw_angle = 0.0
    spiral_trace_samples: list[SpiralTraceSample] = []
    last_observed_tip: Optional[np.ndarray] = None
    last_observed_time: Optional[float] = None
    peg_speed = 0.0
    velocity_hold_elapsed = 0.0
    velocity_condition_met = False

    def current_shaft_axis() -> np.ndarray:
        if shaft_axis_mode == "tool":
            return tool_shaft_axis(api, peg_tip_site)
        return hole_axis_vector(api, hole_entry_site, hole_axis_site)

    def current_shaft_tip() -> np.ndarray:
        return shaft_tip_pos(api, peg_tip_site, grasp_to_tip, current_shaft_axis())

    def apply_current_grasp_lock(screw_angle: float = 0.0) -> None:
        apply_grasp_lock(
            api,
            peg_tip_site,
            shaft_half_length,
            grasp_to_tip,
            current_shaft_axis(),
            screw_angle=screw_angle,
        )

    vision_mode = str(args.vision_mode).lower()
    vision_rng = np.random.default_rng(args.seed)
    true_initial_peg = get_body_pos(api.model, api.data, "grasp_object") - np.array([0.0, 0.0, shaft_half_length])
    vision_estimate = estimate_random_vision(
        true_hole_world=hole_entry,
        true_peg_world=true_initial_peg,
        initial_offset_xy=initial_offset,
        mode="vision-pending",
        seed=offset_seed,
    )
    control_hole_entry = vision_estimate.hole_world_est.copy()
    vision_debug_output_path = vision_debug_path
    vision_debug_path = None

    def capture_vision_after_grasp() -> None:
        nonlocal vision_estimate, initial_offset, offset_mode, offset_radius, control_hole_entry, vision_debug_path
        true_grasped_peg = current_shaft_tip()
        if vision_mode == "random":
            estimate = estimate_random_vision(
                true_hole_world=hole_entry,
                true_peg_world=true_grasped_peg,
                initial_offset_xy=initial_offset,
                mode=offset_mode,
                seed=offset_seed,
            )
        elif vision_mode == "mujoco-oracle":
            estimate = estimate_mujoco_oracle(
                model=api.model,
                data=api.data,
                camera_name=str(args.vision_camera),
                true_hole_world=hole_entry,
                true_peg_world=true_grasped_peg,
                width=int(args.vision_width),
                height=int(args.vision_height),
                pixel_noise_std=float(args.vision_pixel_noise),
                rng=vision_rng,
            )
        elif vision_mode == "mujoco-unet":
            if args.vision_model is None:
                raise FileNotFoundError("mujoco-unet vision requires --vision-model pointing to a trained .pth file")
            vision_model_path = Path(args.vision_model)
            if not vision_model_path.is_absolute():
                vision_model_path = (PROJECT_ROOT / vision_model_path).resolve()
            estimate = estimate_mujoco_unet(
                model=api.model,
                data=api.data,
                camera_name=str(args.vision_camera),
                true_hole_world=hole_entry,
                true_peg_world=true_grasped_peg,
                width=int(args.vision_width),
                height=int(args.vision_height),
                model_path=vision_model_path,
                device=str(args.vision_device),
                crop_size=int(args.vision_crop_size),
                crop_hole_scale=float(args.vision_crop_hole_scale),
                hole_size_world=float(args.vision_hole_size),
                insertion_axis_world=hole_axis_vector(api, hole_entry_site, hole_axis_site),
            )
        else:
            raise ValueError(f"Unsupported vision mode: {args.vision_mode}")

        vision_estimate = estimate
        initial_offset = vision_estimate.initial_offset_xy.copy()
        offset_mode = vision_mode if vision_mode != "random" else offset_mode
        offset_radius = float(np.linalg.norm(initial_offset))
        control_hole_entry = vision_estimate.hole_world_est.copy()
        if vision_estimate.camera_name != "none":
            rgb, _ = render_camera_rgbd(
                api.model,
                api.data,
                vision_estimate.camera_name,
                int(args.vision_width),
                int(args.vision_height),
            )
            write_keypoint_overlay_svg(vision_debug_output_path, rgb, vision_estimate)
            vision_debug_path = vision_debug_output_path

    def machine_step() -> PegInHoleState:
        nonlocal state_enter_time
        old = machine.state
        new = machine.step(ctx)
        if new != old:
            state_enter_time = sim_time
            ctx.state_elapsed = 0.0
        return new

    def update_context() -> None:
        nonlocal last_observed_tip, last_observed_time, peg_speed, velocity_hold_elapsed, velocity_condition_met
        peg = current_shaft_tip()
        ctx.time = sim_time
        ctx.state_elapsed = sim_time - state_enter_time
        ctx.alignment_error_xy = float(np.linalg.norm(peg[:2] - hole_entry[:2]))
        ctx.insertion_depth = max(0.0, float(hole_entry[2] - peg[2]))
        near_contact_xy = ctx.alignment_error_xy <= max(
            0.06,
            float(args.search_radius) + offset_radius + alignment_tolerance,
        )
        raw_contact_force = pseudo_contact_force(peg, hole_entry, pseudo_stiffness, contact_tolerance)
        ctx.contact_force = raw_contact_force if near_contact_xy else 0.0
        if last_observed_tip is not None and last_observed_time is not None:
            elapsed = sim_time - last_observed_time
            if elapsed > 1e-9:
                peg_speed = float(np.linalg.norm(peg - last_observed_tip) / elapsed)
                if peg_speed <= velocity_threshold:
                    velocity_hold_elapsed += elapsed
                else:
                    velocity_hold_elapsed = 0.0
                velocity_condition_met = velocity_hold_elapsed >= velocity_hold_required
        last_observed_tip = peg.copy()
        last_observed_time = sim_time

        geometric_contact = near_contact_xy and (
            ctx.contact_force > 0.05 or peg[2] <= hole_entry[2] + contact_tolerance
        )
        velocity_contact = (
            use_kinematic_velocity
            and velocity_condition_met
            and near_contact_xy
            and peg[2] <= hole_entry[2] + contact_tolerance
        )
        ctx.contact_detected = ctx.contact_detected or geometric_contact or velocity_contact

    def write_log(writer, phase: str) -> None:
        peg = current_shaft_tip()
        state = api.get_state()
        paper_phase = classify_demo_phase(phase, machine.state.name)
        writer.writerow([
            sim_time,
            machine.state.name,
            phase,
            paper_phase.procedure_step,
            paper_phase.unit_motion,
            paper_phase.condition,
            state.q_arm[0],
            state.q_arm[1],
            state.q_arm[2],
            state.q_arm[3],
            peg[0],
            peg[1],
            peg[2],
            hole_entry[0],
            hole_entry[1],
            hole_entry[2],
            control_hole_entry[0],
            control_hole_entry[1],
            control_hole_entry[2],
            vision_estimate.peg_world_est[0],
            vision_estimate.peg_world_est[1],
            vision_estimate.peg_world_est[2],
            initial_offset[0],
            initial_offset[1],
            vision_estimate.hole_pixel[0],
            vision_estimate.hole_pixel[1],
            vision_estimate.peg_pixel[0],
            vision_estimate.peg_pixel[1],
            vision_estimate.hole_pixel_error,
            vision_estimate.peg_pixel_error,
            ctx.alignment_error_xy,
            ctx.insertion_depth,
            ctx.contact_force,
            peg_speed,
            int(velocity_condition_met),
            last_spiral_offset[0],
            last_spiral_offset[1],
            ctx.search_radius,
            last_wiggle_offset[0],
            last_wiggle_offset[1],
            last_screw_angle,
            machine.failure_reason,
        ])

    def set_tip_target_by_ik(target_tip_pos: np.ndarray, q_init: np.ndarray) -> np.ndarray:
        q_seed = np.asarray(q_init, dtype=float).copy()
        ik = None
        tip_error = float("inf")
        for _ in range(3):
            set_q_arm_kinematic(api, q_seed)
            grasp_target = target_tip_pos - grasp_to_tip * current_shaft_axis()
            grasp_to_ik_site = get_site_pos(api.model, api.data, peg_tip_site) - grasp_site_pos(api, peg_tip_site)
            ik_site_target = grasp_target + grasp_to_ik_site
            ik = api.ik_position(ik_site_target, q_init=q_seed, site_name=peg_tip_site)
            candidates = [ik.q.copy()]
            if shaft_axis_mode == "tool":
                pose_ik = api.kin.solve_ik_pose(
                    ik_site_target,
                    target_tool_quat,
                    q_init=ik.q,
                    site_name=peg_tip_site,
                    pos_weight=1.0,
                    rot_weight=0.12,
                    max_iter=60,
                    tol=max(5e-4, args.ik_tolerance),
                    step_scale=0.35,
                )
                candidates.append(pose_ik.q.copy())

            best_candidate = None
            best_score = float("inf")
            for candidate in candidates:
                set_q_arm_kinematic(api, candidate)
                candidate_tip_error = float(np.linalg.norm(target_tip_pos - current_shaft_tip()))
                candidate_axis = current_shaft_axis()
                candidate_tilt = 1.0 - abs(float(candidate_axis[2]))
                score = candidate_tip_error + 0.05 * candidate_tilt
                if score < best_score:
                    best_score = score
                    best_candidate = candidate.copy()
                    tip_error = candidate_tip_error

            q_seed = best_candidate
            set_q_arm_kinematic(api, q_seed)
            if tip_error <= args.ik_tolerance:
                break
        if ik is None or (not ik.success and tip_error > args.ik_tolerance):
            ctx.ik_failed = True
            machine_step()
            raise RuntimeError(f"IK failed for shaft tip target {array_str(target_tip_pos)}; tip_error={tip_error:.6g}")
        return q_seed.copy()

    def animate_to(target_tip_pos: np.ndarray, duration: float, phase: str, writer, lock_object: bool) -> np.ndarray:
        nonlocal sim_time, current_q
        q_goal = set_tip_target_by_ik(target_tip_pos, current_q)
        traj = JointTrajectory(current_q, q_goal, max(dt, duration), method="quintic")
        steps = max(1, int(duration / dt))
        for k in range(steps + 1):
            u_time = min(duration, k * dt)
            q = traj.sample(u_time)
            set_q_arm_kinematic(api, q)
            if lock_object:
                apply_current_grasp_lock(screw_angle=last_screw_angle)
            update_context()
            write_log(writer, phase)
            if not sync_viewer(viewer, args.no_sleep, dt):
                break
            sim_time += dt
        current_q = q_goal
        set_q_arm_kinematic(api, current_q)
        if lock_object:
            apply_current_grasp_lock(screw_angle=last_screw_angle)
        update_context()
        return current_q

    def animate_joint_to(q_goal: np.ndarray, duration: float, phase: str, writer, lock_object: bool) -> np.ndarray:
        nonlocal sim_time, current_q
        q_goal = np.asarray(q_goal, dtype=float).copy()
        traj = JointTrajectory(current_q, q_goal, max(dt, duration), method="quintic")
        steps = max(1, int(duration / dt))
        for k in range(steps + 1):
            u_time = min(duration, k * dt)
            q = traj.sample(u_time)
            set_q_arm_kinematic(api, q)
            if lock_object:
                apply_current_grasp_lock(screw_angle=last_screw_angle)
            update_context()
            write_log(writer, phase)
            if not sync_viewer(viewer, args.no_sleep, dt):
                break
            sim_time += dt
        current_q = q_goal
        set_q_arm_kinematic(api, current_q)
        if lock_object:
            apply_current_grasp_lock(screw_angle=last_screw_angle)
        update_context()
        return current_q

    def animate_gripper_to(joint5_goal: float, duration: float, phase: str, writer, lock_object: bool) -> None:
        nonlocal sim_time
        j5 = get_id(api.model, mujoco.mjtObj.mjOBJ_JOINT, "joint5")
        qadr = int(api.model.jnt_qposadr[j5])
        start = float(api.data.qpos[qadr])
        steps = max(1, int(duration / dt))
        for k in range(steps + 1):
            u = min(1.0, k / steps)
            smooth = 10.0 * u ** 3 - 15.0 * u ** 4 + 6.0 * u ** 5
            value = (1.0 - smooth) * start + smooth * float(joint5_goal)
            set_slide_pair_direct(api.model, api.data, value)
            set_q_arm_kinematic(api, current_q)
            if lock_object:
                apply_current_grasp_lock(screw_angle=last_screw_angle)
            update_context()
            write_log(writer, phase)
            if not sync_viewer(viewer, args.no_sleep, dt):
                break
            sim_time += dt
        set_slide_pair_direct(api.model, api.data, float(joint5_goal))

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "time",
            "state",
            "phase",
            "paper_step",
            "unit_motion",
            "paper_condition",
            "q1",
            "q2",
            "q3",
            "q4",
            "peg_x",
            "peg_y",
            "peg_z",
            "hole_x",
            "hole_y",
            "hole_z",
            "hole_est_x",
            "hole_est_y",
            "hole_est_z",
            "peg_est_x",
            "peg_est_y",
            "peg_est_z",
            "vision_offset_x",
            "vision_offset_y",
            "hole_pixel_x",
            "hole_pixel_y",
            "peg_pixel_x",
            "peg_pixel_y",
            "hole_pixel_error",
            "peg_pixel_error",
            "alignment_error_xy",
            "insertion_depth",
            "contact_force",
            "peg_speed",
            "velocity_condition_met",
            "spiral_x",
            "spiral_y",
            "spiral_radius",
            "wiggle_x",
            "wiggle_y",
            "screw_angle",
            "failure_reason",
        ])

        machine_step()

        set_slide_pair_direct(api.model, api.data, open_cmd)
        set_q_arm_kinematic(api, current_q)
        for _ in range(max(1, int(float(geom_cfg.get("grasp_ready_hold", 0.3)) / dt))):
            update_context()
            write_log(writer, "initial_ready_open_gripper")
            if not sync_viewer(viewer, args.no_sleep, dt):
                break
            sim_time += dt

        current_q = animate_joint_to(
            initial_grasp_q,
            duration=float(geom_cfg.get("grasp_approach_duration", 2.0)),
            phase="approach_initial_grasp",
            writer=writer,
            lock_object=False,
        )
        for _ in range(max(1, int(float(geom_cfg.get("grasp_settle_time", 0.2)) / dt))):
            update_context()
            write_log(writer, "initial_grasp_pose")
            if not sync_viewer(viewer, args.no_sleep, dt):
                break
            sim_time += dt
        ctx.approach_done = True
        machine_step()

        animate_gripper_to(
            close_cmd,
            duration=float(geom_cfg.get("gripper_close_duration", 0.6)),
            phase="close_initial_gripper",
            writer=writer,
            lock_object=False,
        )
        apply_current_grasp_lock(screw_angle=last_screw_angle)
        for _ in range(max(1, int(float(geom_cfg.get("grasp_settle_time", 0.2)) / dt))):
            update_context()
            write_log(writer, "close_initial_grasp_lock")
            if not sync_viewer(viewer, args.no_sleep, dt):
                break
            sim_time += dt
        ctx.grasp_done = True
        machine_step()

        capture_vision_after_grasp()
        pre_insert_target = control_hole_entry + np.array([0.0, 0.0, pre_insert_clearance], dtype=float)
        current_q = animate_to(pre_insert_target, duration=2.4, phase="carry_to_pre_insert", writer=writer, lock_object=True)
        ctx.pre_insert_done = True
        machine_step()

        contact_target = control_hole_entry.copy()
        current_q = animate_to(contact_target, duration=2.0, phase="reaching_pushing", writer=writer, lock_object=True)

        down_test = current_q.copy()
        up_test = current_q.copy()
        low, high = api.arm_controller.limits()
        down_test[3] = min(high[3], current_q[3] + 0.01)
        up_test[3] = max(low[3], current_q[3] - 0.01)
        set_q_arm_kinematic(api, down_test)
        z_plus = current_shaft_tip()[2]
        set_q_arm_kinematic(api, up_test)
        z_minus = current_shaft_tip()[2]
        down_sign = 1.0 if z_plus < z_minus else -1.0
        set_q_arm_kinematic(api, current_q)
        apply_current_grasp_lock(screw_angle=last_screw_angle)

        admittance = Joint4AdmittanceController(
            initial_command=float(current_q[3]),
            down_sign=down_sign,
            command_min=float(low[3]),
            command_max=float(high[3]),
            target_force=float(args.target_force),
            k_force=float(force_cfg["k_force"]),
            max_speed=float(force_cfg["max_speed"]),
            filter_alpha=float(force_cfg["filter_alpha"]),
            deadband=float(force_cfg["deadband"]),
        )
        admittance.reset(command=float(current_q[3]), measured_force=ctx.contact_force)

        for _ in range(max(1, int(1.2 / dt))):
            reading = ctx.contact_force
            ctrl = admittance.update(reading, dt)
            q_cmd = current_q.copy()
            q_cmd[3] = ctrl.command
            set_q_arm_kinematic(api, q_cmd)
            current_q = q_cmd
            apply_current_grasp_lock(screw_angle=last_screw_angle)
            update_context()
            write_log(writer, "reaching_pushing_velocity_contact")
            if not sync_viewer(viewer, args.no_sleep, dt):
                break
            sim_time += dt
            if ctx.contact_force >= 0.25 * args.target_force:
                ctx.contact_detected = True
                break

        if not ctx.contact_detected:
            ctx.contact_detected = ctx.contact_force > 0.05
        machine_step()

        search_start = sim_time
        while machine.state == PegInHoleState.SEARCHING_SPIRAL and sim_time <= args.max_time:
            t_search = sim_time - search_start
            last_spiral_offset[:] = spiral.sample_xy(t_search)
            ctx.search_elapsed = t_search
            ctx.search_radius = spiral.radius(t_search)
            max_search_radius_used = max(max_search_radius_used, ctx.search_radius)

            target_xy = control_hole_entry[:2] + last_spiral_offset
            target_tip_z = control_hole_entry[2] - max(ctx.insertion_depth, args.target_force / pseudo_stiffness)
            target = np.array([target_xy[0], target_xy[1], target_tip_z], dtype=float)
            current_q = set_tip_target_by_ik(target, current_q)
            set_q_arm_kinematic(api, current_q)
            apply_current_grasp_lock(screw_angle=last_screw_angle)
            update_context()
            actual_tip = current_shaft_tip()
            spiral_trace_samples.append(SpiralTraceSample(
                time=float(t_search),
                command_xy=target_xy - hole_entry[:2],
                actual_xy=actual_tip[:2] - hole_entry[:2],
                spiral_offset_xy=last_spiral_offset.copy(),
                radius=float(ctx.search_radius),
            ))
            write_log(writer, "searching_rubbing_spiral")
            machine_step()
            if not sync_viewer(viewer, args.no_sleep, dt):
                break
            sim_time += dt

        insert_start = sim_time
        insert_center_xy = current_shaft_tip()[:2].copy()
        insert_start_offset = np.zeros(2, dtype=float)
        insert_start_depth = ctx.insertion_depth
        handoff_duration = max(dt, float(args.handoff_duration))
        while machine.state == PegInHoleState.INSERTING_WIGGLE_SCREW and sim_time <= args.max_time:
            t_insert = sim_time - insert_start
            cmd = screw.sample_command(t_insert)
            tapered_wiggle_offset = cmd.wiggle_offset * (1.0 - cmd.progress) ** 2
            last_wiggle_offset[:] = tapered_wiggle_offset
            last_screw_angle = cmd.screw_angle

            blend = min(1.0, t_insert / handoff_duration)
            smooth_offset = (1.0 - blend) * insert_start_offset + blend * tapered_wiggle_offset
            target_xy = insert_center_xy + smooth_offset
            target_tip_z = control_hole_entry[2] - (insert_start_depth + cmd.insertion_depth)
            target = np.array([target_xy[0], target_xy[1], target_tip_z], dtype=float)
            current_q = set_tip_target_by_ik(target, current_q)
            set_q_arm_kinematic(api, current_q)
            apply_current_grasp_lock(screw_angle=last_screw_angle)
            update_context()
            write_log(writer, "inserting_wiggle_screw")
            machine_step()
            if not sync_viewer(viewer, args.no_sleep, dt):
                break
            sim_time += dt

        update_context()
        if machine.state not in (PegInHoleState.COMPLETE, PegInHoleState.FAILED):
            machine.failure_reason = "max_time_reached"
            ctx.failure_reason = machine.failure_reason
            machine.state = PegInHoleState.FAILED

        write_log(writer, "final")

    passed = (
        machine.state == PegInHoleState.COMPLETE
        and ctx.alignment_error_xy <= alignment_tolerance
        and ctx.insertion_depth >= success_depth
    )

    final_shaft_axis = current_shaft_axis()
    final_tool_tilt_deg = float(np.degrees(np.arccos(np.clip(abs(final_shaft_axis[2]), -1.0, 1.0))))
    hole_xy_radius = float(np.linalg.norm(hole_entry[:2]))

    result = DemoResult(
        passed=bool(passed),
        final_state=machine.state,
        final_alignment_error=float(ctx.alignment_error_xy),
        final_insertion_depth=float(ctx.insertion_depth),
        max_search_radius_used=float(max_search_radius_used),
        failure_reason=ctx.failure_reason or machine.failure_reason,
        report_path=report_path,
        csv_path=csv_path,
        spiral_trace_path=spiral_trace_path,
        vision_debug_path=vision_debug_path,
        initial_offset_xy=initial_offset.copy(),
        offset_mode=offset_mode,
        offset_seed=offset_seed,
        final_shaft_axis=final_shaft_axis.copy(),
        final_tool_tilt_deg=final_tool_tilt_deg,
        hole_xy_radius=hole_xy_radius,
        vision_estimate=vision_estimate,
    )
    write_spiral_trace_svg(
        spiral_trace_path,
        samples=spiral_trace_samples,
        initial_offset_xy=initial_offset,
        search_radius=float(args.search_radius),
        alignment_tolerance=alignment_tolerance,
    )
    write_report(
        report_path,
        result,
        target_force=float(args.target_force),
        alignment_tolerance=alignment_tolerance,
        success_depth=success_depth,
        initial_offset_xy=initial_offset,
        search_radius=float(args.search_radius),
        spiral_pitch=float(args.spiral_pitch),
        spiral_angular_speed=float(args.spiral_angular_speed),
        velocity_threshold=velocity_threshold,
        velocity_hold_time=velocity_hold_required,
        wiggle_amplitude=float(args.wiggle_amplitude),
        screw_turns=float(args.screw_turns),
        offset_mode=offset_mode,
        offset_seed=offset_seed,
        offset_radius=offset_radius,
        spiral_trace_path=spiral_trace_path,
        final_shaft_axis=final_shaft_axis,
        final_tool_tilt_deg=final_tool_tilt_deg,
        hole_xy_radius=hole_xy_radius,
    )
    return result


def parse_args() -> argparse.Namespace:
    cfg = load_task_config(PROJECT_ROOT / "configs" / "peg_in_hole_task.yaml")
    geom_cfg = cfg["geometry"]
    search_cfg = cfg["search"]
    insertion_cfg = cfg["insertion"]
    force_cfg = cfg["force_control"]
    contact_cfg = cfg["contact_detection"]
    vision_cfg = cfg.get("vision", {})

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=cfg["model_xml"])
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--no-sleep", action="store_true")
    parser.add_argument("--dt", type=float, default=0.02)
    parser.add_argument("--max-time", type=float, default=float(search_cfg["timeout"]))
    parser.add_argument("--search-radius", type=float, default=float(search_cfg["radius_max"]))
    parser.add_argument("--spiral-pitch", type=float, default=float(search_cfg["pitch"]))
    parser.add_argument("--spiral-angular-speed", type=float, default=float(search_cfg["angular_speed"]))
    parser.add_argument("--insert-depth", type=float, default=float(insertion_cfg["push_depth"]))
    parser.add_argument("--insert-duration", type=float, default=float(insertion_cfg["duration"]))
    parser.add_argument("--handoff-duration", type=float, default=float(insertion_cfg["handoff_duration"]))
    parser.add_argument("--wiggle-amplitude", type=float, default=float(insertion_cfg["wiggle_amplitude"]))
    parser.add_argument("--screw-amplitude", type=float, default=float(insertion_cfg["screw_amplitude_rad"]))
    parser.add_argument("--screw-turns", type=float, default=float(insertion_cfg["screw_turns"]))
    parser.add_argument("--success-depth", type=float, default=float(insertion_cfg["success_depth"]))
    parser.add_argument("--target-force", type=float, default=float(force_cfg["target_force"]))
    parser.add_argument("--alignment-tolerance", type=float, default=float(geom_cfg["alignment_tolerance"]))
    parser.add_argument("--contact-tolerance", type=float, default=float(contact_cfg["contact_tolerance"]))
    parser.add_argument("--ik-tolerance", type=float, default=0.002)
    parser.add_argument("--vision-mode", choices=("random", "mujoco-oracle", "mujoco-unet"), default="random")
    parser.add_argument("--vision-camera", type=str, default=str(vision_cfg.get("camera", "vision_top")))
    parser.add_argument("--vision-width", type=int, default=int(vision_cfg.get("width", 224)))
    parser.add_argument("--vision-height", type=int, default=int(vision_cfg.get("height", 224)))
    parser.add_argument("--vision-crop-size", type=int, default=int(vision_cfg.get("crop_size", 224)))
    parser.add_argument("--vision-crop-hole-scale", type=float, default=float(vision_cfg.get("crop_hole_scale", 5.0)))
    parser.add_argument("--vision-hole-size", type=float, default=float(vision_cfg.get("hole_size_world", 0.06)))
    parser.add_argument("--vision-pixel-noise", type=float, default=float(vision_cfg.get("pixel_noise_std", 2.0)))
    parser.add_argument("--vision-model", type=str, default=vision_cfg.get("model"))
    parser.add_argument("--vision-device", type=str, default="cpu")
    parser.add_argument("--offset-mode", choices=("random", "fixed"), default="random")
    parser.add_argument(
        "--offset-radius",
        type=float,
        default=min(float(np.linalg.norm(geom_cfg["initial_search_offset_xy"])), float(search_cfg["radius_max"])),
        help="Maximum random coarse-vision XY error in meters.",
    )
    parser.add_argument("--seed", type=int, default=None, help="Seed for random initial XY offset.")
    parser.add_argument(
        "--initial-offset-xy",
        type=float,
        nargs=2,
        default=None,
        metavar=("DX", "DY"),
        help="Use a fixed initial XY offset and bypass random offset sampling.",
    )
    parser.add_argument("--log-stem", type=str, default="peg_in_hole")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print("=" * 80)
    print("Peg-in-Hole Paper Algorithm Reproduction Demo")
    print("=" * 80)
    print(f"Model: {PROJECT_ROOT / args.model}")
    print(f"Headless: {args.headless}")
    print(f"Vision mode={args.vision_mode}, camera={args.vision_camera}, pixel_noise={args.vision_pixel_noise:.3f}")
    print(f"Offset mode={args.offset_mode}, offset_radius={args.offset_radius:.4f}, seed={args.seed}")
    print(f"Spiral radius={args.search_radius:.4f}, pitch={args.spiral_pitch:.4f}, angular_speed={args.spiral_angular_speed:.4f}")
    print(f"Target force={args.target_force:.3f} N, insert_depth={args.insert_depth:.4f} m")
    print("Paper mapping: reaching=pushing, searching=pushing+rubbing, inserting=pushing+wiggling+screwing")

    if args.headless:
        result = run_demo(args, viewer=None)
    else:
        api = ArmPlatformAPI((PROJECT_ROOT / args.model).resolve())
        with mujoco.viewer.launch_passive(api.model, api.data) as viewer:
            viewer.cam.distance = 1.2
            result = run_demo(args, viewer=viewer, api=api)

    print("=" * 80)
    print(f"Final state: {result.final_state.name}")
    print(f"Final XY alignment error: {result.final_alignment_error:.6f} m")
    print(f"Final insertion depth: {result.final_insertion_depth:.6f} m")
    print(f"Max search radius used: {result.max_search_radius_used:.6f} m")
    print(f"Vision source: {result.vision_estimate.source}")
    print(f"Estimated hole world: {array_str(result.vision_estimate.hole_world_est, precision=6)} m")
    print(f"Estimated peg world: {array_str(result.vision_estimate.peg_world_est, precision=6)} m")
    print(f"Hole pixel estimate: {array_str(result.vision_estimate.hole_pixel, precision=3)}")
    print(f"Peg pixel estimate: {array_str(result.vision_estimate.peg_pixel, precision=3)}")
    print(f"Sampled initial XY offset: {array_str(result.initial_offset_xy, precision=6)} m")
    print(f"Hole XY radius from base: {result.hole_xy_radius:.6f} m")
    print(f"Final shaft/tool axis: {array_str(result.final_shaft_axis, precision=6)}")
    print(f"Final tool tilt from world vertical: {result.final_tool_tilt_deg:.3f} deg")
    print(f"Report: {result.report_path}")
    print(f"CSV log: {result.csv_path}")
    print(f"Spiral trace: {result.spiral_trace_path}")
    if result.vision_debug_path is not None:
        print(f"Vision keypoint overlay: {result.vision_debug_path}")
    print(f"Result: {'PASSED' if result.passed else 'FAILED'}")
    print("=" * 80)

    if not result.passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
