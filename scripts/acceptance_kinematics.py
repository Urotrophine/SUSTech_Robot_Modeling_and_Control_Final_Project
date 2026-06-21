# -*- coding: utf-8 -*-
"""
acceptance_kinematics.py

可视化正逆运动学验收文件。

运行后会直接打开 MuJoCo Viewer，并按顺序演示：

1. 当前机械臂初始位姿
2. 正运动学 FK 读取 ee_site
3. 多组可达目标的 IK 求解
4. 将 IK 解用“运动学回放”的方式显示出来

注意：
    本脚本验收的是 FK / IK / 轨迹规划接口，不验收真实动力学跟踪。
    因此动画采用直接设置 qpos + mj_forward 的方式播放，避免 position actuator、
    重力、joint4 执行器参数影响运动学验收结果。

运行：
    python scripts/acceptance_kinematics.py

输出：
    logs/acceptance_kinematics_report.md
    logs/acceptance_kinematics_log.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

import mujoco
import mujoco.viewer

from api.arm_platform_api import ArmPlatformAPI
from planning.joint_trajectory import JointTrajectory


@dataclass
class IKCaseResult:
    name: str
    q_reference: np.ndarray
    target_pos: np.ndarray
    q_solution: np.ndarray
    success: bool
    ik_error_norm: float
    fk_error_norm: float
    iterations: int
    passed: bool


def array_str(x: np.ndarray, precision: int = 6) -> str:
    return np.array2string(np.asarray(x), precision=precision, suppress_small=False)


def set_q_arm_kinematic(api: ArmPlatformAPI, q_arm: np.ndarray) -> None:
    """Directly set arm qpos for kinematic visualization."""
    api.data.qpos[api.kin.qpos_idx] = q_arm
    api.data.qvel[:] = 0.0
    mujoco.mj_forward(api.model, api.data)


def animate_q_kinematic(api: ArmPlatformAPI, viewer, q_start: np.ndarray, q_goal: np.ndarray, duration: float = 2.0) -> None:
    traj = JointTrajectory(q_start, q_goal, duration, method="quintic")

    t0 = time.time()
    while viewer.is_running():
        t = time.time() - t0
        q = traj.sample(min(t, duration))
        set_q_arm_kinematic(api, q)
        viewer.sync()
        time.sleep(0.01)
        if t >= duration:
            break


def hold(api: ArmPlatformAPI, viewer, seconds: float = 0.8) -> None:
    t0 = time.time()
    while viewer.is_running() and time.time() - t0 < seconds:
        mujoco.mj_forward(api.model, api.data)
        viewer.sync()
        time.sleep(0.01)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="models/robot_with_gripper.xml")
    parser.add_argument("--fk-threshold", type=float, default=5e-3)
    args = parser.parse_args()

    model_path = (PROJECT_ROOT / args.model).resolve()
    log_dir = PROJECT_ROOT / "logs"
    log_dir.mkdir(exist_ok=True)

    report_path = log_dir / "acceptance_kinematics_report.md"
    csv_path = log_dir / "acceptance_kinematics_log.csv"

    api = ArmPlatformAPI(model_path)

    state0 = api.get_state()
    q0 = state0.q_arm.copy()
    ee0 = state0.ee_pos.copy()

    print("=" * 80)
    print("Visual Kinematics Acceptance Test")
    print("=" * 80)
    print(f"Model: {model_path}")
    print(f"Initial q_arm: {array_str(q0)}")
    print(f"Initial ee_pos: {array_str(ee0)}")

    test_q_list = [
        q0 + np.array([+0.10, -0.08, +0.06, +0.010]),
        q0 + np.array([-0.08, +0.10, -0.05, -0.010]),
        q0 + np.array([+0.15, +0.06, -0.08, +0.015]),
        q0 + np.array([-0.12, -0.05, +0.10, -0.015]),
    ]

    low, high = api.arm_controller.limits()
    test_q_list = [np.minimum(np.maximum(q, low), high) for q in test_q_list]

    results: List[IKCaseResult] = []

    for idx, q_ref in enumerate(test_q_list, start=1):
        case_name = f"reachable_fk_target_{idx}"
        fk_ref = api.fk(q_ref)
        target_pos = fk_ref["position"]

        ik_result = api.ik_position(target_pos, q_init=q0)
        fk_solution = api.fk(ik_result.q)
        fk_error = float(np.linalg.norm(fk_solution["position"] - target_pos))

        passed = bool(ik_result.success and fk_error <= args.fk_threshold)

        res = IKCaseResult(
            name=case_name,
            q_reference=q_ref.copy(),
            target_pos=target_pos.copy(),
            q_solution=ik_result.q.copy(),
            success=bool(ik_result.success),
            ik_error_norm=float(ik_result.error_norm),
            fk_error_norm=fk_error,
            iterations=int(ik_result.iterations),
            passed=passed,
        )
        results.append(res)

        print()
        print(f"[{case_name}]")
        print(f"  q_reference : {array_str(res.q_reference)}")
        print(f"  target_pos  : {array_str(res.target_pos)}")
        print(f"  q_solution  : {array_str(res.q_solution)}")
        print(f"  IK success  : {res.success}")
        print(f"  IK error    : {res.ik_error_norm:.6e} m")
        print(f"  FK error    : {res.fk_error_norm:.6e} m")
        print(f"  iterations  : {res.iterations}")
        print(f"  passed      : {res.passed}")

    overall_passed = all(r.passed for r in results)

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "case",
            "passed",
            "ik_success",
            "ik_error_norm_m",
            "fk_error_norm_m",
            "iterations",
            "target_x",
            "target_y",
            "target_z",
            "qsol_1",
            "qsol_2",
            "qsol_3",
            "qsol_4",
        ])
        for r in results:
            writer.writerow([
                r.name,
                r.passed,
                r.success,
                r.ik_error_norm,
                r.fk_error_norm,
                r.iterations,
                r.target_pos[0],
                r.target_pos[1],
                r.target_pos[2],
                r.q_solution[0],
                r.q_solution[1],
                r.q_solution[2],
                r.q_solution[3],
            ])

    lines = []
    lines.append("# 可视化正逆运动学验收报告\n")
    lines.append(f"- 模型文件：`{model_path}`")
    lines.append(f"- 初始关节：`{array_str(q0)}`")
    lines.append(f"- 初始末端位置：`{array_str(ee0)}`")
    lines.append(f"- FK 误差阈值：`{args.fk_threshold}` m")
    lines.append(f"- 总体验收结果：**{'通过' if overall_passed else '未通过'}**\n")

    lines.append("## IK 测试结果\n")
    lines.append("| Case | Pass | IK success | IK error (m) | FK error (m) | Iterations |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for r in results:
        lines.append(
            f"| {r.name} | {r.passed} | {r.success} | "
            f"{r.ik_error_norm:.6e} | {r.fk_error_norm:.6e} | {r.iterations} |"
        )

    lines.append("\n## 说明\n")
    lines.append(
        "本脚本会直接打开 MuJoCo Viewer，并用 qpos + mj_forward 进行运动学回放。"
        "因此它验收的是 FK、IK 和轨迹规划接口，不把动力学执行器跟踪误差算作失败。"
    )
    report_path.write_text("\n".join(lines), encoding="utf-8")

    print()
    print("=" * 80)
    print(f"Overall result: {'PASSED' if overall_passed else 'FAILED'}")
    print(f"Report: {report_path}")
    print(f"CSV log: {csv_path}")
    print("=" * 80)
    print("Opening MuJoCo Viewer for visual kinematics acceptance...")

    # Visual playback.
    set_q_arm_kinematic(api, q0)

    with mujoco.viewer.launch_passive(api.model, api.data) as viewer:
        viewer.cam.distance = 1.4
        hold(api, viewer, 1.0)

        q_current = q0.copy()
        for r in results:
            if not viewer.is_running():
                break
            print(f"[Viewer] Showing IK result: {r.name}")
            animate_q_kinematic(api, viewer, q_current, r.q_solution, duration=2.0)
            hold(api, viewer, 0.8)
            q_current = r.q_solution.copy()

        print("[Viewer] Returning to initial pose.")
        animate_q_kinematic(api, viewer, q_current, q0, duration=2.0)
        hold(api, viewer, 2.0)

    if not overall_passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
