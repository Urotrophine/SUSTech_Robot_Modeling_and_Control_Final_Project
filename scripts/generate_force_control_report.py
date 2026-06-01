# -*- coding: utf-8 -*-
"""Generate quantitative matplotlib charts for the force-control demo.

Run:
    python scripts/generate_force_control_report.py --seed 1
"""

from __future__ import annotations

import argparse
import builtins
import csv
import importlib.util
import math
import os
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mujoco  # noqa: E402
import numpy as np  # noqa: E402


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEMO_PATH = PROJECT_ROOT / "scripts" / "run_v6_peg_in_hole_spiral_search_demo.py"
REPORT_DIR = PROJECT_ROOT / "reports"


def load_demo_module():
    spec = importlib.util.spec_from_file_location("peg_spiral_demo", DEMO_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load demo module: {DEMO_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def norm(values: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(values, dtype=float)))


class MetricRecorder:
    def __init__(self, demo) -> None:
        self.demo = demo
        self.rows: list[dict[str, Any]] = []
        self.phase = "initialization"
        self.grasp_lock_active = False
        self.original_print = builtins.print
        self.original_apply_grasp_lock = demo.apply_grasp_lock

    def patch(self) -> None:
        def patched_print(*args, **kwargs):
            text = " ".join(str(arg) for arg in args)
            if text.startswith("Phase:"):
                self.phase = text.replace("Phase:", "", 1).strip()
            return self.original_print(*args, **kwargs)

        def patched_apply_grasp_lock(api, grasp_lock, position_q_arm=None):
            if grasp_lock is not None:
                self.grasp_lock_active = True
            return self.original_apply_grasp_lock(api, grasp_lock, position_q_arm=position_q_arm)

        builtins.print = patched_print
        self.demo.apply_grasp_lock = patched_apply_grasp_lock
        self.demo.step = self.step

    def restore(self) -> None:
        builtins.print = self.original_print
        self.demo.apply_grasp_lock = self.original_apply_grasp_lock

    def step(self, api, viewer=None) -> None:
        api.update_control()
        mujoco.mj_step(api.model, api.data)
        q = api.arm_controller.get_q()
        qd = api.arm_controller.get_qvel()
        q_target = api.arm_controller.q_target.copy()
        ctrl = api.data.ctrl.copy()
        pos, _ = self.demo.object_pose(api)
        center_err = norm(pos[:2] - self.demo.HOLE_CENTER[:2])
        q_error = q_target - q
        row = {
            "time": float(api.data.time),
            "phase": self.phase,
            "grasp_lock_active": int(self.grasp_lock_active),
            "contact_force_n": self.demo.max_contact_force(api),
            "inserted_depth_m": self.demo.peg_inserted_depth(pos),
            "peg_x_m": float(pos[0]),
            "peg_y_m": float(pos[1]),
            "peg_z_m": float(pos[2]),
            "peg_hole_xy_error_m": center_err,
            "q1_rad": float(q[0]),
            "q2_rad": float(q[1]),
            "q3_rad": float(q[2]),
            "q4_rad": float(q[3]),
            "q5_m": float(q[4]),
            "q6_rad": float(q[5]),
            "q3_abs_rad": abs(float(q[2])),
            "q4_abs_rad": abs(float(q[3])),
            "q6_turns": float(q[5]) / (2.0 * math.pi),
            "q_tracking_error": norm(q_error),
            "q_tracking_error_without_q6": norm(q_error[:5]),
            "q34_tracking_error": norm(q_error[2:4]),
            "q6_tracking_error_rad": abs(float(q_error[5])),
            "q6_speed_rad_s": float(qd[5]),
            "ctrl_joint1": float(ctrl[0]),
            "ctrl_joint2": float(ctrl[1]),
            "ctrl_joint3": float(ctrl[2]),
            "ctrl_joint4": float(ctrl[3]),
            "ctrl_joint5": float(ctrl[4]),
            "ctrl_joint6": float(ctrl[5]),
        }
        self.rows.append(row)


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        raise RuntimeError("No metric rows recorded")
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def col(rows: list[dict[str, Any]], key: str) -> np.ndarray:
    return np.asarray([float(row[key]) for row in rows], dtype=float)


def moving_average(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or len(values) < window:
        return values
    kernel = np.ones(window, dtype=float) / float(window)
    pad_left = window // 2
    pad_right = window - 1 - pad_left
    padded = np.pad(values, (pad_left, pad_right), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def phase_mask(rows: list[dict[str, Any]], *needles: str) -> np.ndarray:
    needles_lower = tuple(n.lower() for n in needles)
    return np.asarray(
        [any(n in str(row["phase"]).lower() for n in needles_lower) for row in rows],
        dtype=bool,
    )


def plot_contact_force_and_depth(rows: list[dict[str, Any]], path: Path) -> None:
    t = col(rows, "time")
    force = col(rows, "contact_force_n")
    depth_mm = 1000.0 * col(rows, "inserted_depth_m")
    fig, ax1 = plt.subplots(figsize=(11, 5.8), dpi=150)
    ax2 = ax1.twinx()
    ax1.plot(t, force, color="#d62728", linewidth=1.2, label="contact force")
    ax2.plot(t, depth_mm, color="#1f77b4", linewidth=2.0, label="inserted depth")
    ax1.set_xlabel("time (s)")
    ax1.set_ylabel("contact force (N)", color="#d62728")
    ax2.set_ylabel("inserted depth (mm)", color="#1f77b4")
    ax1.grid(True, alpha=0.25)
    ax1.set_title("Contact Force and Insertion Depth")
    lines = ax1.get_lines() + ax2.get_lines()
    ax1.legend(lines, [line.get_label() for line in lines], loc="upper left")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_joint_stability(rows: list[dict[str, Any]], path: Path) -> None:
    t = col(rows, "time")
    fig, ax = plt.subplots(figsize=(11, 5.8), dpi=150)
    ax.plot(t, col(rows, "q3_rad"), label="joint3", linewidth=1.5)
    ax.plot(t, col(rows, "q4_rad"), label="joint4", linewidth=1.5)
    ax.plot(t, col(rows, "q6_turns"), label="joint6 turns", linewidth=1.8)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("joint value")
    ax.set_title("Joint Stability and Screwing Rotation")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_tracking_error(rows: list[dict[str, Any]], path: Path) -> None:
    t = col(rows, "time")
    fig, ax = plt.subplots(figsize=(11, 5.8), dpi=150)
    ax.plot(t, col(rows, "q_tracking_error"), label="full arm error incl. joint6", linewidth=1.3)
    ax.plot(t, col(rows, "q_tracking_error_without_q6"), label="arm error excl. joint6", linewidth=1.8)
    ax.plot(t, col(rows, "q34_tracking_error"), label="joint3/4 error", linewidth=1.8)
    ax.plot(t, col(rows, "q6_tracking_error_rad"), label="joint6 error", linewidth=1.1, alpha=0.65)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("tracking error")
    ax.set_title("Impedance Tracking Error Decomposition")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_peg_xy(rows: list[dict[str, Any]], demo, path: Path) -> None:
    mask = np.asarray([bool(row["grasp_lock_active"]) for row in rows], dtype=bool)
    mask &= phase_mask(rows, "lift", "search", "spiral", "insertion")
    if mask.sum() < 8:
        mask = np.asarray([bool(row["grasp_lock_active"]) for row in rows], dtype=bool)
    x = col(rows, "peg_x_m")[mask]
    y = col(rows, "peg_y_m")[mask]
    smooth_x = moving_average(x, 15)
    smooth_y = moving_average(y, 15)
    hx, hy = demo.HOLE_CENTER[:2]
    hole = plt.Circle((hx, hy), demo.HOLE_RADIUS, color="#1f77b4", fill=False, linewidth=2.0, label="hole")
    fig, ax = plt.subplots(figsize=(7.2, 7.2), dpi=150)
    ax.plot(x, y, color="#bbbbbb", linewidth=0.8, label="raw carried peg path")
    ax.plot(smooth_x, smooth_y, color="#d62728", linewidth=2.2, label="smoothed path")
    ax.add_patch(hole)
    ax.scatter([x[0]], [y[0]], color="#2ca02c", s=28, label="start")
    ax.scatter([x[-1]], [y[-1]], color="#111111", s=28, label="end")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title("Peg Center XY Path After Grasp Lock")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def summarize(rows: list[dict[str, Any]]) -> dict[str, float]:
    depth = col(rows, "inserted_depth_m")
    screwing = phase_mask(rows, "insertion")
    if screwing.any():
        insertion_depth_change = float(depth[screwing][-1] - depth[screwing][0])
    else:
        insertion_depth_change = float(depth[-1] - depth[0])
    return {
        "duration_s": float(rows[-1]["time"] - rows[0]["time"]),
        "max_contact_force_n": float(np.max(col(rows, "contact_force_n"))),
        "mean_contact_force_n": float(np.mean(col(rows, "contact_force_n"))),
        "final_inserted_depth_m": float(depth[-1]),
        "insertion_phase_depth_change_m": insertion_depth_change,
        "max_abs_joint3_rad": float(np.max(col(rows, "q3_abs_rad"))),
        "max_abs_joint4_rad": float(np.max(col(rows, "q4_abs_rad"))),
        "final_joint6_turns": float(col(rows, "q6_turns")[-1]),
        "max_tracking_error": float(np.max(col(rows, "q_tracking_error"))),
        "max_tracking_error_without_q6": float(np.max(col(rows, "q_tracking_error_without_q6"))),
        "max_q34_tracking_error": float(np.max(col(rows, "q34_tracking_error"))),
        "final_xy_error_m": float(col(rows, "peg_hole_xy_error_m")[-1]),
    }


def write_summary(path: Path, metrics: dict[str, float]) -> None:
    lines = ["Force/impedance-control quantitative summary", ""]
    for key, value in metrics.items():
        lines.append(f"{key}: {value:.6g}")
    lines.extend(
        [
            "",
            "Notes:",
            "- The full arm tracking error contains joint6 screwing error. The large transient at the start of insertion is expected because joint6 receives a multi-turn target.",
            "- The peg XY chart filters out initialization and grasp-establishment discontinuities, then shows the carried peg path used for search/insertion.",
            "- Insertion depth is plotted in millimeters on a separate y-axis; it is not constant during the insertion phase.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_report_md(path: Path, metrics: dict[str, float]) -> None:
    rows = ["# Force Control Evaluation", "", "## Quantitative Results", "", "| Metric | Value |", "| --- | ---: |"]
    for key, value in metrics.items():
        rows.append(f"| {key} | {value:.6g} |")
    rows.extend(
        [
            "",
            "## Charts",
            "",
            "![Contact force and insertion depth](contact_force_and_depth.png)",
            "",
            "![Joint stability](joint_stability.png)",
            "",
            "![Tracking error](tracking_error.png)",
            "",
            "![Peg center XY path](peg_xy_path.png)",
            "",
            "## Interpretation",
            "",
            "The controller is torque/force based: MuJoCo `<motor>` actuators receive generalized force commands. The impedance law computes `tau = qfrc_bias + Kp(q_des-q) + Kd(qd_des-qd)`, then clips it by actuator limits.",
            "",
            "The full tracking-error spike near insertion is caused by the deliberate multi-turn `joint6` screwing command. The chart therefore also shows tracking error excluding `joint6`, which better reflects translational insertion stability.",
            "",
            "Compared with the earlier position-control demo, force control allows contact to create tracking error and measurable contact force. This is less rigid than position control, but closer to deployable behavior because force/torque limits and damping shape the contact response.",
        ]
    )
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    REPORT_DIR.mkdir(exist_ok=True)
    demo = load_demo_module()
    recorder = MetricRecorder(demo)
    recorder.patch()
    try:
        demo.run_demo(headless=True, seed=args.seed)
    finally:
        recorder.restore()

    rows = recorder.rows
    write_csv(rows, REPORT_DIR / "force_control_metrics.csv")
    metrics = summarize(rows)
    write_summary(REPORT_DIR / "force_control_summary.txt", metrics)
    plot_contact_force_and_depth(rows, REPORT_DIR / "contact_force_and_depth.png")
    plot_joint_stability(rows, REPORT_DIR / "joint_stability.png")
    plot_tracking_error(rows, REPORT_DIR / "tracking_error.png")
    plot_peg_xy(rows, demo, REPORT_DIR / "peg_xy_path.png")
    write_report_md(REPORT_DIR / "force_control_report.md", metrics)

    print(f"Wrote report files to {REPORT_DIR}")
    for key, value in metrics.items():
        print(f"{key}: {value:.6g}")


if __name__ == "__main__":
    main()
