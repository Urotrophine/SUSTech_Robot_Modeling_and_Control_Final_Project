# -*- coding: utf-8 -*-
"""
Run a small algorithm-level parameter sweep for the peg-in-hole paper demo.

The sweep varies random simulated vision offsets, spiral radius/pitch, and
wiggling/screwing insertion parameters. Each case writes its own demo log and
report, plus a combined summary:

    logs/peg_in_hole_parameter_sweep.csv
    logs/peg_in_hole_parameter_sweep.md
"""

from __future__ import annotations

import argparse
import csv
from argparse import Namespace
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from run_peg_in_hole_demo import PROJECT_ROOT, load_task_config, run_demo


@dataclass(frozen=True)
class SweepCase:
    name: str
    seed: int
    offset_radius: float
    search_radius: float
    spiral_pitch: float
    spiral_angular_speed: float
    wiggle_amplitude: float
    screw_turns: float


def default_cases(cfg: dict) -> list[SweepCase]:
    geom = cfg["geometry"]
    search = cfg["search"]
    insertion = cfg["insertion"]
    base_offset_radius = min(
        float((sum(float(x) ** 2 for x in geom["initial_search_offset_xy"])) ** 0.5),
        float(search["radius_max"]),
    )
    base_radius = float(search["radius_max"])
    base_pitch = float(search["pitch"])
    base_speed = float(search["angular_speed"])
    base_wiggle = float(insertion["wiggle_amplitude"])
    base_turns = float(insertion["screw_turns"])

    return [
        SweepCase("random_seed_1", 1, base_offset_radius, base_radius, base_pitch, base_speed, base_wiggle, base_turns),
        SweepCase("random_seed_2", 2, base_offset_radius, base_radius, base_pitch, base_speed, base_wiggle, base_turns),
        SweepCase("random_seed_3", 3, base_offset_radius, base_radius, base_pitch, base_speed, base_wiggle, base_turns),
        SweepCase("search_radius_small", 4, base_offset_radius, 0.020, base_pitch, base_speed, base_wiggle, base_turns),
        SweepCase("spiral_pitch_coarse", 5, base_offset_radius, base_radius, 0.004, base_speed, base_wiggle, base_turns),
        SweepCase("wiggle_screw_variant", 6, base_offset_radius, base_radius, base_pitch, base_speed, 0.003, 4.0),
    ]


def args_for_case(cfg: dict, cli: argparse.Namespace, case: SweepCase) -> Namespace:
    insertion = cfg["insertion"]
    force = cfg["force_control"]
    contact = cfg["contact_detection"]
    geom = cfg["geometry"]
    vision = cfg.get("vision", {})
    return Namespace(
        model=cfg["model_xml"],
        headless=True,
        no_sleep=True,
        dt=float(cli.dt),
        max_time=float(cli.max_time),
        search_radius=float(case.search_radius),
        spiral_pitch=float(case.spiral_pitch),
        spiral_angular_speed=float(case.spiral_angular_speed),
        insert_depth=float(insertion["push_depth"]),
        insert_duration=float(insertion["duration"]),
        handoff_duration=float(insertion["handoff_duration"]),
        wiggle_amplitude=float(case.wiggle_amplitude),
        screw_amplitude=float(insertion["screw_amplitude_rad"]),
        screw_turns=float(case.screw_turns),
        success_depth=float(insertion["success_depth"]),
        target_force=float(force["target_force"]),
        alignment_tolerance=float(geom["alignment_tolerance"]),
        contact_tolerance=float(contact["contact_tolerance"]),
        ik_tolerance=float(cli.ik_tolerance),
        vision_mode="random",
        vision_camera=str(vision.get("camera", "vision_top")),
        vision_width=int(vision.get("width", 224)),
        vision_height=int(vision.get("height", 224)),
        vision_crop_size=int(vision.get("crop_size", 224)),
        vision_crop_hole_scale=float(vision.get("crop_hole_scale", 5.0)),
        vision_hole_size=float(vision.get("hole_size_world", 0.06)),
        vision_pixel_noise=float(vision.get("pixel_noise_std", 2.0)),
        vision_model=vision.get("model"),
        vision_device="cpu",
        offset_mode="random",
        offset_radius=float(case.offset_radius),
        seed=int(case.seed),
        initial_offset_xy=None,
        log_stem=f"peg_in_hole_sweep_{case.name}",
    )


def write_summary(rows: list[dict[str, Any]], csv_path: Path, md_path: Path) -> None:
    fieldnames = [
        "case",
        "passed",
        "final_state",
        "failure_reason",
        "seed",
        "initial_offset_x",
        "initial_offset_y",
        "offset_radius",
        "search_radius",
        "spiral_pitch",
        "spiral_angular_speed",
        "wiggle_amplitude",
        "screw_turns",
        "final_alignment_error",
        "final_insertion_depth",
        "max_search_radius_used",
        "report",
        "csv_log",
        "spiral_trace",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    passed = sum(1 for row in rows if row["passed"])
    lines = [
        "# Peg-in-Hole Parameter Sweep",
        "",
        f"- Reproduction level: `algorithm-level simulation`",
        f"- Cases passed: `{passed}/{len(rows)}`",
        "- Purpose: compare random simulated vision offsets, Archimedes spiral parameters, and wiggling/screwing variants.",
        "",
        "| Case | Seed | Offset XY (m) | Pass | Final state | XY error (m) | Insertion depth (m) | Search radius | Pitch | Wiggle | Screw turns |",
        "| --- | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| {case} | {seed} | ({initial_offset_x:+.4f}, {initial_offset_y:+.4f}) | {passed} | {final_state} | {final_alignment_error:.6f} | "
            "{final_insertion_depth:.6f} | {search_radius:.3f} | {spiral_pitch:.4f} | "
            "{wiggle_amplitude:.4f} | {screw_turns:.1f} |".format(**row)
        )
    lines.extend([
        "",
        "## Notes",
        "",
        "This sweep reuses the same task-level demo. It is not a torque-control or high-fidelity contact reproduction.",
    ])
    md_path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dt", type=float, default=0.02)
    parser.add_argument("--max-time", type=float, default=30.0)
    parser.add_argument("--ik-tolerance", type=float, default=0.002)
    parser.add_argument("--require-all-pass", action="store_true")
    return parser.parse_args()


def main() -> None:
    cli = parse_args()
    cfg = load_task_config(PROJECT_ROOT / "configs" / "peg_in_hole_task.yaml")
    log_dir = PROJECT_ROOT / "logs"
    log_dir.mkdir(exist_ok=True)

    rows: list[dict[str, Any]] = []
    for case in default_cases(cfg):
        args = args_for_case(cfg, cli, case)
        try:
            result = run_demo(args, viewer=None)
            row = {
                "case": case.name,
                "passed": bool(result.passed),
                "final_state": result.final_state.name,
                "failure_reason": result.failure_reason or "",
                "final_alignment_error": float(result.final_alignment_error),
                "final_insertion_depth": float(result.final_insertion_depth),
                "max_search_radius_used": float(result.max_search_radius_used),
                "initial_offset_x": float(result.initial_offset_xy[0]),
                "initial_offset_y": float(result.initial_offset_xy[1]),
                "report": str(result.report_path),
                "csv_log": str(result.csv_path),
                "spiral_trace": str(result.spiral_trace_path),
            }
        except Exception as exc:  # noqa: BLE001 - sweep records per-case failures.
            row = {
                "case": case.name,
                "passed": False,
                "final_state": "ERROR",
                "failure_reason": str(exc),
                "final_alignment_error": float("nan"),
                "final_insertion_depth": float("nan"),
                "max_search_radius_used": float("nan"),
                "initial_offset_x": float("nan"),
                "initial_offset_y": float("nan"),
                "report": "",
                "csv_log": "",
                "spiral_trace": "",
            }
        row.update({
            "seed": int(case.seed),
            "offset_radius": float(case.offset_radius),
            "search_radius": float(case.search_radius),
            "spiral_pitch": float(case.spiral_pitch),
            "spiral_angular_speed": float(case.spiral_angular_speed),
            "wiggle_amplitude": float(case.wiggle_amplitude),
            "screw_turns": float(case.screw_turns),
        })
        rows.append(row)
        print(
            f"{case.name}: {'PASSED' if row['passed'] else 'FAILED'} "
            f"state={row['final_state']} error={row['final_alignment_error']:.6f}"
        )

    csv_path = log_dir / "peg_in_hole_parameter_sweep.csv"
    md_path = log_dir / "peg_in_hole_parameter_sweep.md"
    write_summary(rows, csv_path, md_path)
    print(f"Summary: {md_path}")
    print(f"CSV: {csv_path}")

    if cli.require_all_pass and not all(row["passed"] for row in rows):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
