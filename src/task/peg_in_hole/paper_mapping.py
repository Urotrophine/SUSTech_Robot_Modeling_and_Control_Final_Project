# -*- coding: utf-8 -*-
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PaperPhase:
    procedure_step: str
    unit_motion: str
    condition: str


PHASE_TO_PAPER = {
    "initial_grasp_pose": PaperPhase("setup", "initial peg grasp pose", "not a paper transition"),
    "close_initial_grasp": PaperPhase("setup", "close gripper and logical grasp lock", "not a paper transition"),
    "carry_to_pre_insert": PaperPhase("setup", "carry grasped peg to estimated pre-insert pose", "not a paper transition"),
    "approach_hole": PaperPhase("setup", "coarse visual approach", "not a paper transition"),
    "grasp_peg": PaperPhase("setup", "logical grasp lock", "not a paper transition"),
    "move_to_pre_insert": PaperPhase("setup", "move to estimated pre-insert pose", "not a paper transition"),
    "reach_contact_plane": PaperPhase("reaching", "pushing", "contact near estimated hole plane"),
    "reaching_pushing": PaperPhase("reaching", "pushing", "contact near estimated hole plane"),
    "joint4_admittance_contact": PaperPhase("reaching", "pushing", "velocity/contact condition"),
    "reaching_pushing_velocity_contact": PaperPhase("reaching", "pushing", "velocity/contact condition"),
    "archimedes_spiral_search": PaperPhase(
        "searching",
        "pushing + rubbing / Archimedes spiral",
        "alignment or stopped-contact condition",
    ),
    "searching_rubbing_spiral": PaperPhase(
        "searching",
        "pushing + rubbing / Archimedes spiral",
        "alignment or stopped-contact condition",
    ),
    "wiggle_screw_insert": PaperPhase(
        "inserting",
        "pushing + wiggling + screwing",
        "alignment and insertion-depth condition",
    ),
    "inserting_wiggle_screw": PaperPhase(
        "inserting",
        "pushing + wiggling + screwing",
        "alignment and insertion-depth condition",
    ),
    "final": PaperPhase("complete", "task evaluation", "final acceptance criteria"),
}


STATE_TO_PAPER = {
    "INITIAL_GRASP": PaperPhase("setup", "initial peg grasp pose", "not a paper transition"),
    "REACHING_PUSH": PaperPhase("reaching", "pushing", "contact near estimated hole plane"),
    "SEARCHING_SPIRAL": PaperPhase(
        "searching",
        "pushing + rubbing / Archimedes spiral",
        "alignment or stopped-contact condition",
    ),
    "INSERTING_WIGGLE_SCREW": PaperPhase(
        "inserting",
        "pushing + wiggling + screwing",
        "alignment and insertion-depth condition",
    ),
    "COMPLETE": PaperPhase("complete", "task evaluation", "final acceptance criteria"),
    "FAILED": PaperPhase("failed", "task evaluation", "failure condition"),
}


def classify_demo_phase(phase: str, state_name: str = "") -> PaperPhase:
    if phase in PHASE_TO_PAPER:
        return PHASE_TO_PAPER[phase]
    if state_name in STATE_TO_PAPER:
        return STATE_TO_PAPER[state_name]
    return PaperPhase("setup", "demo support motion", "not a paper transition")


def algorithm_reproduction_notes() -> list[str]:
    return [
        "This is an algorithm-level reproduction of the paper workflow.",
        "The coarse vision estimate can come from random XY error simulation or a MuJoCo camera keypoint projection.",
        "Pushing, rubbing, wiggling, and screwing are represented as task-level motion primitives.",
        "The demo uses deterministic geometric contact cues instead of a full physical contact-state estimator.",
        "It does not reproduce Kinect recognition, torque-level compliant control, or 0.01 mm clearance experiments.",
    ]
