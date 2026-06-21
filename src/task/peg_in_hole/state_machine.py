# -*- coding: utf-8 -*-
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto


class PegInHoleState(Enum):
    IDLE = auto()
    INITIAL_GRASP = auto()
    GRASP_PEG = auto()
    MOVE_TO_PRE_INSERT = auto()
    REACHING_PUSH = auto()
    SEARCHING_SPIRAL = auto()
    INSERTING_WIGGLE_SCREW = auto()
    COMPLETE = auto()
    FAILED = auto()


class PegInHoleStateMachine:
    """Task-level state machine for the peg-in-hole demo.

    Motion generation is intentionally kept outside this class. The demo feeds
    back simple facts through PegInHoleContext, and the state machine decides
    when the task advances or fails.
    """

    def __init__(self):
        self.state = PegInHoleState.IDLE
        self.failure_reason = ""

    def reset(self):
        self.state = PegInHoleState.IDLE
        self.failure_reason = ""

    def step(self, context):
        if self.state in (PegInHoleState.COMPLETE, PegInHoleState.FAILED):
            return self.state

        if context.ik_failed:
            return self._fail(context, "ik_failed")

        if self.state == PegInHoleState.IDLE:
            self.state = PegInHoleState.INITIAL_GRASP
        elif self.state == PegInHoleState.INITIAL_GRASP:
            if context.approach_done:
                self.state = PegInHoleState.GRASP_PEG
        elif self.state == PegInHoleState.GRASP_PEG:
            if context.grasp_done:
                self.state = PegInHoleState.MOVE_TO_PRE_INSERT
        elif self.state == PegInHoleState.MOVE_TO_PRE_INSERT:
            if context.pre_insert_done:
                self.state = PegInHoleState.REACHING_PUSH
        elif self.state == PegInHoleState.REACHING_PUSH:
            if context.contact_detected:
                self.state = PegInHoleState.SEARCHING_SPIRAL
        elif self.state == PegInHoleState.SEARCHING_SPIRAL:
            if context.alignment_error_xy <= context.alignment_tolerance:
                self.state = PegInHoleState.INSERTING_WIGGLE_SCREW
            elif context.search_elapsed >= context.search_timeout:
                return self._fail(context, "search_timeout")
            elif context.search_radius >= context.max_search_radius and context.alignment_error_xy > context.alignment_tolerance:
                return self._fail(context, "search_radius_exceeded")
        elif self.state == PegInHoleState.INSERTING_WIGGLE_SCREW:
            aligned = context.alignment_error_xy <= context.alignment_tolerance
            inserted = context.insertion_depth >= context.insert_depth_target
            if aligned and inserted:
                self.state = PegInHoleState.COMPLETE
            elif context.state_elapsed >= context.insert_timeout:
                return self._fail(context, "insert_timeout")

        return self.state

    def _fail(self, context, reason: str):
        self.state = PegInHoleState.FAILED
        self.failure_reason = reason
        context.failure_reason = reason
        return self.state


@dataclass
class PegInHoleContext:
    time: float = 0.0
    state_elapsed: float = 0.0
    approach_done: bool = False
    grasp_done: bool = False
    pre_insert_done: bool = False
    contact_detected: bool = False
    contact_force: float = 0.0
    alignment_error_xy: float = float("inf")
    alignment_tolerance: float = 0.004
    insertion_depth: float = 0.0
    insert_depth_target: float = 0.025
    insert_timeout: float = 8.0
    search_elapsed: float = 0.0
    search_timeout: float = 30.0
    search_radius: float = 0.0
    max_search_radius: float = 0.03
    ik_failed: bool = False
    failure_reason: str = ""
