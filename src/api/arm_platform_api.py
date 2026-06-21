# -*- coding: utf-8 -*-
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Literal, Dict, Any

import numpy as np
import mujoco
import yaml

from kinematics.robot_kinematics import RobotKinematics, KinematicsResult
from control.joint_space_controller import JointSpaceController
from gripper.parallel_gripper import ParallelGripper
from planning.joint_trajectory import JointTrajectory


TrajectoryType = Literal["linear", "cubic", "quintic"]


@dataclass
class RobotState:
    time: float
    q_arm: np.ndarray
    qvel_arm: np.ndarray
    gripper_opening: float
    ee_pos: np.ndarray
    ctrl: np.ndarray


class ArmPlatformAPI:
    """Single high-level interface for algorithm developers.

    Algorithm group should use this class instead of directly touching MuJoCo
    internals. This keeps the model, gripper, IK, trajectory and controller
    replaceable.

    Main calls:
        get_state()
        fk(q_arm)
        ik_position(target_pos)
        set_arm_target(q_arm)
        set_gripper(opening)
        plan_joint_trajectory(q_start, q_goal, duration, method)
        step()
    """

    def __init__(self, xml_path: str | Path, config_path: str | Path | None = None):
        self.xml_path = Path(xml_path)
        self.model = mujoco.MjModel.from_xml_path(str(self.xml_path))
        self.data = mujoco.MjData(self.model)

        self.config = self._load_config(config_path)
        if config_path is None and not self._config_matches_model(self.config):
            legacy_config = Path(__file__).resolve().parents[2] / "configs" / "robot_description_v2.yaml"
            if legacy_config.exists():
                legacy = self._load_config(legacy_config)
                if self._config_matches_model(legacy):
                    self.config = legacy

        self.arm_joint_names = self.config["arm"]["joints"]
        gripper_cfg = self.config["gripper"]
        gripper_master_joint = gripper_cfg["master_joint"]
        gripper_actuator = gripper_cfg["actuator"]
        gripper_mimic_joint = gripper_cfg.get("mimic_joint")
        gripper_mimic_actuator = gripper_cfg.get("mimic_actuator")
        mimic_cfg = gripper_cfg.get("mimic", {})

        self.arm_controller = JointSpaceController(
            self.model,
            self.data,
            self.arm_joint_names,
            kp=self._fit_gain([260.0, 260.0, 520.0, 520.0, 3000.0, 120.0], len(self.arm_joint_names)),
            kd=self._fit_gain([42.0, 52.0, 72.0, 72.0, 120.0, 28.0], len(self.arm_joint_names)),
            joint_ki=self._fit_gain([1.5, 1.5, 3.0, 3.0, 14.0, 0.3], len(self.arm_joint_names)),
            task_site_name=self.config["sites"]["end_effector"],
            task_kp_pos=[420.0, 420.0, 430.0],
            task_ki_pos=[5.0, 5.0, 16.0],
            task_kd_pos=[56.0, 56.0, 64.0],
            task_kp_rot=[78.0, 78.0, 46.0],
            task_ki_rot=[0.9, 0.9, 0.35],
            task_kd_rot=[11.0, 11.0, 7.5],
            damping_shape=self._fit_gain([3.2, 7.5, 9.0, 9.0, 38.0, 6.5], len(self.arm_joint_names)),
            friction_coeff=self._fit_gain([0.16, 0.16, 0.24, 0.24, 0.95, 0.12], len(self.arm_joint_names)),
        )

        gripper_joints = [gripper_master_joint]
        gripper_actuators = [gripper_actuator]
        mimic_multiplier = float(mimic_cfg.get("multiplier", -1.0))
        mimic_offset = float(mimic_cfg.get("offset", 0.0))
        if gripper_mimic_joint and gripper_mimic_actuator:
            gripper_joints.append(gripper_mimic_joint)
            gripper_actuators.append(gripper_mimic_actuator)

        self.gripper_controller = JointSpaceController(
            self.model,
            self.data,
            gripper_joints,
            gripper_actuators,
            kp=[1200.0, 1200.0][: len(gripper_joints)],
            kd=[95.0, 95.0][: len(gripper_joints)],
            bias_compensation=True,
        )
        self.gripper = ParallelGripper(
            self.gripper_controller,
            open_command=float(gripper_cfg.get("open_command", gripper_cfg.get("opening_range", {}).get("min", 0.0))),
            close_command=float(gripper_cfg.get("close_command", gripper_cfg.get("opening_range", {}).get("max", 0.065))),
            mimic_multiplier=mimic_multiplier if len(gripper_joints) > 1 else None,
            mimic_offset=mimic_offset,
        )

        self.kin = RobotKinematics(
            self.model,
            self.data,
            arm_joint_names=self.arm_joint_names,
            ee_site_name=self.config["sites"]["end_effector"],
        )

        self.active_trajectory: Optional[JointTrajectory] = None
        self.trajectory_start_time: float = 0.0
        self.arm_target = self.arm_controller.get_q()
        mujoco.mj_forward(self.model, self.data)

    def _load_config(self, config_path: str | Path | None) -> dict[str, Any]:
        if config_path is None:
            project_root = Path(__file__).resolve().parents[2]
            config_path = project_root / "configs" / "robot_description.yaml"

        path = Path(config_path)
        if path.exists():
            with path.open("r", encoding="utf-8") as f:
                return yaml.safe_load(f)

        return {
            "arm": {"joints": ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]},
            "gripper": {
                "master_joint": "joint71",
                "mimic_joint": "joint72",
                "actuator": "gripper_opening_left",
                "mimic_actuator": "gripper_opening_right",
                "mimic": {"multiplier": -1.0, "offset": 0.0},
            },
            "sites": {"end_effector": "ee_site"},
        }

    def _config_matches_model(self, config: dict[str, Any]) -> bool:
        joint_names = list(config.get("arm", {}).get("joints", []))
        gripper_cfg = config.get("gripper", {})
        joint_names.append(gripper_cfg.get("master_joint", ""))
        if gripper_cfg.get("mimic_actuator"):
            joint_names.append(gripper_cfg.get("mimic_joint", ""))

        for name in joint_names:
            if name and mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name) < 0:
                return False

        actuator_names = [gripper_cfg.get("actuator", "")]
        if gripper_cfg.get("mimic_actuator"):
            actuator_names.append(gripper_cfg.get("mimic_actuator", ""))
        for name in actuator_names:
            if name and mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name) < 0:
                return False

        site_name = config.get("sites", {}).get("end_effector", "")
        return not site_name or mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, site_name) >= 0

    @staticmethod
    def _fit_gain(values: Sequence[float], size: int) -> list[float]:
        values = list(values)
        if len(values) >= size:
            return values[:size]
        return values + [values[-1]] * (size - len(values))

    def reset(self) -> None:
        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)
        self.active_trajectory = None
        self.arm_target = self.arm_controller.get_q()

    def get_state(self) -> RobotState:
        fk = self.kin.forward_kinematics()
        return RobotState(
            time=float(self.data.time),
            q_arm=self.arm_controller.get_q(),
            qvel_arm=self.arm_controller.get_qvel(),
            gripper_opening=float(self.gripper.get_opening()),
            ee_pos=fk["position"],
            ctrl=self.data.ctrl.copy(),
        )

    def fk(self, q_arm: Optional[Sequence[float]] = None, site_name: str = "ee_site") -> Dict[str, np.ndarray]:
        return self.kin.forward_kinematics(q_arm=q_arm, site_name=site_name)

    def ik_position(
        self,
        target_pos: Sequence[float],
        q_init: Optional[Sequence[float]] = None,
        site_name: str = "ee_site",
    ) -> KinematicsResult:
        return self.kin.solve_ik_position(target_pos, q_init=q_init, site_name=site_name)

    def set_arm_target(self, q_arm: Sequence[float]) -> np.ndarray:
        self.active_trajectory = None
        self.arm_target = self.arm_controller.set_target(q_arm)
        return self.arm_target

    def set_gripper(self, opening: float) -> np.ndarray:
        return self.gripper.set_opening(opening)

    def open_gripper(self) -> np.ndarray:
        return self.gripper.open()

    def close_gripper(self) -> np.ndarray:
        return self.gripper.close()

    def plan_joint_trajectory(
        self,
        q_goal: Sequence[float],
        duration: float = 4.0,
        method: TrajectoryType = "quintic",
        q_start: Optional[Sequence[float]] = None,
    ) -> None:
        if q_start is None:
            q_start = self.arm_controller.get_q()

        self.active_trajectory = JointTrajectory(q_start, q_goal, duration, method)
        self.trajectory_start_time = float(self.data.time)

    def update_control(self) -> None:
        if self.active_trajectory is not None:
            t_local = float(self.data.time) - self.trajectory_start_time
            q_des = self.active_trajectory.sample(t_local)
            self.arm_target = self.arm_controller.set_target(q_des)

            if t_local >= self.active_trajectory.duration:
                self.active_trajectory = None
        else:
            self.arm_controller.update()
        self.gripper_controller.update()

    def step(self, n: int = 1) -> None:
        for _ in range(n):
            self.update_control()
            mujoco.mj_step(self.model, self.data)
