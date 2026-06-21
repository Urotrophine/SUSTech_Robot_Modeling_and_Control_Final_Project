# -*- coding: utf-8 -*-
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import mujoco

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from planning.screw_motion import ScrewMotion
from planning.spiral_search import ArchimedesSpiralSearch
from planning.vision_error import sample_random_xy_offset
from task.peg_in_hole.paper_mapping import classify_demo_phase
from task.peg_in_hole.spiral_trace import SpiralTraceSample, write_spiral_trace_svg
from task.peg_in_hole.state_machine import PegInHoleContext, PegInHoleState, PegInHoleStateMachine
from vision.mujoco_keypoint_vision import (
    estimate_mujoco_oracle,
    estimate_mujoco_unet,
    heatmap_argmax,
    pixel_depth_to_world,
    project_world_to_pixel,
    render_camera_rgbd,
    site_world_pos,
    write_keypoint_overlay_svg,
)


class ArchimedesSpiralSearchTest(unittest.TestCase):
    def test_radius_is_monotonic_and_capped(self):
        search = ArchimedesSpiralSearch(radius_max=0.03, pitch=0.003, angular_speed=2.0)
        radii = [search.radius(t) for t in np.linspace(0.0, 80.0, 200)]

        self.assertGreaterEqual(min(radii), 0.0)
        self.assertLessEqual(max(radii), 0.03)
        self.assertTrue(all(a <= b + 1e-12 for a, b in zip(radii, radii[1:])))

    def test_radius_increases_by_pitch_per_turn(self):
        search = ArchimedesSpiralSearch(radius_max=0.03, pitch=0.003, angular_speed=2.0)
        one_turn_time = 2.0 * np.pi / search.angular_speed

        self.assertAlmostEqual(search.radius(one_turn_time), 0.003, places=8)
        self.assertAlmostEqual(search.radius(2.0 * one_turn_time), 0.006, places=8)

    def test_offset_is_in_xy_plane(self):
        search = ArchimedesSpiralSearch(radius_max=0.03, pitch=0.003, angular_speed=2.0)
        offset = search.sample_offset(7.5)

        self.assertEqual(offset.shape, (3,))
        self.assertAlmostEqual(float(offset[2]), 0.0, places=12)


class VisionErrorTest(unittest.TestCase):
    def test_random_xy_offset_is_inside_disk(self):
        rng = np.random.default_rng(42)
        radius = 0.02
        offsets = [sample_random_xy_offset(radius, rng=rng) for _ in range(100)]

        for offset in offsets:
            self.assertEqual(offset.shape, (2,))
            self.assertLessEqual(float(np.linalg.norm(offset)), radius + 1e-12)

    def test_random_xy_offset_is_seed_reproducible(self):
        a = sample_random_xy_offset(0.02, rng=np.random.default_rng(7))
        b = sample_random_xy_offset(0.02, rng=np.random.default_rng(7))

        np.testing.assert_allclose(a, b)


class MuJoCoVisionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = mujoco.MjModel.from_xml_path(str(PROJECT_ROOT / "models" / "peg_in_hole_scene.xml"))
        cls.data = mujoco.MjData(cls.model)
        mujoco.mj_forward(cls.model, cls.data)

    def test_heatmap_argmax_returns_xy_pixels(self):
        heatmaps = np.zeros((2, 8, 9), dtype=float)
        heatmaps[0, 3, 4] = 1.0
        heatmaps[1, 6, 2] = 2.0

        points = heatmap_argmax(heatmaps)

        np.testing.assert_allclose(points, np.array([[4.0, 3.0], [2.0, 6.0]]))

    def test_camera_project_unproject_roundtrip(self):
        hole = site_world_pos(self.model, self.data, "hole_entry_site")
        pixel, depth = project_world_to_pixel(
            self.model,
            self.data,
            "vision_top",
            hole,
            width=224,
            height=224,
        )

        recovered = pixel_depth_to_world(
            self.model,
            self.data,
            "vision_top",
            pixel,
            depth,
            width=224,
            height=224,
        )

        np.testing.assert_allclose(recovered, hole, atol=1e-9)

    def test_mujoco_unet_missing_weights_errors_clearly(self):
        hole = site_world_pos(self.model, self.data, "hole_entry_site")
        peg = site_world_pos(self.model, self.data, "peg_tip_site")

        with tempfile.TemporaryDirectory() as tmp:
            missing = Path(tmp) / "missing_model.pth"
            with self.assertRaisesRegex(FileNotFoundError, "requires --vision-model"):
                estimate_mujoco_unet(
                    model=self.model,
                    data=self.data,
                    camera_name="vision_top",
                    true_hole_world=hole,
                    true_peg_world=peg,
                    width=224,
                    height=224,
                    model_path=missing,
                    device="cpu",
                )

    def test_mujoco_oracle_pixel_noise_generates_xy_offset(self):
        hole = site_world_pos(self.model, self.data, "hole_entry_site")
        peg = site_world_pos(self.model, self.data, "peg_tip_site")
        estimate = estimate_mujoco_oracle(
            model=self.model,
            data=self.data,
            camera_name="vision_top",
            true_hole_world=hole,
            true_peg_world=peg,
            width=224,
            height=224,
            pixel_noise_std=2.0,
            rng=np.random.default_rng(7),
        )

        self.assertEqual(estimate.initial_offset_xy.shape, (2,))
        self.assertGreater(float(np.linalg.norm(estimate.initial_offset_xy)), 0.0)
        self.assertLess(float(np.linalg.norm(estimate.initial_offset_xy)), 0.02)

    def test_keypoint_overlay_svg_is_written(self):
        hole = site_world_pos(self.model, self.data, "hole_entry_site")
        peg = site_world_pos(self.model, self.data, "peg_tip_site")
        estimate = estimate_mujoco_oracle(
            model=self.model,
            data=self.data,
            camera_name="vision_top",
            true_hole_world=hole,
            true_peg_world=peg,
            width=224,
            height=224,
            pixel_noise_std=0.0,
            rng=np.random.default_rng(9),
        )
        rgb, _ = render_camera_rgbd(self.model, self.data, "vision_top", 224, 224)

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "vision.svg"
            write_keypoint_overlay_svg(path, rgb, estimate)
            text = path.read_text(encoding="utf-8")

            self.assertIn("MuJoCo Vision Keypoints", text)
            self.assertIn("hole est", text)
            self.assertIn("data:image/png;base64", text)


class ScrewMotionTest(unittest.TestCase):
    def test_depth_and_wiggle_are_bounded(self):
        motion = ScrewMotion(push_depth=0.03, wiggle_amplitude=0.002, screw_amplitude=0.2, duration=5.0, screw_turns=3.0)

        start = motion.sample_command(0.0)
        mid = motion.sample_command(2.5)
        end = motion.sample_command(5.0)

        self.assertAlmostEqual(start.insertion_depth, 0.0)
        self.assertAlmostEqual(mid.insertion_depth, 0.015)
        self.assertAlmostEqual(end.insertion_depth, 0.03)
        self.assertLessEqual(float(np.linalg.norm(start.wiggle_offset)), 0.002 + 1e-12)
        self.assertLessEqual(float(np.linalg.norm(mid.wiggle_offset)), 0.002 + 1e-12)
        self.assertLessEqual(float(np.linalg.norm(end.wiggle_offset)), 0.002 + 1e-12)
        self.assertAlmostEqual(end.screw_angle, 6.0 * np.pi)


class PegInHoleStateMachineTest(unittest.TestCase):
    def test_normal_path_reaches_complete(self):
        machine = PegInHoleStateMachine()
        ctx = PegInHoleContext(alignment_tolerance=0.004, insert_depth_target=0.025)

        self.assertEqual(machine.step(ctx), PegInHoleState.INITIAL_GRASP)

        ctx.approach_done = True
        self.assertEqual(machine.step(ctx), PegInHoleState.GRASP_PEG)

        ctx.grasp_done = True
        self.assertEqual(machine.step(ctx), PegInHoleState.MOVE_TO_PRE_INSERT)

        ctx.pre_insert_done = True
        self.assertEqual(machine.step(ctx), PegInHoleState.REACHING_PUSH)

        ctx.contact_detected = True
        self.assertEqual(machine.step(ctx), PegInHoleState.SEARCHING_SPIRAL)

        ctx.alignment_error_xy = 0.001
        self.assertEqual(machine.step(ctx), PegInHoleState.INSERTING_WIGGLE_SCREW)

        ctx.insertion_depth = 0.026
        self.assertEqual(machine.step(ctx), PegInHoleState.COMPLETE)

    def test_search_timeout_fails(self):
        machine = PegInHoleStateMachine()
        ctx = PegInHoleContext(
            approach_done=True,
            grasp_done=True,
            pre_insert_done=True,
            contact_detected=True,
            alignment_error_xy=0.02,
            search_elapsed=31.0,
            search_timeout=30.0,
        )

        machine.step(ctx)
        machine.step(ctx)
        machine.step(ctx)
        machine.step(ctx)
        self.assertEqual(machine.step(ctx), PegInHoleState.SEARCHING_SPIRAL)
        self.assertEqual(machine.step(ctx), PegInHoleState.FAILED)
        self.assertEqual(machine.failure_reason, "search_timeout")


class PaperMappingTest(unittest.TestCase):
    def test_demo_phases_map_to_paper_steps(self):
        reaching = classify_demo_phase("reaching_pushing_velocity_contact")
        searching = classify_demo_phase("searching_rubbing_spiral")
        inserting = classify_demo_phase("inserting_wiggle_screw")
        initial = classify_demo_phase("initial_grasp_pose")

        self.assertEqual(initial.procedure_step, "setup")
        self.assertEqual(reaching.procedure_step, "reaching")
        self.assertEqual(searching.procedure_step, "searching")
        self.assertEqual(inserting.procedure_step, "inserting")
        self.assertIn("rubbing", searching.unit_motion)
        self.assertIn("screwing", inserting.unit_motion)


class SpiralTraceSvgTest(unittest.TestCase):
    def test_spiral_trace_svg_is_written(self):
        samples = [
            SpiralTraceSample(
                time=float(i),
                command_xy=np.array([0.001 * i, 0.0005 * i], dtype=float),
                actual_xy=np.array([0.001 * i, 0.0004 * i], dtype=float),
                spiral_offset_xy=np.array([0.001 * i, 0.0005 * i], dtype=float),
                radius=0.001 * i,
            )
            for i in range(4)
        ]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "trace.svg"
            write_spiral_trace_svg(
                path,
                samples=samples,
                initial_offset_xy=np.array([0.01, -0.005], dtype=float),
                search_radius=0.03,
                alignment_tolerance=0.004,
            )

            text = path.read_text(encoding="utf-8")
            self.assertIn("Archimedes Spiral Rubbing Trace", text)
            self.assertIn("<polyline", text)


if __name__ == "__main__":
    unittest.main()
