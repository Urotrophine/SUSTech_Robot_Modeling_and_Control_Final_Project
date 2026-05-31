# Robotic Arm Peg-in-Hole Spiral Search Demo

This repository contains a MuJoCo simulation for a v7 6-DOF robotic arm that grasps a cylindrical peg, moves it above a fixed block with a circular hole, performs an Archimedean spiral search on the top surface, and inserts the peg once the projected peg center enters the hole projection.

The project currently stops at the spiral-search stage. The later paper stage involving natural tilted three-point contact and wiggling was evaluated, but it is not kept as a supported result because the current MuJoCo model uses position-controlled grasping and simplified contact geometry. That stage would need a more physically faithful force/impedance-control model.

## Quick Start

Use the local Conda environment that already has MuJoCo installed:

```powershell
& "D:\miniconda\envs\rmc\python.exe" scripts\run_v6_peg_in_hole_spiral_search_demo.py
```

Headless verification:

```powershell
& "D:\miniconda\envs\rmc\python.exe" scripts\run_v6_peg_in_hole_spiral_search_demo.py --headless --seed 1
```

The `--seed` option fixes the randomized cylinder spawn position for repeatable tests.

## Current Supported Demo

### `scripts/run_v6_peg_in_hole_spiral_search_demo.py`

This is the main simulation entry point.

It performs:

1. Random cylinder generation inside the reachable workspace.
2. Maximum gripper opening.
3. Planar alignment to the cylinder center using only `joint1` and `joint2`.
4. Vertical descent using only prismatic `joint5`.
5. Gripper closing and peg lifting.
6. Intentional lateral offset near the hole to simulate alignment error.
7. Constant-height contact/search posture near the hole top surface.
8. Archimedean spiral search using only `joint1` and `joint2`.
9. Final insertion using only `joint5` after the peg projection is inside the hole projection.

The script generates a runtime MJCF scene under `models/_*.xml`. These files are ignored by Git and can be deleted at any time.

## Project Structure

```text
assets/
  meshes/                 v7 STL mesh assets used by MuJoCo
  urdf/
    robotic_arm_v7.urdf   source URDF from SW2URDF after project-specific fixes
    package.xml           package metadata for URDF mesh paths

configs/
  robot_description.yaml  single registry for joint names, gripper names, sites, and limits

models/
  simple_grasp_scene.xml  v7 robot, gripper, cylinder, ground, sites, and actuators

scripts/
  run_v6_peg_in_hole_spiral_search_demo.py
                           main supported peg-in-hole spiral-search simulation

src/
  api/
    arm_platform_api.py   high-level API around MuJoCo model/data, controllers, gripper, FK
  control/
    joint_space_controller.py
                           position controller wrapper for arm and gripper joints
  gripper/
    parallel_gripper.py   symmetric gripper command mapping for joint71/joint72
  kinematics/
    robot_kinematics.py   MuJoCo-based FK/Jacobian utilities
  planning/
    joint_trajectory.py   simple joint trajectory interpolation helper

environment.yml           Conda environment description
requirements.txt          minimal Python dependencies
```

## Robot Model

The current robot is the v7 model:

- Arm joints: `joint1`, `joint2`, `joint3`, `joint4`, `joint5`, `joint6`
- `joint5` is prismatic and moves the end effector vertically.
- Gripper joints: `joint71`, `joint72`
- `joint71` range: `-0.05` to `0.015`
- `joint72` range: `-0.015` to `0.05`
- The gripper is controlled as a mirrored parallel gripper.

The mesh and URDF sources live in `assets/`. The MuJoCo runtime model used by the demo is `models/simple_grasp_scene.xml`.

## Control Logic

The demo intentionally restricts which joints move in each phase:

- Cylinder and hole planar alignment: only `joint1` and `joint2`.
- Grasp descent, lift, and insertion: only `joint5`.
- `joint3`, `joint4`, and `joint6` are fixed in the supported spiral-search demo.

This is implemented in the script through helper target constructors:

- `joint12_only_target(...)` copies only the solved `joint1/joint2` values into the current posture.
- `joint5_only_target(...)` changes only `joint5`.
- `solve_revolute12_xy(...)` uses MuJoCo site Jacobians to solve the planar finger midpoint target with `joint1/joint2`.
- `solve_joint5_for_finger_z(...)` and `solve_joint5_for_object_z(...)` compute vertical prismatic commands from target heights.

The gripper is controlled through `ArmPlatformAPI.set_gripper(...)`, which delegates to `ParallelGripper`. The current close command is conservative enough to keep the peg attached during the search demo.

## Spiral Search

The Archimedean spiral is generated in `archimedean_spiral_offsets()` inside the main script:

```text
r = pitch * theta / (2*pi)
x = r*cos(theta)
y = r*sin(theta)
```

The search center is the peg center at the start of the search phase. Each spiral point is converted into a target finger midpoint. The solver then computes `joint1/joint2` only, while `joint5` stays fixed to maintain a constant top-contact/search height.

The search succeeds when:

```text
norm(peg_center_xy - hole_center_xy) <= HOLE_RADIUS - CYL_RADIUS + tolerance
```

After success, `joint5` performs the insertion.

## Why Wiggling Is Not Supported

The paper describes a later stage where the peg naturally becomes tilted after entering the hole, forms a three-point contact state, and is corrected by random pitch/roll wiggling.

That stage was tested but removed from the supported project scope for now:

- The current demo uses position-controlled grasp following to keep the peg held by the gripper.
- MuJoCo contacts are soft numerical constraints, not perfectly rigid non-penetration constraints.
- The round hole is approximated by many collision boxes rather than a true analytic hole surface.
- Reliable natural tilted insertion would require contact-rich force or impedance control, improved collision geometry, and no pose-locking of the peg.

For this repository, the reliable and repeatable deliverable is therefore the grasp, alignment, spiral search, and insertion stage.

## Useful Commands

Run GUI:

```powershell
& "D:\miniconda\envs\rmc\python.exe" scripts\run_v6_peg_in_hole_spiral_search_demo.py
```

Run headless:

```powershell
& "D:\miniconda\envs\rmc\python.exe" scripts\run_v6_peg_in_hole_spiral_search_demo.py --headless
```

Run repeatable headless case:

```powershell
& "D:\miniconda\envs\rmc\python.exe" scripts\run_v6_peg_in_hole_spiral_search_demo.py --headless --seed 1
```

Syntax check:

```powershell
& "D:\miniconda\envs\rmc\python.exe" -m py_compile scripts\run_v6_peg_in_hole_spiral_search_demo.py
```

## GitHub Notes

Runtime XML files named `models/_*.xml`, Python caches, logs, backups, and raw SW2URDF export folders are ignored by `.gitignore`.

Before committing, a clean working tree should contain the v7 assets, the main model, the source modules, and the single supported spiral-search demo script.
