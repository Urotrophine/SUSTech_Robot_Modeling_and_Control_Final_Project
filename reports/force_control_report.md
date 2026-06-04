# Force Control Evaluation

## Quantitative Results

| Metric | Value |
| --- | ---: |
| duration_s | 10.022 |
| max_contact_force_n | 72.116 |
| mean_contact_force_n | 17.7312 |
| final_inserted_depth_m | 0.0614764 |
| insertion_phase_depth_change_m | 0.0612501 |
| max_abs_joint3_rad | 0.00772366 |
| max_abs_joint4_rad | 0.0403601 |
| final_joint6_turns | 1.9142 |
| max_tracking_error | 1.05183 |
| max_tracking_error_without_q6 | 0.0484712 |
| max_q34_tracking_error | 0.0383317 |
| final_xy_error_m | 0.00395846 |

## Charts

![Contact force and insertion depth](contact_force_and_depth.png)

![Joint stability](joint_stability.png)

![Tracking error](tracking_error.png)

![Peg center XY path](peg_xy_path.png)

## Interpretation

The controller is torque/force based: MuJoCo `<motor>` actuators receive generalized force commands. The impedance law computes `tau = qfrc_bias + Kp(q_des-q) + Kd(qd_des-qd)`, then clips it by actuator limits.

The full tracking-error spike near insertion is caused by the deliberate multi-turn `joint6` screwing command. The chart therefore also shows tracking error excluding `joint6`, which better reflects translational insertion stability.

Compared with the earlier position-control demo, force control allows contact to create tracking error and measurable contact force. This is less rigid than position control, but closer to deployable behavior because force/torque limits and damping shape the contact response.
