# (License for _apply_ctrl_constraint function)
# Copyright (c) 2017, United States Government, as represented by the
# Administrator of the National Aeronautics and Space Administration.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations
# under the License.

from smallsat_sim.controllers.base_controller import BaseController
from smallsat_sim.envs.base_env import BaseEnv
from smallsat_sim.utils.helpers import quat_multiply, quat_conjugate, Rquat, sgn_quat

import numpy as np


class PDController(BaseController):
    def __init__(self, env, planner) -> None:
        # Fetch correct controller config
        ctrl_cfg = env.env_cfg.control.PD

        # Initialize base class
        super().__init__(env, planner, ctrl_cfg)

        # Set some reference quantities (Deprecated?)
        self.x_ref = self.planner.get_reference(env.obs)
        self.quat_ref = np.array([1.0, 0.0, 0.0, 0.0])
        self.v_ref = np.zeros((3, 1))
        self.omega_ref = np.zeros((3, 1))  # Angular velocity

        # Extract PD gains
        self.Kp_x = ctrl_cfg.gains.Kp_x
        self.Kd_x = ctrl_cfg.gains.Kd_x
        self.Kp_q = ctrl_cfg.gains.Kp_q
        self.Kd_q = ctrl_cfg.gains.Kd_q

        self.B_matrix = self.calc_B_matrix(env.model)

    def calc_B_matrix(self, model) -> np.ndarray:
        """Create B matrix (thruster configuration matrix, mixer, etc.)"""
        B_matrix = np.empty(shape=(6, model.nu))
        for i in range(model.nu):
            actuator = model.actuator(i)
            force = model.actuator(i).gear[0:3]  # Force produced by thruster 'i'
            actuator_pos = model.site(
                i
            ).pos  # Position of thruster 'i' relative to body origin
            # Add direct forces and torques to B matrix. Then calculate force arms and add
            B_matrix[:, i] = actuator.gear + np.append(
                [0, 0, 0], np.cross(actuator_pos, force)
            )
        return B_matrix

    def _apply_ctrl_constraint(self, env: BaseEnv, u: np.ndarray) -> np.ndarray:
        """Apply non-negative control constraints to the input control signal u."""
        # Get index of all thrusters that give propulsion in x,y,z
        # This assumes that thrusters only have propulsion in one direction!
        x_thrusters_id = [
            i for i, gear in enumerate(env.model.actuator_gear) if gear[0] != 0
        ]
        y_thrusters_id = [
            i for i, gear in enumerate(env.model.actuator_gear) if gear[1] != 0
        ]
        z_thrusters_id = [
            i for i, gear in enumerate(env.model.actuator_gear) if gear[2] != 0
        ]

        for list in [x_thrusters_id, y_thrusters_id, z_thrusters_id]:
            min_thrust = np.amin(u[list])  # Find lowest thrust value in the list
            if min_thrust < 0:
                u[list] -= min_thrust  # Subtract the minimum thrust from all thrusters
        return u

    def get_control_input(self, env: BaseEnv) -> np.ndarray:
        """Defines the controller callback for the simulation step."""
        desired_pos = self.planner.get_reference(env.obs).reshape(3, 1)  # Linear
        desired_quat = self.quat_ref
        desired_linvel = self.v_ref  # Linear
        desired_angvel = np.zeros((3, 1))
        current_pos = np.reshape(env.obs[:3], (3, 1))
        current_quat = np.reshape(env.obs[3:7], (4, 1))
        current_linvel = np.reshape(env.obs[7:10], (3, 1))  # Linear
        current_angvel = np.reshape(env.obs[10:13], (3, 1))  # Angular

        x_error = desired_pos - current_pos  # Linear, world frame
        v_error = desired_linvel - current_linvel  # Linear, world frame
        # Desired quaternion - current quaternion:
        quat_error = quat_multiply(desired_quat, quat_conjugate(current_quat))
        eta_error = quat_error[0]
        eps_error = quat_error[1:]

        # PD, linear part. Note: both position and velocity are decomposed
        # in the world frame. R.T rotates them to the body frame
        R_WB = Rquat(current_quat)  # Rotation matrix from body to world
        desired_linacc = (
            self.Kp_x * R_WB.T @ x_error  # Desired linear acceleration
            + self.Kd_x * R_WB.T @ v_error
        )
        # PD, angular part, (alpha=angular acceleration).
        # Note: angular velocity is decomposed in the body frame
        desired_alpha = (
            self.Kp_q * sgn_quat(eta_error) * eps_error
            + self.Kd_q * (desired_angvel - current_angvel)
        )
        desired_acceleration = np.append(desired_linacc, desired_alpha)

        desired_control = desired_acceleration
        # Distribute the desired forces and torques to the actuators, least squares
        u_unconstrained = np.dot(np.linalg.pinv(self.B_matrix), desired_control)

        # Only allow non-negative thrust values
        u = self._apply_ctrl_constraint(env, u_unconstrained)
        return u
