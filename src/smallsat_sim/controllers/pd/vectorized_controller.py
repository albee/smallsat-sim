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
from smallsat_sim.planners.base_planner import BasePlanner
from smallsat_sim.envs.base_env import BaseEnv
from smallsat_sim.utils.helpers_jax import quat_multiply, quat_conjugate, Rquat, sgn_quat

import jax
import jax.numpy as jnp


class VectorizedPDController(BaseController):
    def __init__(self, env: BaseEnv, planner: BasePlanner) -> None:
        # Fetch correct controller config
        ctrl_cfg = env.env_cfg.control.PD

        # Initialize base class
        super().__init__(env, planner, ctrl_cfg)

        # Set some reference quantities (Deprecated?)
        self.v_ref = jnp.zeros((3, 1))
        self.omega_ref = jnp.zeros((3, 1))  # Angular velocity

        # Extract PD gains
        self.Kp_x = ctrl_cfg.gains.Kp_x
        self.Kd_x = ctrl_cfg.gains.Kd_x
        self.Kp_q = ctrl_cfg.gains.Kp_q
        self.Kd_q = ctrl_cfg.gains.Kd_q

        self.B_matrix = self.calc_B_matrix(env.model)

        # jax.jit and jax.vmap the relevant functions
        self.quat_conjugate_vec = jax.jit(jax.vmap(quat_conjugate))
        self.quat_multiply_vec = jax.jit(jax.vmap(quat_multiply, in_axes=0))
        self.Rquat_vec = jax.jit(jax.vmap(Rquat))
        self._compute_desired_linacc_vec = jax.jit(
            jax.vmap(self._compute_desired_linacc)
        )
        self._compute_desired_alpha_vec = jax.jit(jax.vmap(self._compute_desired_alpha))
        self._compute_u_unconstrained_vec = jax.jit(
            jax.vmap(self._compute_u_unconstrained)
        )

    def calc_B_matrix(self, model) -> jnp.ndarray:
        """Create B matrix (thruster configuration matrix, mixer, etc.)"""
        B_matrix = jnp.zeros(shape=(6, model.nu))
        for i in range(model.nu):
            actuator = model.actuator(i)
            force = model.actuator(i).gear[0:3]  # Force produced by thruster 'i'
            actuator_pos = model.site(
                i
            ).pos  # Position of thruster 'i' relative to body origin
            # Add direct forces and torques to B matrix. Then calculate force arms and add
            B_matrix = B_matrix.at[:, i].set(
                actuator.gear
                + jnp.concatenate([jnp.zeros(3), jnp.cross(actuator_pos, force)])
            )

        return B_matrix

    def _apply_ctrl_constraint(self, env: BaseEnv, u: jnp.ndarray) -> jnp.ndarray:
        """Apply non-negative control constraints to the input control signal u."""
        # Get index of all thrusters that give propulsion in x,y,z
        # This assumes that thrusters only have propulsion in one direction!
        x_thrusters_id = [
            i for i, gear in enumerate(env.model.actuator_gear) if gear[0] != 0
        ]
        y_thrusters_id = [
            i for i, gear in enumerate(env.model.actuator_gear) if gear[1] != 0
        ]

        combined_lists = jnp.stack(
            [jnp.array(x_thrusters_id), jnp.array(y_thrusters_id)]
        )
        min_elems = jnp.amin(u[:, combined_lists], axis=-1)
        negative_min_mask = min_elems < 0
        u_x = u.at[:, jnp.array(x_thrusters_id)].set(
            u[:, jnp.array(x_thrusters_id)]
            - jnp.where(negative_min_mask[:, 0, None], min_elems[:, 0, None], 0)
        )
        u_y = u.at[:, jnp.array(y_thrusters_id)].set(
            u[:, jnp.array(y_thrusters_id)]
            - jnp.where(negative_min_mask[:, 1, None], min_elems[:, 1, None], 0)
        )
        u = jnp.concatenate([u_x[:, 0:4], u_y[:, 4:8]], axis=1)

        return u

    def get_control_input(self, env: BaseEnv) -> jnp.ndarray:
        """Defines the controller callback for the simulation step."""
        # _desired_pos, _desired_quat = self.planner.get_reference(env.obs)
        _desired_pos, _desired_quat = jnp.array([0.0, 0.0, 10.17]).reshape(3, 1), jnp.array([1, 0, 0, 0]).reshape(4, 1)
        desired_pos = jnp.asarray(_desired_pos)
        desired_quat = jnp.asarray(_desired_quat)
        desired_linvel = self.v_ref  # Linear
        desired_angvel = jnp.zeros((3, 1))
        current_pos = env.obs[:, :3]
        current_quat = env.obs[:, 3:7]
        current_linvel = env.obs[:, 7:10]  # Linear
        current_angvel = env.obs[:, 10:13]  # Angular

        x_error = (
            jnp.full((env.num_envs, 3), desired_pos.reshape(1, -1)) - current_pos
        )  # Linear, world frame
        v_error = (
            jnp.full((env.num_envs, 3), desired_linvel.reshape(1, -1)) - current_linvel
        )  # Linear, world frame

        # Desired quaternion - current quaternion:
        current_quat_conj = self.quat_conjugate_vec(current_quat)
        quat_error = self.quat_multiply_vec(
            jnp.full((env.num_envs, 4), desired_quat.reshape(1, -1)), current_quat_conj
        )
        eta_error = quat_error[:, 0].reshape(-1, 1)
        eps_error = quat_error[:, 1:]

        # PD, linear part. Note: both position and velocity are decomposed
        # in the world frame. R.T rotates them to the body frame
        R_WB = self.Rquat_vec(current_quat)  # Rotation matrix from body to world
        desired_linacc = self._compute_desired_linacc_vec(R_WB, x_error, v_error)
        # PD, angular part, (alpha=angular acceleration).
        # Note: angular velocity is decomposed in the body frame
        desired_alpha = self._compute_desired_alpha_vec(
            eta_error,
            eps_error,
            jnp.full((env.num_envs, 3), desired_angvel.reshape(1, -1)),
            current_angvel,
        )
        desired_acceleration = jnp.concatenate([desired_linacc, desired_alpha], axis=1)

        desired_control = desired_acceleration
        # Distribute the desired forces and torques to the actuators, least squares
        p_inv_B_matrix = jnp.asarray(jnp.linalg.pinv(self.B_matrix))
        u_unconstrained = self._compute_u_unconstrained_vec(
            jnp.full((env.num_envs, 8, 6), p_inv_B_matrix), desired_control
        )

        # Only allow non-negative thrust values
        u = self._apply_ctrl_constraint(env, u_unconstrained)
        return u

    def _compute_desired_linacc(
        self, R_WB: jnp.ndarray, x_error: jnp.ndarray, v_error: jnp.ndarray
    ) -> jnp.ndarray:
        """
        Compute the desired linear acceleration.
        """
        return self.Kp_x * R_WB.T @ x_error + self.Kd_x * R_WB.T @ v_error

    def _compute_desired_alpha(
        self,
        eta_error: jnp.ndarray,
        eps_error: jnp.ndarray,
        desired_angvel: jnp.ndarray,
        current_angvel: jnp.ndarray,
    ) -> jnp.ndarray:
        """
        Compute desired alpha.
        """
        sgn_quat_eta_error = jnp.where(eta_error > 0, 1, -1)
        return self.Kp_q * sgn_quat_eta_error * eps_error + self.Kd_q * (
            desired_angvel - current_angvel
        )

    def _compute_u_unconstrained(
        self,
        p_inv_B_matrix: jnp.ndarray,
        desired_control: jnp.ndarray,
    ) -> jnp.ndarray:
        """
        Compute the unconstrained control inputs.
        """
        return jnp.dot(p_inv_B_matrix, desired_control)

    def _log(self, run_id: int, timestamp: float, env: BaseEnv) -> None:
        """
        Logs desired quantities if flag is enabled
        """
        raise NotImplementedError(
            f"The _log method is not implemented for the class {self.__class__.__name__}"
        )
