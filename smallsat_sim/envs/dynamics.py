"""
This file serves as a tool to generate a symbolical, mathematical model of 
actuated objects in space.
CasADi is used to achieve this task, which is a symbolic framework. 
"""

import numpy as np
import casadi as ca

from casadi import SX, DM


def skew(x: np.ndarray) -> np.ndarray:
    """
    Computes the skew matrix of a vector x.
    """
    return np.array([[0, -x[2], x[1]], [x[2], 0, -x[0]], [-x[1], x[0], 0]])


class SymbolicModel:
    """
    Class that contains all necessary components of a symbolic model.
    """

    def __init__(self, cfg: dict) -> None:
        """
        Initialize the symbolic model using the given configuration.

        Args:
            cfg (dict): Configuration containing physical properties and thruster info.
        """
        # Load physical properties
        props = cfg.pp
        self.mass = props.mass
        self.inertia = props.diag_inertia
        self.com_offset = props.com_offset

        # Load thruster information
        self.thrusters = cfg.Thrusters
        self.nu = self.thrusters.n_thrusters

        # Calculate mixer matrix and set up the CasADi model
        self._calc_mixer()
        self._setup_model()

    def _calc_mixer(self) -> None:
        """
        Calculates the mixer matrix which converts thruster inputs to forces.

        Needed information:
        - force_gears: Array of shape (nu, 3) representing the forces produced by each thruster.
        - actuator_pos: Array of shape (nu, 3) representing the position of each thruster relative to body origin.

        Returns:
        - mixer: Array of shape (6, nu), mapping thruster inputs to force and torques.
        """
        # Initialize mixer matrix
        self.mixer = np.zeros((6, self.nu))
        for idx, thruster in enumerate(self.thrusters.thruster_list):
            self.mixer[:, idx] = np.concatenate(
                (
                    thruster.gear,
                    np.cross(thruster.pos, thruster.gear),
                )
            )

        # self.mixer[:, 0] = 0
        # self.mixer[:, 2] = 0

    def _setup_model(self) -> None:
        """
        Sets up the symbolic model for integration and generation of solvers in CasADi
        """
        # Create state and input symbols
        r = ca.vertcat(*[SX.sym(name) for name in ["rx", "ry", "rz"]])
        q = ca.vertcat(*[SX.sym(name) for name in ["eta", "eps1", "eps2", "eps3"]])
        v = ca.vertcat(
            *[SX.sym(name) for name in ["vx", "vy", "vz"]]
        )  # Velocity in inertial frame
        omega = ca.vertcat(
            *[SX.sym(name) for name in ["omega_x", "omega_y", "omega_z"]]
        )
        x = ca.vertcat(r, q, v, omega)
        u = SX.sym("u", self.nu)  # Thruster inputs

        # Create derivative symbols for each state
        r_dot = ca.vertcat(
            *[SX.sym(f"{name}_dot") for name in ["rx_dot", "ry_dot", "rz_dot"]]
        )
        q_dot = ca.vertcat(
            *[
                SX.sym(f"{name}_dot")
                for name in ["eta_dot", "eps1_dot", "eps2_dot", "eps3_dot"]
            ]
        )
        v_dot = ca.vertcat(
            *[SX.sym(f"{name}_dot") for name in ["vx_dot", "vy_dot", "vz_dot"]]
        )
        omega_dot = ca.vertcat(
            *[
                SX.sym(f"{name}_dot")
                for name in ["omega_x_dot", "omega_y_dot", "omega_z_dot"]
            ]
        )
        x_dot = ca.vertcat(r_dot, q_dot, v_dot, omega_dot)

        # Initialize CasADi parameter and algebraic symbols
        z = ca.vertcat([])  # Empty since not used
        p = ca.vertcat([])  # Empty by default, can be adjusted later on

        # Mass matrix setup
        m = self.mass
        I = np.diag(self.inertia)
        M_com = np.eye(6, 6)  # Full inertia matrix (6x6)
        M_com[0:3, 0:3] = m * np.eye(3)
        M_com[3:6, 3:6] = I

        # System transformation matrix from CG to CO
        # CG = Center of Gravity, CO = Center origin (body frame)
        H = np.eye(6, 6)
        H[0:3, 3:6] = np.transpose(skew(np.array(self.com_offset)))

        # Transform system matrices to body frame by similarity transformation
        M_body = H.T @ M_com @ H
        M_body_inv = ca.DM(np.linalg.inv(M_body))

        # Rotation matrix from quaternion
        # Notation: R_{IB} (from body to inertial frame)
        R_quat = ca.vertcat(
            ca.horzcat(
                1 - 2 * (q[2] ** 2 + q[3] ** 2),
                2 * (q[1] * q[2] - q[0] * q[3]),
                2 * (q[1] * q[3] + q[0] * q[2]),
            ),
            ca.horzcat(
                2 * (q[1] * q[2] + q[0] * q[3]),
                1 - 2 * (q[1] ** 2 + q[3] ** 2),
                2 * (q[2] * q[3] - q[0] * q[1]),
            ),
            ca.horzcat(
                2 * (q[1] * q[3] - q[0] * q[2]),
                2 * (q[2] * q[3] + q[0] * q[1]),
                1 - 2 * (q[1] ** 2 + q[2] ** 2),
            ),
        )

        # Quaternion kinematics transformation (for Hamiltonian product)
        T_quat = 0.5 * ca.vertcat(
            ca.horzcat(-q[1], -q[2], -q[3]),
            ca.horzcat(q[0], -q[3], q[2]),
            ca.horzcat(q[3], q[0], -q[1]),
            ca.horzcat(-q[2], q[1], q[0]),
        )

        # Centrifugal effect in body frame (due to CoM offset)
        c = ca.vertcat(
            m * ca.mtimes([ca.skew(omega), ca.skew(omega), self.com_offset]),
            ca.mtimes(
                [
                    ca.skew(omega),
                    (
                        I
                        - m
                        * ca.mtimes(ca.skew(self.com_offset), ca.skew(self.com_offset))
                    ),
                    omega,
                ]
            ),
        )

        # Calculate the applied wrench in body frame
        # wrench = [Fx, Fy, Fz, Tx, Ty, Tz]
        wrench = ca.mtimes(M_body_inv, -c + ca.mtimes(DM(self.mixer), u))

        # State space equations
        # NOTE: Velocity in BODY frame
        f_expl = ca.vertcat(
            ca.mtimes(R_quat, v),  # r_dot = v (since v is in the inertial frame)
            ca.mtimes(T_quat, omega),  # q_dot
            wrench[0:3] - ca.mtimes(ca.skew(omega), v),
            wrench[3:6],
        )

        # Inertial frame velocity
        # NOTE: Force vector rotated to inertial frame, Torque still in body frame
        # f_expl = ca.vertcat(
        #     v,  # r_dot = v (since v is in the inertial frame)
        #     ca.mtimes(T_quat, omega),  # q_dot
        #     ca.mtimes(R_quat, wrench[0:3]),
        #     wrench[3:6],
        # )

        # Save everything to symbolic model object (for MPC generation)
        self.x = x
        self.xdot = x_dot
        self.u = u
        self.z = z
        self.p = p
        self.f_expl_expr = f_expl
        self.f_impl_expr = x_dot - f_expl

        # Create some utils
        self.f_expl_expr_func = ca.Function("f_expl_expr_func", [x, u], [f_expl])

    def integrate(self, x, u) -> np.ndarray:
        """
        Propagates the system dynamics for a given state and input
        """
        return self.f_int(x, u).toarray()

    def get_integrator(self, dt: float):
        """
        Method which creates an integrator if needed by a control algorithm.
        """
        # Euler forward integration
        x_next = self.x + dt * self.f_expl_expr_func(self.x, self.u)

        self.f_int = ca.Function("f_int", [self.x, self.u], [x_next])

        # TODO: Implement Euler semi-implicit integration?

        return self.integrate
