"""
This file serves as a tool to generate a symbolical, mathematical model of 
actuated objects in space.
CasADi is used to achieve this task, which is a symbolic framework. 
"""

import numpy as np
import casadi as ca

from casadi import SX, DM


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

    def _setup_model(self) -> None:
        """
        Sets up the symbolic model for integration and generation of solvers in CasADi
        """
        # Create state and input symbols
        r = ca.vertcat(*[SX.sym(name) for name in ["rx", "ry", "rz"]])
        q = ca.vertcat(*[SX.sym(name) for name in ["eta", "eps1", "eps2", "eps3"]])
        v = ca.vertcat(*[SX.sym(name) for name in ["vx", "vy", "vz"]])
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
        I = SX(3, 3)
        I[0:3, 0:3] = np.diag(self.inertia)
        M_com = SX(6, 6)  # Full inertia matrix (6x6)
        M_com[0:3, 0:3] = m * SX.eye(3)
        M_com[3:6, 3:6] = I

        # System transformation matrix from CG to CO
        # CG = Center of Gravity, CO = Center origin (body frame)
        H = SX(6, 6)
        H[0:3, 0:3] = SX.eye(3)
        H[0:3, 3:6] = ca.transpose(ca.skew(np.array(self.com_offset)))
        H[3:6, 3:6] = SX.eye(3)

        # Transform system matrices to body frame by similarity transformation
        M_body = ca.mtimes(ca.mtimes(ca.transpose(H), M_com), H)
        M_body_inv = ca.solve(M_body, SX.eye(M_body.size1()))

        # Rotation matrix from quaternion
        eta, eps = q[0], q[1:4]
        S = ca.skew(eps)
        R_quat = SX.eye(3) + 2 * eta * S + 2 * ca.mtimes(S, S)

        # Quaternion kinematics transformation
        T_quat = 0.5 * np.array(
            [
                [-eps[0], -eps[1], -eps[2]],
                [eta, -eps[2], eps[1]],
                [eps[2], eta, -eps[0]],
                [-eps[1], eps[0], eta],
            ]
        )

        # System Jacobian
        J_quat = SX(7, 6)
        J_quat[0:3, 0:3] = R_quat
        J_quat[3:7, 3:6] = T_quat

        # Coriolis effect in body frame
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

        # State space equations
        f_expl = ca.vertcat(
            ca.mtimes(J_quat, ca.vertcat(v, omega)),
            ca.mtimes(
                M_body_inv,
                -c + ca.mtimes(DM(self.mixer), u),
            ),
        )

        # Save everything to symbolic model object (for MPC generation)
        self.x = x
        self.xdot = x_dot
        self.u = u
        self.z = z
        self.p = p
        self.f_expl_expr = f_expl
        self.f_impl_expr = x_dot - f_expl

        # Create some utils
        self.f_expl_expr_func = ca.Function('f_expl_expr_func', [x, u], [f_expl])

    def integrate(self, x, u) -> np.ndarray:
        """
        Propagates the system dynamics for a given state and input
        """
        return self.f_int(x,[], u, [], [], [], [])[0].toarray()
    
    def get_integrator(self, dt: float):
        """
        Method which creates an integrator if needed by a control algorithm.
        """
        self.f_int = ca.integrator(
            "f_int", "rk", {"x": self.x, "p": self.u, "ode": self.f_expl_expr}, 0, dt, {"number_of_finite_elements": 1}
        )

        return self.integrate