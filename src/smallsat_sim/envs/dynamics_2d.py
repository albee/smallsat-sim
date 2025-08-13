"""
This file serves as a tool to generate a symbolical, mathematical model of 
actuated objects in space.
CasADi is used to achieve this task, which is a symbolic framework. 
"""

import numpy as np
import casadi as ca

from casadi import SX, DM
from smallsat_sim.envs.dynamics import SymbolicModel

class SymbolicModel2D(SymbolicModel):
    """
    Class that contains all necessary components of a symbolic model.
    """
    def _setup_model(self) -> None:
        """
        Sets up the symbolic model for integration and generation of solvers in CasADi
        """
        # Create state and input symbols
        r = ca.vertcat(*[SX.sym(name) for name in ["rx", "ry"]])
        phi = ca.vertcat(*[SX.sym(name) for name in ["phi"]])
        v = ca.vertcat(
            *[SX.sym(name) for name in ["vx", "vy"]]
        )  # Velocity in inertial frame
        omega = ca.vertcat(
            *[SX.sym(name) for name in ["omega"]]
        )
        x = ca.vertcat(r, phi, v, omega)
        u = SX.sym("u", self.nu)  # Thruster inputs

        # Create derivative symbols for each state
        r_dot = ca.vertcat(
            *[SX.sym(f"{name}_dot") for name in ["rx_dot", "ry_dot"]]
        )
        phi_dot = ca.vertcat(
            *[
                SX.sym(f"{name}_dot")
                for name in ["phi_dot"]
            ]
        )
        v_dot = ca.vertcat(
            *[SX.sym(f"{name}_dot") for name in ["vx_dot", "vy_dot"]]
        )
        omega_dot = ca.vertcat(
            *[
                SX.sym(f"{name}_dot")
                for name in ["omega_dot"]
            ]
        )
        x_dot = ca.vertcat(r_dot, phi_dot, v_dot, omega_dot)

        # Initialize CasADi parameter and algebraic symbols
        z = ca.vertcat([])  # Empty since not used
        p = ca.vertcat([])  # Empty by default, can be adjusted later on

        # Mass matrix setup
        m = self.mass
        I = self.inertia[2]
        M_com = np.eye(3, 3)  # Full inertia matrix (6x6)
        M_com[0:2, 0:2] = m * np.eye(2)
        M_com[2, 2] = I

        # System transformation matrix from CG to CO
        # CG = Center of Gravity, CO = Center origin (body frame)
        H = np.eye(3, 3)

        # TODO, assume zero for now
        # H[0:3, 3:6] = np.transpose(skew(np.array(self.com_offset)))

        # Transform system matrices to body frame by similarity transformation
        M_body = H.T @ M_com @ H
        M_body_inv = ca.DM(np.linalg.inv(M_body))

        # Rotation matrix from yaw angle
        R_rot = ca.vertcat(
            ca.horzcat(ca.cos(phi), -ca.sin(phi)),
            ca.horzcat(ca.sin(phi), ca.cos(phi)),
        )

        # Forces due to thrusters
        F_th = ca.mtimes(DM(self.mixer), u)

        # Remove all forces that are not in 2d plane.
        M_Force_Selection = np.zeros((3,6))
        M_Force_Selection[0, 0] = 1
        M_Force_Selection[1, 1] = 1
        M_Force_Selection[2, 5] = 1
        wrench = ca.mtimes(M_body_inv, ca.mtimes(M_Force_Selection, F_th))

        # State space equations
        # NOTE: Velocity in BODY frame
        f_expl = ca.vertcat(
            ca.mtimes(R_rot, v),  # r_dot = v (since v is in the inertial frame)
            omega,  # q_dot
            wrench[0] + v[1]*omega, # vx_dot = Fx/m + vy*omega
            wrench[1] + v[0]*omega, # vy_dot = Fy/m + vx*omega
            wrench[2] - omega*omega, # omega_dot = T/I - omega^2
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
        self.f_expl_expr_func = ca.Function("f_expl_expr_func", [x, u], [f_expl])