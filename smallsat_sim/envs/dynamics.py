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
        # Load physical properties
        self.length = cfg["props"]["length"]
        self.width = cfg["props"]["width"]
        self.height = cfg["props"]["height"]
        self.density = cfg["props"]["density"]
        self.nu = cfg["actuators"]["nu"]

        # Calculate remaining physical properties
        self.volume = self.length * self.width * self.height
        self.mass = self.volume * self.density
        self.I11 = (1 / 12) * self.mass * (self.width**2 + self.height**2)
        self.I22 = (1 / 12) * self.mass * (self.length**2 + self.height**2)
        self.I33 = (1 / 12) * self.mass * (self.length**2 + self.width**2)

        # Load actuator information
        self.actuators = cfg["actuators"]

        # Calculate mixer matrix
        self._calc_mixer()

        # Setup the CasADi model
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

        for idx in range(self.nu):
            # Populate mixer
            self.mixer[:, idx] = np.concatenate(
                (
                    self.actuators[f"thruster{idx}"]["gear"],
                    np.cross(
                        self.actuators[f"thruster{idx}"]["gear"],
                        self.actuators[f"thruster{idx}"]["pos"],
                    ),
                )
            )

    def _setup_model(self) -> None:
        """
        Sets up the symbolic model for integration and generation of solvers in CasADi
        """
        # CasADi: states
        # r = [rx,ry,rz]': position vector
        rx, ry, rz = SX.sym("rx"), SX.sym("ry"), SX.sym("rz")
        r = ca.vertcat(rx, ry, rz)

        # q = [eta,eps1,eps2,eps3]': quaternion
        eta, eps1, eps2, eps3 = (
            SX.sym("eta"),
            SX.sym("eps1"),
            SX.sym("eps2"),
            SX.sym("eps3"),
        )
        q = ca.vertcat(eta, eps1, eps2, eps3)
        eps = ca.vertcat(eps1, eps2, eps3)

        # v = [vx,vy,vz]': velocity vector
        vx, vy, vz = SX.sym("vx"), SX.sym("vy"), SX.sym("vz")
        v = ca.vertcat(vx, vy, vz)

        # omega = [omega_x,omega_y,omega_z]': angular velocity vector
        omega_x, omega_y, omega_z = (
            SX.sym("omega_x"),
            SX.sym("omega_y"),
            SX.sym("omega_z"),
        )
        omega = ca.vertcat(omega_x, omega_y, omega_z)

        # x = [r,q,v,omega]': full state vector
        x = ca.vertcat(r, q, v, omega)

        # CasADi: inputs
        Fx, Fy, Fz = SX.sym("Fx"), SX.sym("Fy"), SX.sym("Fz")
        F = ca.vertcat(Fx, Fy, Fz)

        Tx, Ty, Tz = SX.sym("Tx"), SX.sym("Ty"), SX.sym("Tz")
        T = ca.vertcat(Tx, Ty, Tz)

        u = SX.sym("u", self.nu)  # Thruster inputs
        d = SX.sym("d", 6)

        # Casadi: derivatives
        rx_dot = SX.sym("rx_dot")
        ry_dot = SX.sym("ry_dot")
        rz_dot = SX.sym("rz_dot")
        r_dot = ca.vertcat(rx_dot, ry_dot, rz_dot)

        eta_dot = SX.sym("eta_dot")
        eps1_dot = SX.sym("eps1_dot")
        eps2_dot = SX.sym("eps2_dot")
        eps3_dot = SX.sym("eps3_dot")
        q_dot = ca.vertcat(eta_dot, eps1_dot, eps2_dot, eps3_dot)

        vx_dot = SX.sym("vx_dot")
        vy_dot = SX.sym("vy_dot")
        vz_dot = SX.sym("vz_dot")
        v_dot = ca.vertcat(vx_dot, vy_dot, vz_dot)

        omega_x_dot = SX.sym("omega_x_dot")
        omega_y_dot = SX.sym("omega_y_dot")
        omega_z_dot = SX.sym("omega_z_dot")
        omega_dot = ca.vertcat(omega_x_dot, omega_y_dot, omega_z_dot)

        x_dot = ca.vertcat(r_dot, q_dot, v_dot, omega_dot)

        # Casadi: parameters
        m = self.mass
        I = SX(3, 3)
        I[0, 0] = self.I11
        I[1, 1] = self.I22
        I[2, 2] = self.I33
        M_com = SX(6, 6)  # Full inertia matrix (6x6)
        M_com[0:3, 0:3] = m * SX.eye(3)
        M_com[3:6, 3:6] = I

        C_com = SX(6, 6)  # Coriolis matrix (6x6)
        C_com[0:3, 0:3] = m * ca.skew(omega)
        C_com[3:6, 3:6] = -ca.skew(ca.mtimes(I, omega))

        # System transformation matrix from CG to CO
        # CG = Center of Gravity, CO = Center origin (body frame)
        H = SX(6, 6)
        H[0:3, 0:3] = SX.eye(3)
        H[0:3, 3:6] = ca.transpose(ca.skew(np.array([0,0,0])))
        H[3:6, 3:6] = SX.eye(3)

        # Transform system matrices to body frame by similarity transformation
        M_body = ca.mtimes(ca.mtimes(ca.transpose(H), M_com), H)
        M_body_inv = ca.solve(M_body, SX.eye(M_body.size1()))
        C_body = ca.mtimes(ca.mtimes(ca.transpose(H), C_com), H)

        R_quat = SX(3, 3)
        S = ca.skew(eps)
        R_quat = SX.eye(3) + 2 * eta * S + 2 * ca.mtimes(S, S)

        T_quat = SX(4, 3)
        T_quat = 0.5 * np.array(
            [
                [-eps1, -eps2, -eps3],
                [eta, -eps3, eps2],
                [eps3, eta, -eps1],
                [-eps2, eps1, eta],
            ]
        )

        J_quat = SX(7, 6)
        J_quat[0:3, 0:3] = R_quat
        J_quat[3:7, 3:6] = T_quat

        dynamics = ca.vertcat(
            ca.mtimes(J_quat, ca.vertcat(v, omega)),
            ca.mtimes(
                M_body_inv,
                -ca.mtimes(C_body, ca.vertcat(v, omega)) + ca.mtimes(DM(self.mixer), u),
            ),
        )

        # Create model struct to collect all relevant components
        self.model = {"vars": {"x": x, "u": u}, "dynamics": dynamics}

        # Define explicit (continuous time) dynamics x_dot = f_expl(x,u,p)
        self.f_expl = ca.Function("f_expl", [x, u], [dynamics], ["x", "u"], ["f"])

        # Define integrator dynamics
        self.f_int = ca.integrator(
            "f_impl", "rk", {"x": x, "p": u, "ode": dynamics}, 0, 0.002, {"number_of_finite_elements": 1}
        )
