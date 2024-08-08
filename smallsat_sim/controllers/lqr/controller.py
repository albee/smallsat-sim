from smallsat_sim.controllers.base_controller import BaseController
from smallsat_sim.envs.base_env import BaseEnv

import numpy as np
import casadi as ca
import control

from scipy.linalg import solve_discrete_are


class LQRController(BaseController):
    """
    This is an implementation of a LQR for tracking problems. The reference is provided by the planner.
    """

    def __init__(self, env, planner) -> None:
        # Fetch correct controller config
        self.ctrl_cfg = env.env_cfg.control.LQR

        # Initialize base class
        super().__init__(env, planner, self.ctrl_cfg)

        # Fetch symbolic model
        self.model = env.symbolic_model
        self.f = self.model.f_expl_expr
        self.x = self.model.x
        self.u = self.model.u
        self.nx = self.x.shape[0]
        self.nu = self.u.shape[0]

        # Compute the Jacobians
        self.A_sym = ca.jacobian(self.f, self.x)
        self.B_sym = ca.jacobian(self.f, self.u)

        # Set reference values for parts of the states
        self.v_ref = np.zeros((3, 1))
        self.omega_ref = np.zeros((3, 1))

        # Get the cost function matrices
        self.Q = self.ctrl_cfg.cost.Q + 1e-5 * np.eye(self.nx, self.nx) # Perturb for numerical stability
        self.R = self.ctrl_cfg.cost.R

    
    def check_controllability(self, A: np.ndarray, B: np.ndarray) -> bool:
        """
        Check if the system is controllable.
        """
        ctrb_matrix = control.ctrb(A, B)

        if (np.linalg.matrix_rank(ctrb_matrix) != A.shape[0]):
            print("System is not controllable.")
            return False
        
        return True


    def get_lqr_gain(self, x: np.ndarray) -> np.ndarray:
        """
        Calculate the LQR gain.
        """

        # Define the operating point
        x_s = ca.SX(x)
        u_s = ca.SX.zeros(self.model.nu, 1)

        # Linearize the model around the operating point
        A_num = ca.substitute(self.A_sym, self.x, x_s)
        B_num = ca.substitute(self.B_sym, self.u, u_s)

        # Convert the A and B matrices to NumPy
        A = np.array(ca.DM(A_num).full())
        B = np.array(ca.DM(B_num).full())

        # Discretize the system (Euler discretization)
        A = np.eye(self.nx) + A * 0.5
        B = B * 0.5

        # Check if the system is controllable (for debugging purposes)
        is_controllable = self.check_controllability(A, B)

        # Solve the DARE
        P = solve_discrete_are(A, B, self.Q, self.R)

        # Compute the LQR gain
        K = np.linalg.inv(B.T @ P @ B + self.R) @ (B.T @ P @ A)

        return K

        
    def get_control_input(self, env: BaseEnv) -> np.ndarray:
        """
        Calculate the control input based on current observation.
        """

        # Get current state
        x = env.obs[0:13]

        # Compute the LQR gain
        self.K = self.get_lqr_gain(x)
        
        # Get the reference position
        r_ref, quat_ref = self.planner.get_reference(env.obs)
        x_ref = np.concatenate((r_ref, quat_ref, self.v_ref, self.omega_ref)).squeeze(-1)

        # Compute optimal control signal
        u_opt = - self.K @ (x - x_ref)

        return u_opt
