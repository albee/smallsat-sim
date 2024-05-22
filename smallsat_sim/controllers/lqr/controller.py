from smallsat_sim.controllers.base_controller import BaseController
from smallsat_sim.envs.base_env import BaseEnv

import numpy as np
import casadi as ca

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

        # Check if there is a viewer. In case there is not,
        # dynamically allocate the visualize method to a lambda
        # function doing nothing.
        if env.viewer:
            self.viewer = env.viewer
        else:
            self._visualize = lambda *args, **kwargs: None


    def get_ss_lqr_gain(self, env: BaseEnv) -> np.ndarray:
        """
        Calculate the LQR gain around a steady-state.
        """

        # Fetch symbolic model
        model = env.symbolic_model
        f = model.f_expl_expr
        x = model.x
        u = model.u

        # Define the operating point
        x_s = ca.SX.zeros(model.x.shape[0], 1)
        x_s[3] = 1.0
        u_s = ca.SX.zeros(model.u.shape[0], 1)

        # Linearize the model around the operating point
        A_sym = ca.jacobian(f, x)
        A_num = ca.substitute(A_sym, x, x_s)
        B_sym = ca.jacobian(f, u)
        B_num = ca.substitute(B_sym, u, u_s)

        # Convert the A and B matrices to NumPy
        A = np.zeros((A_num.shape[0], A_num.shape[1]))
        for i in range(A_num.shape[0]):
            for j in range(A_num.shape[1]):
                A[i, j] = A_num[i, j]
        B = np.zeros((B_num.shape[0], B_num.shape[1]))
        for i in range(B_num.shape[0]):
            for j in range(B_num.shape[1]):
                B[i, j] = B_num[i, j]

        # Get the cost function matrices
        Q = self.ctrl_cfg.cost.Q
        R = self.ctrl_cfg.cost.R

        # Solve the DARE
        P = solve_discrete_are(A, B, Q, R)

        # Compute the LQR gain
        K = np.linalg.inv(B.T @ P @ B + R) @ (B.T @ P @ A)

        return K

        

    def get_control_input(self, env: BaseEnv) -> np.ndarray:
        """
        Calculate the control input based on current observation.
        """

        # Get current state
        x = env.obs[0:13]

        # Get the LQR gain
        K = self.get_ss_lqr_gain(env) # Not necessary to compute it every time (move the function to dynamics.py)

        # Compute optimal control signal
        uopt = - K @ x # Penalize the diff. with x_s instead

        return uopt
