from smallsat_sim.controllers.base_controller import BaseController
from smallsat_sim.envs.base_env import BaseEnv

import numpy as np
import casadi as ca
import control

from scipy.linalg import solve_discrete_are, solve_continuous_are, expm


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

        # Set reference values for parts of the states
        self.quat_ref = np.array([1.0, 0.0, 0.0, 0.0]).reshape(4,)
        self.v_ref = np.zeros((3, 1)).reshape(3,)
        self.omega_ref = np.zeros((3, 1)).reshape(3,)

        # Get the cost function matrices
        nx = self.ctrl_cfg.cost.Q.shape[0]
        self.Q = self.ctrl_cfg.cost.Q + 1e-5 * np.eye(nx, nx) # Perturb for numerical stability
        self.R = self.ctrl_cfg.cost.R

    
    def check_controllability(self, A, B) -> bool:
        """
        Check if the system is controllable.
        """
        ctrb_matrix = control.ctrb(A, B)
        if (np.linalg.matrix_rank(ctrb_matrix) != A.shape[0]):
            print("System is not controllable.")
            return False
        return True


    def get_lqr_gain(self, env: BaseEnv) -> np.ndarray:
        """
        Calculate the LQR gain.
        """

        # Fetch symbolic model
        model = env.symbolic_model
        f = model.f_expl_expr
        x = model.x
        u = model.u
        nx = x.shape[0]
        nu = u.shape[0]

        # Define the operating point
        x_s = ca.SX(env.obs[0:nx])
        u_s = ca.SX.zeros(model.nu, 1)

        # Linearize the model around the operating point
        A_sym = ca.jacobian(f, x)
        A_num = ca.substitute(A_sym, x, x_s)
        B_sym = ca.jacobian(f, u)
        B_num = ca.substitute(B_sym, u, u_s)

        # Convert the A and B matrices to NumPy
        A = np.zeros((nx, nx))
        for i in range(nx):
            for j in range(nx):
                A[i, j] = A_num[i, j]
        B = np.zeros((nx, nu))
        for i in range(nx):
            for j in range(nu):
                B[i, j] = B_num[i, j]

        # Discretize the system (Euler discretization)
        A = np.eye(nx) + A * 0.5
        B = B * 0.5

        # Check if the system is controllable (for debugging purposes)
        # is_controllable = self.check_controllability(A, B)

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
        self.K = self.get_lqr_gain(env)
        
        # Get the reference position
        r_ref = self.planner.get_reference(env.obs).reshape(3,)
        x_ref = np.concatenate((r_ref, self.quat_ref, self.v_ref, self.omega_ref))

        # Compute optimal control signal
        u_opt = - self.K @ (x - x_ref)

        return u_opt
