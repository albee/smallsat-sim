from smallsat_sim.controllers.base_controller import BaseController
from smallsat_sim.envs.base_env import BaseEnv
from smallsat_sim.utils.helpers import (
    calc_lateral_tracking_error,
    calc_attitude_error,
    quat_multiply,
    quat_conjugate,
    Rquat,
)

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
        self.Q = self.ctrl_cfg.cost.Q + 1e-5 * np.eye(
            self.nx, self.nx
        )  # Perturb for numerical stability
        self.R = self.ctrl_cfg.cost.R
        # Use reduced error-state for LQR (drop quaternion scalar component).
        self._red_idx = [0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12]
        self._red_nx = len(self._red_idx)
        self._last_stable_gain = np.zeros((self.nu, self._red_nx))
        self.dt = env.env_cfg.sim.dt * self.ctrl_cfg.control_decimation

    def check_controllability(self, A: np.ndarray, B: np.ndarray) -> bool:
        """
        Check if the system is controllable.
        """
        ctrb_matrix = control.ctrb(A, B)

        if np.linalg.matrix_rank(ctrb_matrix) != A.shape[0]:
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

        if not (np.all(np.isfinite(A)) and np.all(np.isfinite(B))):
            print("[LQR] Non-finite entries in linearization; reusing previous gain.")
            return self._last_stable_gain

        # Discretize the system (Euler discretization)
        A = np.eye(self.nx) + A * self.dt
        B = B * self.dt

        # Reduce to a 12D error-state by dropping quaternion scalar component.
        A_red = A[np.ix_(self._red_idx, self._red_idx)]
        B_red = B[self._red_idx, :]

        # Check if the system is controllable (for debugging purposes)
        _ = self.check_controllability(A_red, B_red)

        try:
            Q_red = self.Q[np.ix_(self._red_idx, self._red_idx)]
            P = solve_discrete_are(A_red, B_red, Q_red, self.R)
        except np.linalg.LinAlgError:
            print("[LQR] DARE failed; reusing previous gain.")
            return self._last_stable_gain

        # Compute the LQR gain
        K = np.linalg.inv(B_red.T @ P @ B_red + self.R) @ (B_red.T @ P @ A_red)

        self._last_stable_gain = K
        return K

    def _normalize_quat(self, q: np.ndarray) -> np.ndarray:
        eps = 1e-12
        n = np.linalg.norm(q)
        if n < eps:
            return np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
        return q / n

    def _quat_error(self, q_ref: np.ndarray, q: np.ndarray) -> np.ndarray:
        """
        Return error quaternion q_err = q_ref ⊗ conj(q),
        adjusted to the shortest path.
        """
        q_ref = self._normalize_quat(q_ref)
        q = self._normalize_quat(q)
        if np.dot(q_ref, q) < 0:
            q = -q
        q_err = quat_multiply(q_ref, quat_conjugate(q))
        if q_err[0] < 0:
            q_err = -q_err
        return q_err

    def _body_to_inertial_vel(self, q: np.ndarray, v_body: np.ndarray) -> np.ndarray:
        """
        Convert body-frame velocity to inertial frame using quaternion.
        """
        R_ib = Rquat(q)  # body -> inertial
        return (R_ib @ v_body.reshape(3, 1)).reshape(3,)

    def get_control_input(self, env: BaseEnv) -> np.ndarray:
        """
        Calculate the control input based on current observation.
        """

        # Get current state
        x = env.obs[0:13]
        r = x[0:3]
        q = x[3:7]
        v_body = x[7:10]
        omega = x[10:13]
        v_inertial = self._body_to_inertial_vel(q, v_body)

        # Get the reference position
        r_ref, quat_ref = self.planner.get_reference(env.obs)
        r_ref = np.asarray(r_ref).reshape(3,)
        quat_ref = np.asarray(quat_ref).reshape(4,)
        x_ref = np.concatenate(
            (
                r_ref,
                quat_ref,
                self.v_ref.reshape(3,),
                self.omega_ref.reshape(3,),
            )
        )

        # Compute the LQR gain around the reference state
        self.K = self.get_lqr_gain(x_ref)

        # Build error state (zero at reference)
        q_err = self._quat_error(quat_ref, q)
        eps_err = q_err[1:4]
        x_err = np.concatenate(
            (
                r - r_ref,
                eps_err,
                v_inertial - self.v_ref.reshape(3,),
                omega - self.omega_ref.reshape(3,),
            )
        )

        # Compute optimal control signal
        u_opt = -self.K @ x_err

        return u_opt

    def _log(self, run_id: int, timestamp: float, env: BaseEnv) -> None:
        """
        Logs desired quantities if flag is enabled
        """
        if self.has_logger:
            obs_gt = env.get_obs()

            tracking_error = calc_lateral_tracking_error(
                obs=obs_gt, planner=self.planner
            )
            q_ref = np.array([1.0, 0.0, 0.0, 0.0])
            if hasattr(self.planner, "trajectory") and hasattr(
                self.planner.trajectory, "get_intermediate_reference"
            ):
                _, curr_arc_length = self.planner.closest_point_on_trajectory(
                    point=obs_gt[:3]
                )
                q_ref = self.planner.trajectory.get_intermediate_reference(
                    curr_arc_length
                ).attitude
            attitude_error = calc_attitude_error(
                q_ref=np.asarray(q_ref).squeeze(), q=obs_gt[3:7]
            )

            self.logger.log(
                run_id=run_id,
                timestamp=timestamp,
                tracking_error=tracking_error,
                attitude_error=attitude_error,
            )
