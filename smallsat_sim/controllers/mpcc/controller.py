from smallsat_sim.controllers.base_mpc_controller import BaseMPCController
from smallsat_sim.envs.base_env import BaseEnv
from smallsat_sim.planners.base_planner import BasePlanner
from smallsat_sim.utils.helpers import calc_lateral_tracking_error, calc_attitude_error

from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver

import numpy as np
import os
import casadi as ca
import mujoco

from casadi import SX


class NominalMPCCController(BaseMPCController):
    """
    This class implements a nominal MPC controller based on acados.
    More specifically. this is a positional tracking controller,
    taking waypoints from some external planner module.

    See acados documentation for details:
    https://docs.acados.org/
    """

    def __init__(self, env: BaseEnv, planner: BasePlanner) -> None:
        # Fetch correct controller config
        self.ctrl_cfg = env.env_cfg.control.NominalMPCC

        # Initialize base class
        super().__init__(env, planner, self.ctrl_cfg)

        # Generate solver
        self._generate_solver(env)

        # Initialize solver
        self._initialize_solver(env)

        # Check if there is a viewer. In case there is not,
        # dynamically allocate the visualize method to a lambda
        # function doing nothing.
        if env.viewer:
            self.viewer = env.viewer

            # Setup util parameters for visualization
            self.viz_offset = self.viewer.user_scn.ngeom
            self.viewer.user_scn.ngeom += self.ctrl_cfg.N + 1
        else:
            self._visualize_prediction = lambda *args, **kwargs: None

        # Check if there is a renderer. In case there is not,
        # dynamically allocate the visualize_renderer method
        # to a lambda function doing nothing.
        if env.renderer:
            self.renderer = env.renderer
            self.frames = env.frames
            self.data = env.data
            self.env_cfg = env.env_cfg
        else:
            self._visualize_prediction_renderer = lambda *args, **kwargs: None

    def _generate_solver(self, env) -> None:
        """
        This method generates the necessary solver C code
        """
        # Initialize OCP instance
        ocp = AcadosOcp()

        # Fetch symbolic model
        model = env.symbolic_model

        # Setup acados model (interface between acados and CasADi)
        acados_model = AcadosModel()
        acados_model.f_impl_expr = model.f_impl_expr
        acados_model.f_expl_expr = model.f_expl_expr
        acados_model.x = model.x
        acados_model.xdot = model.xdot
        acados_model.u = model.u
        acados_model.z = model.z
        acados_model.p = model.p
        acados_model.name = "OCPsolver"

        # Define theta and dtheta
        theta = ca.SX.sym("theta", 1, 1)
        d_theta = ca.SX.sym("d_theta", 1, 1)

        acados_model.x = ca.vertcat(acados_model.x, theta)
        acados_model.u = ca.vertcat(acados_model.u, d_theta)
        acados_model.xdot = ca.vertcat(acados_model.xdot, d_theta)

        acados_model.f_expl_expr = ca.vertcat(acados_model.f_expl_expr, d_theta)

        acados_model.f_impl_expr = ca.vertcat(acados_model.f_impl_expr, 0)

        # Assign parameters and model
        Ts = self.ctrl_cfg.Ts
        p_start = SX.sym("p_start", (3, 1))  # Start point of line segment<
        t = SX.sym("t", (3, 1))  # Direction of line segment
        theta_start = SX.sym("theta_start")  # Arclength start of line segment
        q_des = SX.sym("q_des", (4, 1))  # Desired attitude

        p = ca.vertcat(p_start, t, theta_start, q_des)
        acados_model.p = p
        ocp.model = acados_model

        # Define and assign cost functions
        R = self.ctrl_cfg.cost.R
        q_l = self.ctrl_cfg.cost.q_l
        Q_c = self.ctrl_cfg.cost.Q_c
        Q_omega = self.ctrl_cfg.cost.Q_omega
        r_d_theta = self.ctrl_cfg.cost.r_d_theta
        q_theta = self.ctrl_cfg.cost.q_theta
        Q_q = self.ctrl_cfg.cost.Q_q

        # Calculate line representation
        tx, ty, tz = t[0], t[1], t[2]
        g = p_start + (theta - theta_start) * t

        # Extract states for ease of use
        r = model.x[0:3]
        q = model.x[3:7]
        omega = model.x[7:10]

        # Calculate errors
        e = r - g
        e_l = t.T @ e

        # Normal projection matrix
        P_n = ca.SX(3, 3)
        P_n[0, 0] = 1 - tx**2
        P_n[0, 1] = -tx * ty
        P_n[0, 2] = -tx * tz
        P_n[1, 0] = -tx * ty
        P_n[1, 1] = 1 - ty**2
        P_n[1, 2] = -ty * tz
        P_n[2, 0] = -tx * tz
        P_n[2, 1] = -ty * tz
        P_n[2, 2] = 1 - tz**2

        # Contouring error
        e_c = P_n @ e

        # Quaternion error
        q_conj = np.array([q[0], -q[1], -q[2], -q[3]])
        e_q = np.array(
            [
                q_des[0] * q_conj[0]
                - q_des[1] * q_conj[1]
                - q_des[2] * q_conj[2]
                - q_des[3] * q_conj[3],
                q_des[0] * q_conj[1]
                + q_des[1] * q_conj[0]
                + q_des[2] * q_conj[3]
                - q_des[3] * q_conj[2],
                q_des[0] * q_conj[2]
                - q_des[1] * q_conj[3]
                + q_des[2] * q_conj[0]
                + q_des[3] * q_conj[1],
                q_des[0] * q_conj[3]
                + q_des[1] * q_conj[2]
                - q_des[2] * q_conj[1]
                + q_des[3] * q_conj[0],
            ]
        )

        # We only want to minimize eps part of error quaternion
        e_q = e_q[1:4]

        # Setup cost
        ocp.cost.cost_type = "EXTERNAL"
        ocp.model.cost_expr_ext_cost = (
            q_l * e_l * e_l
            + e_c.T @ Q_c @ e_c
            + e_q.T @ Q_q @ e_q
            + omega.T @ Q_omega @ omega
            + (model.u.T) @ R @ (model.u)
            + r_d_theta * d_theta**2
            - q_theta * theta
        )

        # Nonlinear constraint
        acados_model.con_h_expr = e.T @ e
        ocp.constraints.lh = np.array([0.0])
        ocp.constraints.uh = np.array([1.0 * 1.0])

        # Set OCP dimensions
        nx = acados_model.x.size()[0]  # number of states
        nu = acados_model.u.size()[0]  # number of inputs
        ocp.dims.nx = nx
        ocp.dims.nu = nu
        ocp.dims.np = p.size()[0]  # number of parameters
        ocp.dims.N = self.ctrl_cfg.N  # prediction horizon length
        if acados_model.con_h_expr is not None:
            ocp.dims.nh = acados_model.con_h_expr.size()[0]
        else:
            ocp.dims.nh = 0

        if acados_model.con_h_expr_e is not None:
            ocp.dims.nh_e = acados_model.con_h_expr_e.size()[0]
        else:
            ocp.dims.nh_e = 0

        # Define state constraints
        # Lower and Upper bound constraints for intermediate stages
        ocp.constraints.lbx = np.array(
            [
                -100,
                -100,
                -100,
                -1.1,
                -1.1,
                -1.1,
                -1.1,
                -0.4,
                -0.4,
                -0.4,
                -0.5,
                -0.5,
                -0.5,
                0,
            ]
        )
        ocp.constraints.ubx = np.array(
            [100, 100, 100, 1.1, 1.1, 1.1, 1.1, 0.4, 0.4, 0.4, 0.5, 0.5, 0.5, 1000]
        )
        ocp.constraints.idxbx = np.arange(nx)

        # Define input constraints
        # Fetch thurster limits from the model configuration
        thruster_forces = [
            thruster.forcerange for thruster in env.model_cfg.Thrusters.thruster_list
        ]
        ocp.constraints.lbu = np.array([forces[0] for forces in thruster_forces])
        ocp.constraints.ubu = np.array([forces[1] for forces in thruster_forces])
        ocp.constraints.idxbu = np.arange(nu - 1)

        # Set intial condition
        ocp.constraints.idxbx_0 = np.arange(nx - 1)
        ocp.constraints.lbx_0 = env.obs[0:13].copy()
        ocp.constraints.ubx_0 = env.obs[0:13].copy()
        ocp.parameter_values = np.zeros(ocp.dims.np)

        # Configure solver options
        ocp.solver_options.Tsim = Ts
        ocp.solver_options.tf = Ts * self.ctrl_cfg.N
        ocp.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM"
        ocp.solver_options.nlp_solver_type = "SQP_RTI"
        ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
        ocp.solver_options.integrator_type = "ERK"
        ocp.solver_options.print_level = 0

        # Set code generation directory
        save_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "c_generated_code"
        )
        ocp.code_export_directory = save_dir

        # Create solver with agent specific code files
        filename = os.path.join(save_dir, "acados_pacejka_mpcc_solver_config.json")

        self.ocp_solver = AcadosOcpSolver(ocp, json_file=filename)

        print("Solver generated successfully.")

    def _initialize_solver(self, env: BaseEnv) -> np.ndarray:
        """
        Initializes the solver. Also known as "warm start".
        """

        # Retrieve closest point on track (relevant for theta)
        _, theta_init = self.planner.closest_point_on_trajectory(env.obs[0:3])

        # Array to store previous theta
        self.theta_prev = [theta_init for i in range(self.ctrl_cfg.N + 1)]

        # Warm start solver
        # Initial condition and Warm start
        x_guess = np.zeros((14, 1))
        x_guess[0:13, 0] = env.obs[0:13].copy()
        x_guess[-1, 0] = theta_init

        [self.ocp_solver.set(i, "x", x_guess) for i in range(self.ctrl_cfg.N + 1)]
        [self.ocp_solver.set(i, "u", np.zeros((13, 1))) for i in range(self.ctrl_cfg.N)]

    def get_control_input(self, env: BaseEnv) -> np.ndarray:
        """
        Calculate the control input based on current observation
        """
        # Check solver status and re-initialize if needed
        if self.ocp_solver.status != 0:
            print(f"Solution optimal, solver status: {self.ocp_solver.status}")
            self._initialize_solver(env)

        # Set parameters
        self._set_params()

        # Solve for the first control input in receding horizon fashion
        u0 = self.ocp_solver.solve_for_x0(
            env.obs[0:13], print_stats_on_failure=True, fail_on_nonzero_status=False
        )
        self._visualize_prediction()

        if hasattr(self, "renderer") and self.renderer is not None:
            _, _ = self.planner.get_reference(env.obs)
            self._visualize_prediction_renderer()

        if False:
            solve_time = self.ocp_solver.get_stats("time_tot")
            print(f"Solve time: {solve_time}")

        # Save theta for next iteration
        for i in range(self.ctrl_cfg.N + 1):
            self.theta_prev[i] = self.ocp_solver.get(i, "x")[-1]

        # Log quantities
        self._log(run_id=env.run_id, timestamp=env.data.time, env=env)

        return u0[0:12]

    def _set_params(self) -> None:
        """
        Sets the parameters of the solver at runtime
        """
        # Shift the previous solution for theta by one for reinitialization
        theta_shifted = self.theta_prev.copy()
        theta_shifted.append(theta_shifted[-1])
        theta_shifted.pop(0)

        # Set parameters
        for i in range(self.ctrl_cfg.N + 1):

            theta_curr = theta_shifted[i]

            p_start = self.planner.trajectory._get_start_point_segment(theta_curr)
            t = self.planner.trajectory._get_tangent_segment(theta_curr)
            theta_1 = self.planner.trajectory._get_start_arc_length_segment(theta_curr)
            q_ref = self.planner.trajectory.get_intermediate_reference(
                theta_curr
            ).attitude

            ref = np.concatenate((p_start, t, theta_1, q_ref))

            self.ocp_solver.set(i, "p", ref)

    def _log(self, run_id: int, timestamp: float, env: BaseEnv) -> None:
        """
        Logs desired quantities if flag is enabled
        """
        if self.has_logger:
            obs_gt = (
                env.get_obs()
            )  # TODO: Change this to get GT obs, once MR has been merged

            # Tracking error
            tracking_error = calc_lateral_tracking_error(
                obs=obs_gt, planner=self.planner
            )

            # Attitude error
            _, curr_arc_length = self.planner.closest_point_on_trajectory(
                point=obs_gt[:3]
            )
            q_ref = self.planner.trajectory.get_intermediate_reference(
                curr_arc_length
            ).attitude

            attitude_error = calc_attitude_error(q_ref=q_ref, q=obs_gt[3:7])

            # Log quantities
            self.logger.log(
                run_id=run_id,
                timestamp=timestamp,
                tracking_error=tracking_error,
                attitude_error=attitude_error,
            )

    def _visualize_prediction(self) -> None:
        """
        Plot predicted trajectory of MPC in MuJoCo viewer.
        """
        for i in range(self.ctrl_cfg.N + 1):
            point = self.ocp_solver.get(i, "x")[0:3]
            mujoco.mjv_initGeom(
                self.viewer.user_scn.geoms[i + self.viz_offset],
                type=mujoco.mjtGeom.mjGEOM_SPHERE,
                size=[0.05, 0, 0],
                pos=point,
                mat=np.eye(3).flatten(),
                rgba=np.array([0, 0, 1, 2]),
            )

    def _visualize_prediction_renderer(self) -> None:
        """
        Plot predicted trajectory of MPC in MuJoCo renderer.
        """
        if (
            self.data.time >= self.env_cfg.renderer.start_recording
            and self.data.time <= self.env_cfg.renderer.end_recording
        ):
            for i in range(self.ctrl_cfg.N + 1):
                point = self.ocp_solver.get(i, "x")[0:3]
                mujoco.mjv_initGeom(
                    self.renderer.scene.geoms[i + self.renderer.scene.ngeom],
                    type=mujoco.mjtGeom.mjGEOM_SPHERE,
                    size=[0.05, 0, 0],
                    pos=point,
                    mat=np.eye(3).flatten(),
                    rgba=np.array([0, 0, 1, 2]),
                )

            self.renderer.scene.ngeom += self.ctrl_cfg.N + 1

            # Extract image from renderer and append it for post-processing
            sim_img = self.renderer.render().copy()
            self.frames.append(sim_img)
