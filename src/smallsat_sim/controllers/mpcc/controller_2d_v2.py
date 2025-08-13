from smallsat_sim.envs.base_env import BaseEnv
from smallsat_sim.utils.conversions import convert_6dof_3dof, yaw_from_quaternion
from smallsat_sim.controllers.mpcc.controller import NominalMPCCController
from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver
# from smallsat_sim.utils.helpers import calc_lateral_tracking_error, calc_attitude_error

import numpy as np
import os
import casadi as ca
import mujoco

from casadi import SX
import time


class NominalMPCCController2D(NominalMPCCController):
    """
    This class implements a nominal MPC controller based on acados.
    More specifically. this is a positional tracking controller,
    taking waypoints from some external planner module.

    See acados documentation for details:
    https://docs.acados.org/
    """


    def _generate_solver(self, env: BaseEnv) -> None:
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

        # =============== COST ===============

        # Define and assign cost functions
        R = self.ctrl_cfg.cost.R
        Q_l = self.ctrl_cfg.cost.q_l
        Q_c = self.ctrl_cfg.cost.Q_c
        Q_omega = self.ctrl_cfg.cost.Q_omega
        q_theta = self.ctrl_cfg.cost.q_theta
        Q_q = self.ctrl_cfg.cost.Q_q

        # Extract states for ease of use (Note this is the BaseEnv model, i.e. 6Dof)
        r = np.array([model.x[0], model.x[1], 0])
        yaw = model.x[2]
        q = np.array([np.cos(yaw/2), 0, 0, np.sin(yaw/2)]) # q = qw, qx, qy, qz
        omega = np.array([0, 0,  model.x[5]])

        # Calculate line representation. Direction of line segment is t
        tx, ty, tz = t[0], t[1], t[2]
        g = p_start + (theta - theta_start) * t # why is this not just p_end - p_start??? Aren't all way points connected by straight lines?

        # Calculate errors 
        # e < R enforces staying in tube of radius R around reference g
        # r = position of agent, g = reference line
        error_to_ref = r - g
        # Lag Error
        e_l = t.T @ error_to_ref

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

        # Contouring Error
        e_c = P_n @ error_to_ref

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
        e_q_vec = e_q - ca.DM([1, 0, 0, 0])

        # Setup cost
        ocp.cost.cost_type = "EXTERNAL"
        ocp.model.cost_expr_ext_cost = (
            Q_l * e_l * e_l
            + e_c.T @ Q_c @ e_c
            + e_q_vec.T @ (Q_q) @ e_q_vec
            + omega.T @ Q_omega @ omega
            + (model.u.T) @ R @ (model.u)
            - q_theta * d_theta * d_theta
        )

        # Create a casadi cost function for numerical evaluation
        self.cost_function = ca.Function(
            "cost_function",
            [acados_model.x, acados_model.u, p_start, t, theta_start, q_des],
            [ocp.model.cost_expr_ext_cost],
        )

        # =================== CONSTRAINTS ===================
        # Nonlinear constraint to stay in tube around reference: error_to_ref_x^2 + error_to_ref_y^2 < R
        # acados_model.con_h_expr = error_to_ref.T @ error_to_ref
        # Terminal constraint
        # acados_model.con_h_expr_e = acados_model.con_h_expr

        # =================== SET UP SOLVER ===================
        # set dimensions
        nx = acados_model.x.size()[0]  # number of states
        nu = acados_model.u.size()[0]  # number of inputs
        ny = 0
        ny_e = 0

        ocp.dims.nx = nx
        ocp.dims.np = p.size()[0]  # number of parameters
        ocp.dims.ny = ny
        ocp.dims.ny_e = ny_e
        ocp.dims.nbx = nx
        ocp.dims.nbu = nu
        ocp.dims.nu = nu
        ocp.dims.nh = 0 #1
        ocp.dims.nsh = 0 #1
        ocp.dims.ns = nx + nu + ocp.dims.nsh
        ocp.dims.nh_e = 0 #acados_model.con_h_expr_e.size()[0]
        ocp.dims.nsh_e = 0 #1

        ocp.dims.nsbx = nx
        ocp.dims.nsbu = nu

        ocp.dims.N = self.ctrl_cfg.N  # prediction horizon length

        ocp.cost.cost_type = "EXTERNAL"
        ocp.cost.cost_type_e = "EXTERNAL"

        # ocp.model.cost_expr_ext_cost = model.cost_expr_ext_cost
        # ocp.model.cost_expr_ext_cost_e = 0

        # ocp.constraints.lh = np.array([0.0])
        # ocp.constraints.uh = np.array([1.0 * 1.0])

        # ocp.constraints.lsh = np.zeros(ocp.dims.nsh)
        # ocp.constraints.ush = np.zeros(ocp.dims.nsh)
        # ocp.constraints.idxsh = np.array(range(ocp.dims.nsh))

        # ocp.constraints.lh_e = np.array([0.0])
        # ocp.constraints.uh_e = np.array([1.0 * 1.0])

        ocp.constraints.lsbx = np.zeros(ocp.dims.nsbx)
        ocp.constraints.usbx = np.zeros(ocp.dims.nsbx)
        ocp.constraints.idxsbx = np.arange(nx)

        ocp.constraints.lsbu = np.zeros(ocp.dims.nsbu)
        ocp.constraints.usbu = np.zeros(ocp.dims.nsbu)
        ocp.constraints.idxsbu = np.arange(nu)

        ocp.cost.Zl = 5e02 * np.ones(ocp.dims.ns)
        ocp.cost.Zu = 5e02 * np.ones(ocp.dims.ns)
        ocp.cost.zl = 5e03 * np.ones(ocp.dims.ns)
        ocp.cost.zu = 5e03 * np.ones(ocp.dims.ns)

        # Configure slacks on initial state
        ocp.cost.zl_0 = 100 * np.ones(nu)
        ocp.cost.zu_0 = 100 * np.ones(nu)
        ocp.cost.Zl_0 = np.zeros(nu)
        ocp.cost.Zu_0 = np.zeros(nu)

        # State Constraints
        ocp.constraints.lbx = np.array(  # lower bounds on x
            [
                -100,  # x
                -100,  # y
                -1000,  # yaw
                -0.1,  # vx
                -0.1,  # vy
                -0.1,  # omega_z
                0,  # theta
            ]
        )

        ocp.constraints.ubx = np.array(  # upper bounds on x
            [
                100,  # x
                100,  # y
                1000,  # yaw
                0.1,  # vx
                0.1,  # vy
                0.1,  # omega_z
                1000,  # theta
            ]
        )
        ocp.constraints.idxbx = np.arange(nx)  # indices of bounds on x

        # Input Constraints
         # Fetch thurster limits from the model configuration
        thruster_forces = [
            thruster.forcerange for thruster in env.model_cfg.Thrusters.thruster_list
        ]
        ocp.constraints.lbu = np.array([forces[0] for forces in thruster_forces])
        ocp.constraints.ubu = np.array([forces[1] for forces in thruster_forces])
        # Attach dtheta constraints
        ocp.constraints.lbu = np.append(ocp.constraints.lbu, 0.0)
        ocp.constraints.ubu = np.append(ocp.constraints.ubu, 0.2)
        ocp.constraints.idxbu = np.arange(nu)

        # set intial condition
        ocp.constraints.x0 = np.zeros(nx)
        ocp.constraints.idxbx_0 = np.arange(nx)
        ocp.parameter_values = np.zeros(ocp.dims.np)

        # # Set OCP dimensions
        # nx = acados_model.x.size()[0]  # number of states
        # nu = acados_model.u.size()[0]  # number of inputs
        # ocp.dims.nx = nx
        # ocp.dims.nsbx = nx
        # # ocp.dims.nsbu = nu
        # ocp.dims.nu = nu
        # ocp.dims.np = p.size()[0]  # number of parameters
        # ocp.dims.N = self.ctrl_cfg.N  # prediction horizon length
        
        # ocp.dims.nh = 0
        # ocp.dims.nsh = 0

        # # Terminal nonlinear constraints
        # ocp.dims.nh_e = 0
        # ocp.dims.nsh_e = 0

        # # Total number of slacks at stages (1, N-1)
        # ocp.dims.ns = nx + ocp.dims.nsh

        # # Define state constraints
        # # Lower bound constraints for intermediate stages
        # ocp.constraints.lbx = np.array(
        #     [
        #         -100,  # x
        #         -100,  # y
        #         -1000,  # yaw
        #         -0.1,  # vx
        #         -0.1,  # vy
        #         -0.1,  # omega_z
        #         0,  # theta
        #     ]
        # )

        # # Upper bound constraints for intermediate stages
        # ocp.constraints.ubx = np.array(
        #     [
        #         100,  # x
        #         100,  # y
        #         1000,  # yaw
        #         0.1,  # vx
        #         0.1,  # vy
        #         0.1,  # omega_z
        #         1000,  # theta
        #     ]
        # )

        # # Indexes of the state variables to which the constraints apply
        # ocp.constraints.idxbx = np.arange(nx)

        # # Slacks on lower/upper bounds
        # ocp.constraints.lsbx = np.zeros(ocp.dims.nsbx)
        # ocp.constraints.usbx = np.zeros(ocp.dims.nsbx)
        # ocp.constraints.usbx[3:5] = 0.05
        # ocp.constraints.usbx[5:] = 0.02
        # ocp.constraints.idxsbx = np.arange(nx)

        # ocp.cost.Zl = 5e02 * np.ones(ocp.dims.ns)
        # ocp.cost.Zu = 5e02 * np.ones(ocp.dims.ns)
        # ocp.cost.zl = 5e03 * np.ones(ocp.dims.ns)
        # ocp.cost.zu = 5e03 * np.ones(ocp.dims.ns)

        # # Define input constraints
        # # Fetch thurster limits from the model configuration
        # thruster_forces = [
        #     thruster.forcerange for thruster in env.model_cfg.Thrusters.thruster_list
        # ]
        # ocp.constraints.lbu = np.array([forces[0] for forces in thruster_forces])
        # ocp.constraints.ubu = np.array([forces[1] for forces in thruster_forces])

        # # Attach dtheta constraints
        # ocp.constraints.lbu = np.append(ocp.constraints.lbu, 0.0)
        # ocp.constraints.ubu = np.append(ocp.constraints.ubu, 0.2)
        # ocp.constraints.idxbu = np.arange(nu)

        # # Set intial condition
        # ocp.constraints.idxbx_0 = np.arange(nx)
        # ocp.constraints.lbx_0 = ocp.constraints.lbx
        # ocp.constraints.ubx_0 = ocp.constraints.ubx
        # ocp.parameter_values = np.zeros(ocp.dims.np)

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
        _, theta_init = self.planner.closest_point_on_trajectory(env.get_obs()[0:3])

        # Array to store previous theta
        self.theta_prev = [theta_init for i in range(self.ctrl_cfg.N + 1)]

        # Warm start solver
        # Initial condition and Warm start
        v_init = 0.00
        Ts = self.ctrl_cfg.Ts
        distance_on_track = theta_init
        x_guess = np.zeros((7, 1))
        u_guess = np.random.uniform(low=0.0, high=0.3, size=(9, 1))

        for i in range(self.ctrl_cfg.N + 1):
            curr_vel = (0.05 - v_init) * i / self.ctrl_cfg.N + v_init
            distance_on_track += curr_vel * Ts

            point = self.planner.trajectory.get_intermediate_reference(
                distance_on_track
            )

            obs_6dof = env.get_obs()

            obs_3dof = convert_6dof_3dof(obs_6dof)
            x_guess[0:6, 0] = obs_3dof
            x_guess[0:2, 0] = point.position[0:2]
            x_guess[3, 0] = yaw_from_quaternion(point.attitude)
            x_guess[4, 0] = -curr_vel
            x_guess[-1] = distance_on_track

            u_guess[-1] = (0.05 - v_init) / self.ctrl_cfg.N

            self.ocp_solver.set(i, "x", x_guess)

            if i < self.ctrl_cfg.N:
                self.ocp_solver.set(i, "u", u_guess)

    def get_control_input(self, env: BaseEnv) -> np.ndarray:
        """
        Calculate the control input based on current observation
        """
        start_time = time.time()
        # Check solver status and re-initialize if needed
        if self.ocp_solver.status != 0:
            print(f"Solution optimal, solver status: {self.ocp_solver.status}")
            self._initialize_solver(env)

        # Set parameters
        self._set_params(env.get_obs())

        # Solve for the first control input in receding horizon fashion
        xinit = np.append(convert_6dof_3dof(env.get_obs()), self.theta_prev[1])
        u0 = self.ocp_solver.solve_for_x0(
            xinit, print_stats_on_failure=True, fail_on_nonzero_status=False
        )
        self._visualize_prediction()

        if hasattr(self, "renderer") and self.renderer is not None:
            _, _ = self.planner.get_reference(env.obs)
            self._visualize_prediction_renderer()

        if False:
            solve_time = self.ocp_solver.get_stats("time_tot")
            print(f"Solve time: {solve_time}")

        # Save current observation and input
        self.u_past = u0

        # Save theta for next iteration
        for i in range(self.ctrl_cfg.N + 1):
            self.theta_prev[i] = self.ocp_solver.get(i, "x")[-1]
            # print(self.ocp_solver.get(i, "x"))

        # Log quantities
        if self.has_logger:
            self._log(run_id=env.run_id, timestamp=env.data.time, env=env)

        # Record the end time
        end_time = time.time()
        # Calculate the duration
        self.ctrl_input_callback_time = end_time - start_time

        return u0[0:8].copy()

    def _visualize_prediction(self) -> None:
        """
        Plot predicted trajectory of MPC in MuJoCo viewer.
        """
        for i in range(self.ctrl_cfg.N + 1):
            point = self.ocp_solver.get(i, "x")[0:3]
            point[2] = 0.4
            mujoco.mjv_initGeom(
                self.viewer.user_scn.geoms[i + self.viz_offset],
                type=mujoco.mjtGeom.mjGEOM_SPHERE,
                size=[0.01, 0, 0],
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
                point[2] = 0.4
                mujoco.mjv_initGeom(
                    self.renderer.scene.geoms[i + self.renderer.scene.ngeom],
                    type=mujoco.mjtGeom.mjGEOM_SPHERE,
                    size=[0.01, 0, 0],
                    pos=point,
                    mat=np.eye(3).flatten(),
                    rgba=np.array([0, 0, 1, 2]),
                )

            self.renderer.scene.ngeom += self.ctrl_cfg.N + 1

            # Extract image from renderer and append it for post-processing
            sim_img = self.renderer.render().copy()
            self.frames.append(sim_img)

