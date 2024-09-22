# Base classes
from smallsat_sim.controllers.base_mpc_controller import BaseMPCController
from smallsat_sim.envs.base_env import BaseEnv

# Utils
from smallsat_sim.utils.helpers import (
    calc_model_error,
    calc_attitude_error,
    calc_lateral_tracking_error,
)
from smallsat_sim.utils.logger import Logger
from smallsat_sim.controllers.gp_mpc.online_learning.utils import (
    ScaleFeatureSelector,
    ResidualScaler,
)

# General libraries
import gpytorch
from gpytorch.constraints.constraints import Positive
import os
import numpy as np
import torch
import casadi as ca
import mujoco
from scipy.stats import norm
import time

# Acados
from acados_template import (
    AcadosModel,
    AcadosOcp,
    AcadosOcpSolver,
    AcadosSimSolver,
    ZoroDescription,
)
from casadi import SX

# Zero Order GPMPC
import zero_order_gpmpc
from zero_order_gpmpc.controllers import (
    ZeroOrderGPMPC,
)
from zero_order_gpmpc.controllers.zoro_acados_utils import (
    setup_sim_from_ocp,
)

# gpytorch utilities
from smallsat_sim.external.zero_order_gp_mpc_package.external.gpytorch_utils.gp_hyperparam_training import (
    generate_train_inputs_acados,
    generate_train_outputs_at_inputs,
    train_gp_model,
)


from smallsat_sim.external.zero_order_gp_mpc_package.external.gpytorch_utils.gp_utils import (
    gp_data_from_model_and_path,
    gp_derivative_data_from_model_and_path,
    plot_gp_data,
    generate_grid_points,
)

from zero_order_gpmpc.models.gpytorch_models.gpytorch_residual_model import (
    GPyTorchResidualModel,
)

from zero_order_gpmpc.models.gpytorch_models.gpytorch_residual_learning_model import (
    GPyTorchResidualLearningModel,
)

# GPyTorch models
from zero_order_gpmpc.models.gpytorch_models.gpytorch_gp import (
    BatchIndependentMultitaskGPModel,
)

# Import DataProcessing strategies from SmallSatSim
from smallsat_sim.controllers.gp_mpc.online_learning.strategies import (
    SlidingWindow,
    SlidingWindowPlus,
)


# Set default torch dtype
torch.set_default_dtype(torch.float64)


class GPMPC(BaseMPCController):
    """
    This class implements a GP MPC controller based on GPyTorch and acados.

    See the following documentations for details:
    GPyTorch: https://docs.gpytorch.ai/en/stable/
    acados: https://docs.acados.org/
    """

    def __init__(self, env: BaseEnv, planner) -> None:
        # Fetch correct controller config
        self.ctrl_cfg = env.env_cfg.control.GPMPC
        self.N = self.ctrl_cfg.N

        # Initialize base class
        super().__init__(env, planner, self.ctrl_cfg)

        # Save relevant components from environment locally
        self.f = env.symbolic_model.f_expl_expr_func
        self.f_int = env.symbolic_model.get_integrator(dt=self.ctrl_cfg.Ts)

        # # Initialize logger
        # self.logger = Logger()
        # _, self.logged_data = self.logger.load_data_from_hdf5(
        #     "mujoco_log_20240707_145725.h5"
        # )

        # Generate solver
        self._generate_nominal_ocp(env)

        # Create Zoro description
        self._create_zoro_description(env)

        # Setup Gaussian Process
        self._setup_gp(obs=env.get_obs())

        # Miscellaneous
        self.last_solution = {
            "states": np.tile(
                np.zeros(
                    self.nx,
                ),
                (self.N + 1, 1),
            ),
            "inputs": np.tile(
                np.zeros(
                    self.nu,
                ),
                (self.N, 1),
            ),
        }

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

    def _setup_gp(self, obs: np.ndarray) -> None:
        """
        Initializes all needed quantities for the Gaussian Process
        """
        # # Initialize empty feature tensor (z)
        # self.z = torch.from_numpy(self.logged_data["z"].squeeze(1))

        # # Initialize empty ouput tensor (y)
        # self.y = torch.from_numpy(self.logged_data["y"].squeeze(1))

        # Initialize subspace matrix B_d
        # Shape: (nx, 6)
        self.B_d = torch.cat((torch.zeros(7, 6), torch.eye(6), torch.zeros(1, 6))).to(
            torch.float64
        )
        self.B_d_inv = torch.linalg.pinv(self.B_d).to(torch.float64)

        # Initialize buffers to keep track of past (phyisical) states and inputs
        self.x_past = np.zeros(self.nx)
        self.x_past[:-1] = obs
        self.u_past = np.zeros(self.nu)

        # Initialize some hyperparameters for GP
        # TODO: Move to configuration file
        self.M = 300  # number of points in list
        self.gp_update_counter = 0  # Keep track how many times dict has been updated
        self.gp_initialized = False  # Keep track if GP is already initialized

        # Setup residual model with trained GP
        input_feature_selection = np.array(
            [
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                0,
            ]
        )
        self.input_selection = ScaleFeatureSelector(input_feature_selection)
        self.residual_scaler = ResidualScaler(scale=1)

        # Initialize likelihood and overwrite default values
        likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(
            num_tasks=6, noise_constraint=Positive()
        )

        # Hardcode noise parameters in likelihood
        likelihood.noise = torch.tensor([1e-12])
        likelihood.raw_task_noises.data = torch.tensor(
            [-30.0, -30.0, -30.0, -30.0, -30.0, -30.0]
        )

        # Initialize GP model and overwrite default values
        train_x = torch.zeros(1,12)
        train_y = torch.zeros(1,6)
        mode = "Nonlinear Kernel"
        gp_model = BatchIndependentMultitaskGPModel(
            train_x=train_x,
            train_y=train_y,
            likelihood=likelihood,
            residual_dimension=6,
            input_dimension=sum(input_feature_selection),
            use_ard=True,
            mode=mode,
        )

        for name, param in gp_model.named_parameters():
            print(f"Parameter {name} has shape {param.shape} and values:")
            print(param)

        # Set hyperparemters of linear kernel manually.
        gp_model = self._initialize_hyperparameters(gp_model, likelihood, mode)

        # Initialize the Residual Model
        residual_model = GPyTorchResidualLearningModel(
            gp_model=gp_model,
            gp_feature_selector=self.input_selection,
            residual_scaler=self.residual_scaler,
            data_processing_strategy=SlidingWindowPlus(
                max_num_points=self.M, device=next(gp_model.parameters()).device.type
            ),
            verbose=False,
        )

        # File naming stuff
        json_ocp = "zoro_ocp_solver_config.json"
        json_sim = "zoro_sim_solver_config.json"
        filename_ocp = os.path.join(self.save_dir, json_ocp)
        filename_sim = os.path.join(self.save_dir, json_sim)

        self.ocp_init.code_export_directory = os.path.join(
            self.save_dir, "c_generated_code_ocp"
        )

        self.nominal_sim.code_export_directory = os.path.join(
            self.save_dir, "c_generated_code_sim"
        )

        # Generate ZeroOrderGPMPC
        self.gp_mpc = ZeroOrderGPMPC(
            self.ocp_init,
            residual_model=residual_model,
            path_json_ocp=filename_ocp,
            path_json_sim=filename_sim,
            build_c_code=True,
            use_cython=False,  # TODO: Check why not supported
            B=self.B_d.numpy(),
        )

    def _initialize_hyperparameters(self, gp_model, likelihood, mode):
        if mode == "Linear Kernel":
            gp_model.covar_module.variance = torch.tensor([1e-6])
        elif mode == "Nonlinear Kernel":
            pass
        else:
            raise RuntimeError(f"Mode {mode} not known.")

        gp_model.eval()
        likelihood.eval()

        return gp_model

    def _generate_nominal_ocp(self, env) -> None:
        """
        Generates the nominal ocp (w/o residual dynamics)
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
        q_theta = self.ctrl_cfg.cost.q_theta
        Q_q = self.ctrl_cfg.cost.Q_q

        # Calculate line representation
        tx, ty, tz = t[0], t[1], t[2]
        g = p_start + (theta - theta_start) * t

        # Extract states for ease of use
        r = model.x[0:3]
        q = model.x[3:7]
        v = model.x[7:10]
        omega = model.x[10:13]

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
        e_q_vec = e_q - ca.DM([1, 0, 0, 0])

        # Setup cost
        ocp.cost.cost_type = "EXTERNAL"
        ocp.model.cost_expr_ext_cost = (
            q_l * e_l * e_l
            + e_c.T @ Q_c @ e_c
            + e_q_vec.T @ Q_q @ e_q_vec
            + omega.T @ Q_omega @ omega
            + (model.u.T) @ R @ (model.u)
            - q_theta * d_theta
        )

        # Create a casadi cost function for numerical evaluation
        self.cost_function = ca.Function(
            "cost_function",
            [acados_model.x, acados_model.u, p_start, t, theta_start, q_des],
            [ocp.model.cost_expr_ext_cost],
        )

        # Nonlinear constraint
        acados_model.con_h_expr = e.T @ e
        ocp.constraints.lh = np.array([0.0])
        ocp.constraints.uh = np.array([1.0 * 1.0])

        # Terminal constraint
        # acados_model.con_h_expr_e = acados_model.con_h_expr
        # ocp.constraints.lh_e = np.array([0.0])
        # ocp.constraints.uh_e = np.array([0.1 * 0.1])

        # Set OCP dimensions
        nx = acados_model.x.size()[0]  # number of states
        nu = acados_model.u.size()[0]  # number of inputs
        ocp.dims.nx = nx
        self.nx = nx
        ocp.dims.nsbx = nx
        ocp.dims.nu = nu
        self.nu = nu
        ocp.dims.np = p.size()[0]  # number of parameters
        ocp.dims.N = self.ctrl_cfg.N  # prediction horizon length

        # Nonlinear constraints
        if acados_model.con_h_expr is not None:
            ocp.dims.nh = acados_model.con_h_expr.size()[0]
            ocp.dims.nsh = 1
        else:
            ocp.dims.nh = 0
            ocp.dims.nsh = 0
        ocp.constraints.idxsh = np.array(range(ocp.dims.nsh))

        # Terminal nonlinear constraints
        if acados_model.con_h_expr_e is not None:
            ocp.dims.nh_e = acados_model.con_h_expr_e.size()[0]
            ocp.dims.nsh_e = 1
        else:
            ocp.dims.nh_e = 0
            ocp.dims.nsh_e = 0

        # Total number of slacks at stages (1, N-1)
        ocp.dims.ns = nx + ocp.dims.nsh

        # Define state constraints
        # Lower bound constraints for intermediate stages
        ocp.constraints.lbx = np.array(
            [
                -100,  # x
                -100,  # y
                -100,  # z
                -1.0,  # q[0]
                -1.0,  # q[1]
                -1.0,  # q[2]
                -1.0,  # q[3]
                -0.3,  # vx
                -0.3,  # vy
                -0.3,  # vz
                -0.1,  # omega_x
                -0.1,  # omega_y
                -0.1,  # omega_z
                0,  # theta
            ]
        )

        # Upper bound constraints for intermediate stages
        ocp.constraints.ubx = np.array(
            [
                100,  # x
                100,  # y
                100,  # z
                1.0,  # q[0]
                1.0,  # q[1]
                1.0,  # q[2]
                1.0,  # q[3]
                0.3,  # vx
                0.3,  # vy
                0.3,  # vz
                0.1,  # omega_x
                0.1,  # omega_y
                0.1,  # omega_z
                1000,  # theta
            ]
        )

        # Indexes of the state variables to which the constraints apply
        ocp.constraints.idxbx = np.arange(nx)

        # Slacks on lower/upper bounds
        ocp.constraints.lsbx = np.zeros(ocp.dims.nsbx)
        ocp.constraints.usbx = np.zeros(ocp.dims.nsbx)
        ocp.constraints.usbx[7:10] = 0.05
        ocp.constraints.usbx[7:10] = -0.05
        ocp.constraints.usbx[10:13] = 0.02
        ocp.constraints.usbx[10:13] = -0.02
        ocp.constraints.idxsbx = np.arange(nx)

        ocp.cost.Zl = 5e02 * np.ones(ocp.dims.ns)
        ocp.cost.Zu = 5e02 * np.ones(ocp.dims.ns)
        ocp.cost.zl = 5e03 * np.ones(ocp.dims.ns)
        ocp.cost.zu = 5e03 * np.ones(ocp.dims.ns)

        # Define input constraints
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

        # Set intial condition
        ocp.constraints.idxbx_0 = np.arange(nx)
        ocp.constraints.lbx_0 = ocp.constraints.lbx
        ocp.constraints.ubx_0 = ocp.constraints.ubx
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
        self.work_dir = os.path.dirname(os.path.abspath(__file__))
        self.save_dir = os.path.join(self.work_dir, "c_generated_code")
        ocp.code_export_directory = self.save_dir

        # Save ocp for further use
        self.ocp_init = ocp

        # Create integrator for nominal model
        self.nominal_sim = setup_sim_from_ocp(self.ocp_init)

    def _create_zoro_description(self, env: BaseEnv) -> None:
        """
        Creates a zoro description for the GP interface
        """
        # Uncertainty description
        Sigma_x0 = np.diag(
            [
                1.0e-6,
                1.0e-6,
                1.0e-6,
                1.0e-6,
                1.0e-6,
                1.0e-6,
                1.0e-6,
                1.0e-6,
                1.0e-6,
                1.0e-6,
                1.0e-6,
                1.0e-6,
                1.0e-6,
                1.0e-6,
            ]
        )

        Sigma_W = np.diag(
            [
                0.00001,
                0.00001,
                0.00001,
                0.00001,
                0.00001,
                0.00001,
            ]
        )

        unc_jac_G_mat = np.diag(
            [
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                0.0,
            ]
        )

        unc_jac_G_mat = unc_jac_G_mat[:, ~np.all(unc_jac_G_mat == 0, axis=0)]

        # create zoro_description
        zoro_description = ZoroDescription()
        zoro_description.unc_jac_G_mat = unc_jac_G_mat
        zoro_description.backoff_scaling_gamma = (
            0  # constraint tighenting (by how many sigma)
        )
        zoro_description.P0_mat = Sigma_x0  # uncertainty on initial state
        zoro_description.fdbk_K_mat = np.zeros(
            (self.ocp_init.dims.nu, self.ocp_init.dims.nx)
        )
        # zoro_description.unc_jac_G_mat = B
        """G in (nx, nw) describes how noise affects dynamics. I.e. x+ = ... + G@w"""
        zoro_description.W_mat = Sigma_W  # covariance of noise entering the system
        """W in (nw, nw) describes the covariance of the noise on the system"""
        zoro_description.input_P0_diag = True
        zoro_description.input_P0 = False
        zoro_description.input_W_diag = True
        zoro_description.input_W_add_diag = True
        zoro_description.output_P_matrices = False
        self.ocp_init.zoro_description = zoro_description

    def get_control_input(self, env) -> np.ndarray:
        """
        Calculate the control input based on current observation
        """
        # Retrieve some data which are used in multiple functions
        obs = env.get_obs()
        self.run_id = env.run_id
        self.timestamp = env.data.time

        # Check if new observation shall be added to dictionary
        res_output = self._compute_residual(obs)

        # If logging enabled, calculate the prediction errors
        # NOTE: Must be done here, before new observation is added
        # to the GP's dictionary
        self._calc_prediction_errors(env)

        if res_output is not None:
            residual, x_train = res_output
            residual = self.residual_scaler(residual)
            start_time = time.perf_counter()
            self.gp_mpc.residual_model.record_datapoint(
                x_input=x_train, y_target=residual, timestamp=env.data.time
            )
            end_time = time.perf_counter()

        # print(f"Total Update GP time: {(end_time-start_time)}")

        # Check solver status and re-initialize if needed
        if self.gp_mpc.ocp_solver.status != 0:
            print(f"Solution optimal, solver status: {self.gp_mpc.ocp_solver.status}")
            self._initialize_solver(env)

        # Set parameters
        self._set_params()

        # Warm start  solver
        for i in range(self.ctrl_cfg.N):
            i_next = i
            i_next = min(i_next + 1, self.N - 1)
            self.gp_mpc.ocp_solver.set(i, "x", self.last_solution["states"][i_next])
            self.gp_mpc.ocp_solver.set(i, "u", self.last_solution["inputs"][i_next])

        # Set initial condition
        xinit = np.append(env.get_obs(), self.theta_prev[1])
        self.gp_mpc.ocp_solver.set(0, "lbx", xinit)
        self.gp_mpc.ocp_solver.set(0, "ubx", xinit)

        # Solve for the first control input in receding horizon fashion
        self.gp_mpc.solve()
        self.X_res, U_res = self.gp_mpc.get_solution()
        u0 = U_res[0, :]

        # Visualize
        self._visualize_prediction()

        if hasattr(self, "renderer") and self.renderer is not None:
            _, _ = self.planner.get_reference(env.obs)
            self._visualize_prediction_renderer()

        # Save current solution
        for i in range(self.N):
            self.last_solution["states"][i] = self.gp_mpc.ocp_solver.get(i, "x")
            self.last_solution["inputs"][i] = self.gp_mpc.ocp_solver.get(i, "u")
        self.last_solution["states"][self.N] = self.gp_mpc.ocp_solver.get(self.N, "x")

        # Save current observation and input
        self.x_past[:-1], self.u_past = obs, u0

        # Save theta for next iteration
        for i in range(self.ctrl_cfg.N + 1):
            self.theta_prev[i] = self.gp_mpc.ocp_solver.get(i, "x")[-1]

        # Log quantities
        self._log(run_id=env.run_id, timestamp=env.data.time, env=env)

        # Return output
        # NOTE: Very important to copy input to not mess with reference stuff
        return u0[:-1].copy()

    def _set_params(self) -> None:
        """
        Sets the parameters of the solver at runtime
        """
        # Shift the previous solution for theta by one for reinitialization
        theta_shifted = self.theta_prev.copy()
        theta_shifted.append(theta_shifted[-1])
        theta_shifted.pop(0)

        # Set parameters
        for i in range(self.ctrl_cfg.N):

            theta_curr = theta_shifted[i]

            p_start = self.planner.trajectory._get_start_point_segment(theta_curr)
            t = self.planner.trajectory._get_tangent_segment(theta_curr)
            theta_1 = self.planner.trajectory._get_start_arc_length_segment(theta_curr)
            q_des = self.planner.trajectory.get_intermediate_reference(
                theta_curr
            ).attitude

            ref = np.concatenate((p_start, t, theta_1, q_des))

            self.gp_mpc.p_hat_nonlin[i, :] = ref.flatten()

    def _compute_residual(self, x_next: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Computes residual between actual state and expected state

        y_k = B_d^(-1) (x_{k+1} - f(x_k,u_k))
        """

        # Calculate the prediction/model error
        model_error = calc_model_error(
            obs=x_next,
            x_past=self.x_past[:-1],
            u_past=self.u_past[:-1],
            f_int=self.f_int,
        )

        # Ignore last row as theta not relevant
        residual = (self.B_d_inv[:, :-1] @ model_error).squeeze(-1)

        x_train = np.hstack((self.x_past, self.u_past))

        if self.has_logger:
            # Calculate the predicted residual
            with torch.no_grad(), gpytorch.settings.fast_pred_var():
                x_star = torch.atleast_2d(
                    self.input_selection(torch.from_numpy(x_train))
                )
                observed_pred = self.gp_mpc.residual_model.gp_model(x_star)

            # Scale residual
            observed_pred = self.residual_scaler(observed_pred)

            # Get mean and confidence intervals
            mean = observed_pred.mean
            lower, upper = observed_pred.confidence_region()

            try:
                n_points_GP = self.gp_mpc.residual_model.gp_model.train_inputs[0].shape[
                    -2
                ]
            except:
                n_points_GP = 0

            # Log quantities
            self.logger.log(
                run_id=self.run_id,
                timestamp=self.timestamp,
                gp_GT=residual.squeeze(0).cpu().numpy(),
                gp_pred=mean.squeeze(0).cpu().numpy(),
                gp_lower=lower.squeeze(0).cpu().numpy(),
                gp_upper=upper.squeeze(0).cpu().numpy(),
                n_points_GP=n_points_GP,
            )

            # Check if we want to log training data too
            if np.abs(self.timestamp % 10) < 0.1 and self.timestamp > 0.0:
                self.logger.log(
                    run_id=self.run_id,
                    timestamp=self.timestamp,
                    x_train=self.gp_mpc.residual_model.gp_model.train_inputs[0],
                    y_train=self.gp_mpc.residual_model.gp_model.train_targets,
                )

        return residual, x_train

    def _calc_prediction_errors(self, env: BaseEnv) -> None:
        """
        Calculates the prediction errors of both:
            - Nominal model
            - Nominal model + learned residual
        """
        if (
            self.has_logger
            and self.gp_mpc.residual_model.gp_model.train_inputs is not None
        ):
            # Retrieve obs (NOTE: Use GT obs here?)
            obs = env.get_obs()

            # Calculate nominal prediction error
            e_nom = calc_model_error(
                obs=obs,
                x_past=self.x_past[:-1],
                u_past=self.u_past[:-1],
                f_int=self.f_int,
            )

            # Calculate GP prediction error
            x_test = torch.from_numpy(np.hstack((self.x_past, self.u_past))).unsqueeze(
                0
            )
            with torch.no_grad(), gpytorch.settings.fast_pred_var():
                z_test = self.input_selection(x_test)
                predicted_residual = self.gp_mpc.residual_model.gp_model(z_test)
                predicted_residual = self.residual_scaler(predicted_residual)
            e_gp = e_nom - torch.matmul(self.B_d[:-1, :], predicted_residual.mean.T)

            # L2 norm of both
            e_gp = torch.norm(e_gp, 2).item()
            e_nom = torch.norm(e_nom, 2).item()

            # Log quantities
            if self.has_logger:
                self.logger.log(
                    run_id=env.run_id, timestamp=env.data.time, e_gp=e_gp, e_nom=e_nom
                )

    def _train_gp(self) -> None:
        """
        Trains the gp on the offline data
        """
        # Train GP on data offline
        self.gp_model, self.likelihood = train_gp_model(
            self.gp_model,
            torch_seed=456,
            training_iterations=300,
        )

    def _initialize_solver(self, env: BaseEnv) -> None:
        """
        Initializes the solver. Also known as "warm start".
        """
        # Retrieve closest point on track (relevant for theta)
        _, theta_init = self.planner.closest_point_on_trajectory(env.obs[0:3])

        # Array to store previous theta
        self.theta_prev = [theta_init for i in range(self.ctrl_cfg.N + 1)]

        # Warm start solver
        # Initial condition and Warm start
        x_guess = np.zeros(
            self.nx,
        )
        x_guess[0:13] = env.obs[0:13].copy()
        x_guess[-1] = theta_init

        [
            self.gp_mpc.ocp_solver.set(i, "x", x_guess)
            for i in range(self.ctrl_cfg.N + 1)
        ]
        [
            self.gp_mpc.ocp_solver.set(i, "u", np.zeros((self.nu, 1)))
            for i in range(self.ctrl_cfg.N)
        ]

        for i in range(self.N):
            self.last_solution["states"][i] = x_guess
            self.last_solution["inputs"][i] = np.zeros((self.nu))
        self.last_solution["states"][self.N] = x_guess

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

            # Track solve time
            # TODO: Where to get this from?
            solve_time = 0

            # Track cost value of current solution
            mpc_cost = self.cost_function(self.gp_mpc.ocp_solver.get(0, "x"),
                                          self.gp_mpc.ocp_solver.get(0, "u"),
                                          self.gp_mpc.p_hat_nonlin[0, 0:3].copy(),
                                          self.gp_mpc.p_hat_nonlin[0, 3:6].copy(),
                                          self.gp_mpc.p_hat_nonlin[0, 6].copy(),
                                          self.gp_mpc.p_hat_nonlin[0, 7:11].copy()).full()

            # Log quantities
            self.logger.log(
                run_id=run_id,
                timestamp=timestamp,
                tracking_error=tracking_error,
                attitude_error=attitude_error,
                solve_time=solve_time,
                mpc_cost=mpc_cost,
                u_demanded=self.u_past,  # This is the input commanded my the MPC at the current timestep
            )

    def _visualize_prediction(self) -> None:
        """
        Plot predicted trajectory of MPC in MuJoCo viewer.
        """
        for i in range(self.ctrl_cfg.N + 1):
            point = self.gp_mpc.ocp_solver.get(i, "x")[0:3]
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
                point = self.gp_mpc.ocp_solver.get(i, "x")[0:3]
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

            if hasattr(self, "logger"):
                self.logger.log(
                    run_id=self.run_id, timestamp=self.data.time, frames=sim_img
                )
