# Base classes
from smallsat_sim.controllers.base_controller import BaseController
from smallsat_sim.envs.base_env import BaseEnv

# General libraries
import gpytorch
import os
import numpy as np
import torch
import casadi as ca
import mujoco

# Acados
from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver
from casadi import SX

# Zero Order GPMPC
import smallsat_sim.controllers.gp_mpc.external.zero_order_gpmpc as zero_order_gpmpc
from smallsat_sim.controllers.gp_mpc.external.zero_order_gpmpc.controllers import ZeroOrderGPMPC
from smallsat_sim.controllers.gp_mpc.external.zero_order_gpmpc.controllers.zoro_acados_utils import setup_sim_from_ocp

# gpytorch utilities
from smallsat_sim.controllers.gp_mpc.external.gpytorch_utils.gp_hyperparam_training import (
    generate_train_inputs_acados,
    generate_train_outputs_at_inputs,
    train_gp_model,
)
from smallsat_sim.controllers.gp_mpc.external.gpytorch_utils.gp_utils import (
    gp_data_from_model_and_path,
    gp_derivative_data_from_model_and_path,
    plot_gp_data,
    generate_grid_points,
)

# GPyTorch models
from smallsat_sim.controllers.gp_mpc.external.zero_order_gpmpc.models.gpytorch_models.gpytorch_gp import (
    BatchIndependentMultitaskGPModel
)


# Set default torch dtype
torch.set_default_dtype(torch.float64)


class GP(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(GP, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean(batch_shape=torch.Size([6]))
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(batch_shape=torch.Size([6])),
            batch_shape=torch.Size([6]),
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultitaskMultivariateNormal.from_batch_mvn(
            gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
        )


class GPMPC(BaseController):
    """
    This class implements a GP MPC controller based on GPyTorch and acados.

    See the following documentations for details:
    GPyTorch: https://docs.gpytorch.ai/en/stable/
    acados: https://docs.acados.org/
    """

    def __init__(self, env, planner) -> None:
        # Fetch correct controller config
        self.ctrl_cfg = env.env_cfg.control.GPMPC

        # Initialize base class
        super().__init__(env, planner, self.ctrl_cfg)

        # Save relevant components from environment locally
        self.f = env.symbolic_model.f_expl_expr_func
        self.f_int = env.symbolic_model.get_integrator(dt=self.ctrl_cfg.Ts)

        # Setup Gaussian Process
        self._setup_gp()

        # Generate solver
        self._generate_solver(env)

        # Initialize solver
        self._initialize_solver(env)

        # Check if there is a viewer. In case there is not,
        # dynamically allocate the visualize method to a lambda
        # function doing nothing.
        if env.viewer:
            self.viewer = env.viewer
        else:
            self._visualize = lambda *args, **kwargs: None


    def _setup_gp(self):
        """
        Initializes all needed quantaties for the Gaussian Process
        """
        # Initialize empty feature tensor (z)
        z = torch.empty((0, 18), dtype=torch.float64)

        # Initialize empty ouput tensor (y)
        y = torch.empty((0, 6), dtype=torch.float64)

        # Initialize empty timestamp tensor(t)
        t = []

        # Create dictionary
        self.dict = {"z": z, "y": y, "t": t}

        # Initialize subspace matrix B_d
        # Shape: (nx, 6)
        self.B_d = torch.cat((torch.zeros(7, 6), torch.eye(6))).to(torch.float64)
        self.B_d_inv = torch.linalg.pinv(self.B_d).to(torch.float64)

        # Initialize buffers to keep track of past state and input
        self.x_past = np.zeros(13)
        self.u_past = np.zeros(12)

        # Initialize some hyperparameters for GP
        # TODO: Move to configuration file
        self.M = 300  # number of points in list
        self.gp_update_counter = 0  # Keep track how many times dict has been updated
        self.gp_initialized = False  # Keep track if GP is already initialized

    def _generate_solver(self, env) -> None:
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

        # Define artificial reference points which are opt. variables
        x_a = ca.SX.sym('x_a', 13, 1)
        #u_a = ca.SX.sym('u_a', 12, 1)

        x_a_dot = ca.SX.sym('x_a_dot', 13, 1)
        #u_a_dot = ca.SX.sym('u_a_dot', 12, 1)

        acados_model.x = ca.vertcat(acados_model.x,
                                    x_a)
        
        acados_model.f_expl_expr = ca.vertcat(acados_model.f_expl_expr,
                                              ca.SX.zeros(13, 1))
        
        acados_model.f_impl_expr = ca.vertcat(acados_model.f_impl_expr,
                                              x_a_dot)
        
        acados_model.con_h_expr_e = model.x - x_a

        # Assign parameters and model
        Ts = self.ctrl_cfg.Ts
        p = ca.vertcat(
            SX.sym("x_r"),
            SX.sym("y_r"),
            SX.sym("z_r"),
            SX.sym("eta_r"),
            SX.sym("eps1_r"),
            SX.sym("eps2_r"),
            SX.sym("eps3_r"),
            SX.sym("vx_r"),
            SX.sym("vy_r"),
            SX.sym("vz_r"),
            SX.sym("omega_x_r"),
            SX.sym("omega_y_r"),
            SX.sym("omega_z_r"),
        )
        acados_model.p = p
        ocp.model = acados_model

        # Define and assign cost functions
        Q = self.ctrl_cfg.cost.Q
        R = self.ctrl_cfg.cost.R
        T = self.ctrl_cfg.cost.T
        ocp.cost.cost_type = "EXTERNAL"
        ocp.model.cost_expr_ext_cost = (model.u.T) @ R @ (model.u) + (model.x.T-x_a.T) @ Q @ (model.x-x_a) + (x_a - p).T @ T @ (x_a - p)

        # Set OCP dimensions
        nx = acados_model.x.size()[0]  # number of states
        nu = acados_model.u.size()[0]  # number of inputs
        ocp.dims.nx = nx
        ocp.dims.nu = nu
        ocp.dims.np = p.size()[0]  # number of parameters
        ocp.dims.N = self.ctrl_cfg.N  # prediction horizon length
        ocp.dims.nh_e  = acados_model.con_h_expr_e.size()[0]

        # Define state constraints
        # Lower and Upper bound constraints for intermediate stages
        ocp.constraints.lbx = np.array(
            [-100, -100, -100, -1.0, -1, -1, -1, -1, -1, -1, -0.5, -0.5, -0.5, -100, -100, -100, -1.0, -1, -1, -1, 0, 0, 0, 0, 0, 0]
        )
        ocp.constraints.ubx = np.array(
            [100, 100, 100, 1.0, 1, 1, 1, 1, 1, 1, 0.5, 0.5, 0.5, 100, 100, 100, 1.0, 1, 1, 1, 0, 0, 0, 0, 0, 0]
        )
        ocp.constraints.idxbx = np.arange(nx)

        ocp.constraints.lh_e = np.zeros((13,1))
        ocp.constraints.uh_e = np.zeros((13,1))

        # Define input constraints
        # Fetch thurster limits from the model configuration
        thruster_forces = [
            thruster.forcerange for thruster in env.model_cfg.Thrusters.thruster_list
        ]
        ocp.constraints.lbu = np.array([forces[0] for forces in thruster_forces])
        ocp.constraints.ubu = np.array([forces[1] for forces in thruster_forces])
        ocp.constraints.idxbu = np.arange(nu)

        # Set intial condition
        ocp.constraints.idxbx_0 = np.arange(13)
        ocp.constraints.lbx_0 = env.obs[0:13]
        ocp.constraints.ubx_0 = env.obs[0:13]
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


    def get_control_input(self, env) -> np.ndarray:
        """
        Calculate the control input based on current observation
        """
        obs = env.get_obs()
        timestamp = env.data.time

        # Check if GP needs to be updated
        self._update_gp(obs, timestamp)

        # Record error statistics
        self._record_stats(obs, timestamp)

        # Check solver status and re-initialize if needed
        if self.ocp_solver.status != 0:
            print(f"Solution optimal, solver status: {self.ocp_solver.status}")
            self._initialize_solver(env)

        # Set the reference position
        ref_pos = self.planner.get_reference(env.obs).reshape(3, 1)
        ref_quat = np.array([1,0,0,0]).reshape(4,1)
        ref_vel = np.zeros((3,1))
        ref_omega = np.zeros((3,1))
        ref = np.concatenate((ref_pos, ref_quat, ref_vel, ref_omega))

        [self.ocp_solver.set(i, "p", ref) for i in range(self.ctrl_cfg.N + 1)]

        # Solve for the first control input in receding horizon fashion
        u0 = self.ocp_solver.solve_for_x0(env.obs[0:13], print_stats_on_failure=True)
        self._visualize()

        # Save current observation and input
        self.x_past, self.u_past = obs, u0

        return u0

    def _obs_to_features(self, obs) -> torch.Tensor:
        """
        Converts the environment observations to features used
        for Gaussian Process Regression

        [r,q,v,w] -> [v,w]
        """

        return (
            torch.from_numpy(np.concatenate((self.x_past[7:], self.u_past)))
            .unsqueeze(0)
            .to(torch.float64)
        )

    def _calc_outputs(self, obs) -> torch.Tensor:
        """
        Calculates the output data y which is necessary for training
        the GP appropriately.

        y_k = B_d^(-1) (x_{k+1} - f(x_k,u_k))
        """
        model_error = (
            torch.from_numpy(obs - self.f_int(self.x_past, self.u_past).squeeze(-1))
            .to(torch.float64)
            .unsqueeze(-1)
        )

        return (self.B_d_inv @ model_error).T

    def _update_gp(self, obs, timestamp) -> None:
        """
        Main method which checks if the GP needs to be updated
        with a new point.
        """

        # Check if new data shall be added to dictionary
        # For now, always added

        if timestamp == 0:
            return

        # Calculate features
        z_k = self._obs_to_features(obs)

        # Calculate the output y
        y_k = self._calc_outputs(obs)

        if len(self.dict["t"]) == self.M:
            # Train GP first time dict is full
            if not self.gp_initialized:
                self.likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(
                    num_tasks=6
                )
                self.gp = GP(
                    train_x=self.dict["z"],
                    train_y=self.dict["y"],
                    likelihood=self.likelihood,
                )
                self._train_gp(self.gp)
                self.gp_initialized = True

            # Identify oldest element
            oldest_idx = self.dict["t"].index(min(self.dict["t"]))

            # Replace older data by new one
            self.dict["z"][oldest_idx, :] = z_k
            self.dict["y"][oldest_idx, :] = y_k
            self.dict["t"][oldest_idx] = timestamp

            # Replace data in GP

            self.gp.set_train_data(inputs=self.dict["z"], targets=self.dict["y"])

            self.gp_update_counter += 1

            # Retrain hyperparameter every 100 updates
            if self.gp_update_counter == 100:
                self._train_gp(self.gp)
                self.gp_update_counter = 0

        else:
            # Append to current data
            # Replace older data by new one
            self.dict["z"] = torch.cat((self.dict["z"], z_k), dim=0)
            self.dict["y"] = torch.cat((self.dict["y"], y_k), dim=0)
            self.dict["t"].append(timestamp)

        # Update Gaussian Process

    def _train_gp(self, gp: GP) -> None:
        # Set opimizer
        optimizer = torch.optim.Adam(
            gp.parameters(), lr=0.1
        )  # Includes GaussianLikelihood parameters
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, gp)

        self.gp.train()
        self.likelihood.train()

        training_steps = 100
        for i in range(training_steps):
            # Zero gradients from previous iteration
            optimizer.zero_grad(training_steps)
            # Output from model
            output = gp(self.dict["z"])
            # Calc loss and backprop gradients
            loss = -mll(output, self.dict["y"])
            loss.backward()
            print("Iter %d/%d - Loss: %.3f" % (i + 1, training_steps, loss.item()))
            optimizer.step()

    def _initialize_solver(self, env: BaseEnv) -> np.ndarray:
        """
        Initializes the solver. Also known as "warm start".
        """
        xinit = env.obs[0:13]
        x_guess = np.concatenate((xinit, xinit))

        [self.ocp_solver.set(i, "x", x_guess) for i in range(self.ctrl_cfg.N + 1)]
        [self.ocp_solver.set(i, "u", np.zeros((12, 1))) for i in range(self.ctrl_cfg.N)]

    def _visualize(self) -> None:
        """
        Plot predicted trajectory of MPC in MuJoCo viewer.
        """
        offset = self.viewer.user_scn.ngeom
        for i in range(self.ctrl_cfg.N + 1):
            point = self.ocp_solver.get(i, "x")[0:3]
            mujoco.mjv_initGeom(
                self.viewer.user_scn.geoms[i + offset],
                type=mujoco.mjtGeom.mjGEOM_SPHERE,
                size=[0.05, 0, 0],
                pos=point,
                mat=np.eye(3).flatten(),
                rgba=np.array([0, 0, 1, 2]),
            )

        self.viewer.user_scn.ngeom += self.ctrl_cfg.N + 1
    
    def _record_stats(self, obs, timestamp) -> None:
        """
        Record error statistics of the GP
        Can be used for plotting and printing
        """
        if self.gp_initialized:
            # Calculate nominal model error
            e_nom = np.linalg.norm(
                obs - self.f_int(self.x_past, self.u_past).squeeze(-1)
            )

            # Calculate GP error
            self.gp.eval()
            self.likelihood.eval()

            with torch.no_grad(), gpytorch.settings.fast_pred_var():
                features = self._obs_to_features(obs)
                prediction = self.likelihood(self.gp(features))
                mean = prediction.mean

            e_gp = np.linalg.norm(
                (
                    obs
                    - self.f_int(self.x_past, self.u_past).squeeze(-1)
                    - torch.matmul(
                        self.B_d,
                        mean.T,
                    )
                    .squeeze(-1)
                    .detach()
                    .numpy()
                )
            )

            if self.gp_update_counter == 50:
                print(f"Nominal model error: {e_nom}")
                print(f"Corrected model error: {e_gp}")

                # Print the different hyperparameters
                print("Mean Module Hyperparameters:")
                for name, param in self.gp.mean_module.named_parameters():
                    print(f"{name}: {param}")

                print("\nCovariance Module Hyperparameters:")
                for name, param in self.gp.covar_module.named_parameters():
                    print(f"{name}: {param}")

                print("\nLikelihood Hyperparameters:")
                for name, param in self.gp.likelihood.named_parameters():
                    print(f"{name}: {param}")
