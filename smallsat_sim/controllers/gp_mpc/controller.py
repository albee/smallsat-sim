from smallsat_sim.controllers.base_controller import BaseController

import gpytorch
import numpy as np
import torch


class GP(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(GP, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean(batch_shape=torch.Size([6]))
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(batch_shape=torch.Size([6]), ard_num_dims=18),
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

    def _setup_gp(self):
        """
        Initializes all needed quantaties for the Gaussian Process
        """
        # Create the GP
        self._create_gp()

        # Initialize empty feature tensor (z)
        z = torch.empty((0, 18), dtype=torch.float32)

        # Initialize empty ouput tensor (y)
        y = torch.empty((0, 6), dtype=torch.float32)

        # Initialize empty timestamp tensor(t)
        t = []

        # Create dictionary
        self.dict = {"z": z, "y": y, "t": t}

        # Initialize subspace matrix B_d
        # Shape: (nx, 6)
        self.B_d = torch.cat((torch.zeros(7, 6), torch.eye(6))).to(torch.float32)
        self.B_d_inv = torch.linalg.pinv(self.B_d)

        # Initialize buffers to keep track of past state and input
        self.x_past = np.zeros(13)
        self.u_past = np.zeros(12)

        # Initialize some hyperparameters for GP
        # TODO: Move to configuration file
        self.M = 300  # number of points in list
        self.gp_update_counter = 0  # Keep track how many times dict has been updated
        self.gp_initialized = False  # Keep track if GP is already initialized

    def _create_gp(self) -> None:
        # Initialize mean module
        self.mean = gpytorch.means.ConstantMean()

        # Initialize covariance module
        self.covar = gpytorch.kernels.RBFKernel()

    def get_control_input(self, env) -> np.ndarray:
        """
        Calculate the control input based on current observation
        """
        obs = env.get_obs()
        timestamp = env.data.time

        print(f"Timestamp: {timestamp}")

        # Check if GP needs to be updated
        self._update_gp(obs, timestamp)

        # Record error statistics
        self._record_stats(obs, timestamp)

        u = np.random.uniform(0, 0.3, (12,))

        self.x_past, self.u_past = obs, u

        return u

    def _obs_to_features(self, obs) -> torch.Tensor:
        """
        Converts the environment observations to features used
        for Gaussian Process Regression

        [r,q,v,w] -> [v,w]
        """

        return torch.from_numpy(
            np.concatenate((self.x_past[7:], self.u_past))
        ).unsqueeze(0).to(torch.float32)

    def _calc_outputs(self, obs) -> torch.Tensor:
        """
        Calculates the output data y which is necessary for training
        the GP appropriately.

        y_k = B_d^(-1) (x_{k+1} - f(x_k,u_k))
        """
        model_error = (
            torch.from_numpy(obs - self.f_int(self.x_past, self.u_past).squeeze(-1))
            .to(torch.float32)
            .unsqueeze(-1)
        )

        return (self.B_d_inv @ model_error).T

    def _add_training_point(self, env) -> None:
        """
        Add a feature and its corresponding output to the list of
        points used for the GP.
        """

        pass

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
            print("Iter %d/%d - Loss: %.3f" % (i + 1, i, loss.item()))
            optimizer.step()

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

            print(f"Nominal model error: {e_nom}")
            print(f"Corrected model error: {e_gp}")
