from smallsat_sim.controllers.base_controller import BaseController

import gpytorch
import numpy as np
import torch


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
        # Initialize mean module
        self.mean = gpytorch.means.ConstantMean()

        # Initialize covariance module
        self.covar = gpytorch.kernels.RBFKernel()

        # Initialize empty feature tensor (z)
        z = torch.empty((0, 6), dtype=torch.float32)

        # Initialize empty ouput tensor (y)
        y = torch.empty((0, 6), dtype=torch.float32)

        # Initialize empty timestamp tensor(t)
        t = []

        # Create dictionary
        self.dict = {"z": z, "y": y, "t": t}

        # Initialize subspace matrix B_d
        # Shape: (nx, 6)
        self.B_d = torch.cat((torch.zeros(7, 6), torch.eye(6))).to(torch.float64)
        self.B_d_inv = torch.linalg.pinv(self.B_d)

        # Initialize buffers to keep track of past state and input
        self.x_past = np.zeros(13)
        self.u_past = np.zeros(12)

        # Initialize some hyperparameters for GP
        # TODO: Move to configuration file
        self.M = 300  # number of points in list

    def get_control_input(self, env) -> np.ndarray:
        """
        Calculate the control input based on current observation
        """
        obs = env.get_obs()
        timestamp = env.data.time

        print(f"Timestamp: {timestamp}")

        # Check if GP needs to be updated
        self._update_gp(obs, timestamp)

        u = np.random.uniform(0, 0.3, (12,))
        # u = np.array([0.3,0.3])

        self.x_past, self.u_past = obs, u

        return u

    def _obs_to_features(self, obs) -> torch.Tensor:
        """
        Converts the environment observations to features used
        for Gaussian Process Regression

        [r,q,v,w] -> [v,w]
        """

        return torch.from_numpy(obs[7:]).unsqueeze(0)

    def _calc_outputs(self, obs) -> torch.Tensor:
        """
        Calculates the output data y which is necessary for training
        the GP appropriately.

        y_k = B_d^(-1) (x_{k+1} - f(x_k,u_k))
        """
        model_error = torch.from_numpy(
            obs - self.f_int(self.x_past, self.u_past).squeeze(-1)
        ).unsqueeze(-1)

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
            # Identify oldest element
            oldest_idx = self.dict["t"].index(min(self.dict["t"]))

            # Replace older data by new one
            self.dict["z"][oldest_idx, :] = z_k
            self.dict["y"][oldest_idx, :] = y_k
            self.dict["t"][oldest_idx] = timestamp

        else:
            # Append to current data
            # Replace older data by new one
            self.dict["z"] = torch.cat((self.dict["z"], z_k), dim=0)
            self.dict["y"] = torch.cat((self.dict["y"], y_k), dim=0)
            self.dict["t"].append(timestamp)
