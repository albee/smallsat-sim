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
        self.z = torch.empty((0,22), dtype=torch.float32)

        # Initialize empty ouput tensor (y)
        self.y = torch.empty((0,13), dtype=torch.float32)

    def get_control_input(self, env) -> np.ndarray:
        """
        Calculate the control input based on current observation
        """
        
        return np.random.uniform(0,0.6,(12,))
    
    def _obs_to_features(self, env) -> torch.Tensor:
        """
        Converts the environment observations to features used
        for Gaussian Process Regression
        """

        pass

    def _calc_outputs(self, env) -> torch.Tensor:
        """
        Calculates the output data y which is necessary for training
        the GP appropriately.

        y_k = B_d^(-1) (x_{k+1} - f(x_k,u_k))
        """

        pass

    def _add_training_point(self, env) -> None:
        """
        Add a feature and its corresponding output to the list of 
        points used for the GP.
        """

        pass