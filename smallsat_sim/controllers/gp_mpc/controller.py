from smallsat_sim.controllers.base_controller import BaseController

import gpytorch
import numpy as np


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

    def get_control_input(self, env) -> np.ndarray:
        """
        Calculate the control input based on current observation
        """
        
        return np.random.uniform(0,0.6,(12,))