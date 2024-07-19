from smallsat_sim.envs.base_env import BaseEnv
from smallsat_sim.envs.disturbances import ConstantForceDisturbance

import numpy as np


class BaselineEnv(BaseEnv):
    def __init__(self, args) -> None:
        # Load necessary config files
        self.env_cfg, self.model_cfg = self._load_cfg(
            env_name="baseline", model_name="baseline"
        )

        # Initialize parent class
        super().__init__(args)

        # Initialize Disturbance
        self.disturbances = ConstantForceDisturbance(
            magnitude=0.1, direction=np.array([0, 1, 0])
        )
