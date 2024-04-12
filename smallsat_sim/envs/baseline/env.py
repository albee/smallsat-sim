from smallsat_sim.envs.base_env import BaseEnv
from smallsat_sim.envs.disturbances import ConstantForceDisturbance

import numpy as np

class BaselineEnv(BaseEnv):
    def __init__(self, args) -> None:
        # Load necessary config files
        self.env_cfg, self.lib_cfg = self._load_cfg(env_name="baseline",
                                                    lib_name="baseline")

        # Initialize parent class
        super().__init__(args)

        # Initialize Disturbance
        self.disturbance = ConstantForceDisturbance(magnitude=0.1,
                                                    direction=np.array([0,1,0]))
