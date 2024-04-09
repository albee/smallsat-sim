from smallsat_sim.envs.base_env import BaseEnv
from smallsat_sim.envs.disturbances import ConstantForceDisturbance

import numpy as np

class BaselineEnv(BaseEnv):
    def __init__(self, args) -> None:
        self.cfg = self._load_cfg("baseline")
        super().__init__(args)

        # Initialize Disturbance
        self.disturbance = ConstantForceDisturbance(magnitude=0.01,
                                                    direction=np.array([0,1,0]))