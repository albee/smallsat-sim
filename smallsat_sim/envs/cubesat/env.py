from smallsat_sim.envs.base_env import BaseEnv
from smallsat_sim import SMALLSAT_SIM_LIB_DIR

import mujoco
import os

class CubesatEnv(BaseEnv):
    def __init__(self,args) -> None:
        # Load necessary config files
        self.env_cfg, self.lib_cfg = self._load_cfg(env_name="cubesat",
                                                    lib_name="cubesat")
        super().__init__(args=args)