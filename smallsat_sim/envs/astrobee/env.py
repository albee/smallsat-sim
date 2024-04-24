from smallsat_sim.envs.base_env import BaseEnv
from smallsat_sim import SMALLSAT_SIM_MODEL_DIR

import mujoco
import os


class AstrobeeEnv(BaseEnv):
    def __init__(self, args) -> None:
        # Load necessary config files
        self.env_cfg, self.model_cfg = self._load_cfg(
            env_name="astrobee", model_name="astrobee"
        )
        super().__init__(args=args)
