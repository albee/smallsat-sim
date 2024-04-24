from smallsat_sim.envs.base_env import BaseEnv

import mujoco
import os


class CubesatEnv(BaseEnv):
    def __init__(self, args) -> None:
        # Load necessary config files
        self.env_cfg, self.model_cfg = self._load_cfg(
            env_name="cubesat", model_name="cubesat"
        )
        super().__init__(args=args)
