from smallsat_sim.envs.base_env import BaseEnv

import mujoco
import os

class BaselineEnv(BaseEnv):
    def __init__(self, args) -> None:
        self.cfg = self._load_cfg("baseline")
        super().__init__(args)