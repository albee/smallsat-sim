from smallsat_sim.controllers.base_controller import BaseController
from smallsat_sim.envs.vec_env import VecEnv
from smallsat_sim.utils.helpers import quat_multiply, quat_conjugate, Rquat, sgn_quat

import numpy as np

import jax


class DummyRL(BaseController):
    def __init__(self, env, planner) -> None:
        # Fetch correct controller config
        ctrl_cfg = env.env_cfg.control.DummyRL

        # Initialize base class
        super().__init__(env, planner, ctrl_cfg)

    def get_control_input(self, env: VecEnv):
        """Defines the controller callback for the simulation step."""
        
        return jax.random.uniform(jax.random.PRNGKey(0), (env.n_envs, 12), minval=0.0, maxval=0.3)
