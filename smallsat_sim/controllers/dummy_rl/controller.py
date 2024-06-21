# (License for _apply_ctrl_constraint function)
# Copyright (c) 2017, United States Government, as represented by the
# Administrator of the National Aeronautics and Space Administration.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations
# under the License.

from smallsat_sim.controllers.base_controller import BaseController
from smallsat_sim.envs.base_env import BaseEnv
from smallsat_sim.utils.helpers import quat_multiply, quat_conjugate, Rquat, sgn_quat

import numpy as np

import jax


class DummyRL(BaseController):
    def __init__(self, env, planner) -> None:
        # Fetch correct controller config
        ctrl_cfg = env.env_cfg.control.DummyRL

        # Initialize base class
        super().__init__(env, planner, ctrl_cfg)

        

    def get_control_input(self, env: BaseEnv):
        """Defines the controller callback for the simulation step."""
        
        return jax.random.uniform(jax.random.PRNGKey(0), (4096, 12), minval=0.0, maxval=0.3)
