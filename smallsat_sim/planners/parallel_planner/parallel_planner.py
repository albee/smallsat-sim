from smallsat_sim.planners.base_planner import BasePlanner

import numpy as np
import jax
import jax.numpy as jnp

class ParallelPlanner(BasePlanner):
    def __init__(self, env) -> None:
        super().__init__(env)

    def get_reference(self, obs) :
        """
        Returns a reference point based on current observations
        """
        return jnp.zeros((4096,3))