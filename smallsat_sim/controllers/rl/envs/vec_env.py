import numpy as np
import torch
import mujoco

from smallsat_sim.envs.base_env import BaseEnv


class VecEnv(BaseEnv):
    """
    Vectorized training environment for the smallsat.
    """
    def __init__(self) -> None:
        self.init_qpos = self.data.qpos # TODO: update to use MJX instead?
        self.init_qvel = self.data.qvel

        self.reset()

    def reset(self) -> None:
        """
        Reset the agent to the initial state in all the environment instances.
        """
        self.prev_shaping = None # TODO: adapt this to batched environments

        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:] = self.init_qpos
        self.data.qvel[:] = self.init_qvel
        mujoco.forward(self.model, self.data)
    
    def transition(self, actions: torch.tensor) -> tuple[torch.tensor, torch.tensor, torch.tensor]:
        """
        Apply input action on the environment. Returns the states, rewards and wether the terminal state has been reached.
        """
        self.step(input=ctrl_input) # TODO: think about what the main loop lokks like and decide how to handle this

        rewards = torch.zeros(1)
        shaping = torch.zeros(1) # TODO: implement reward shaping
        if self.prev_shaping is not None:
            rewards = shaping - self.prev_shaping
        self.prev_shaping = shaping

        terminal = torch.zeros(1, dtype=bool) # TODO: implement "game over" checking

        return states, rewards, terminal
