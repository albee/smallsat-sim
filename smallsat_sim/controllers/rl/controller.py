import os
import time
import jax.numpy as jnp
from flax import nnx

from smallsat_sim.envs.vec_env import VecEnv
from smallsat_sim.planners.base_planner import BasePlanner
from smallsat_sim.controllers.rl.algorithms.vpg import VPG
from smallsat_sim.controllers.rl.algorithms.ppo import PPO
from smallsat_sim.controllers.rl.runners.runner_utils import load_trained_modules


class RLController(object):
    """
    RL controller that uses the save3d actor weights.
    """

    def __init__(self, env: VecEnv, planner: BasePlanner) -> None:
        # Initialize the environment and agent
        self.env = env
        self.agent = PPO(self.env, planner)
        self.reference_point = planner.reference_points[0]

        # Path to save the checkpoints
        self.ckpt_path = "smallsat_sim/controllers/rl/checkpoints/"
        self.ckpt_filename = "100_training_state.pkl"

    def control(
        self,
    ) -> None:
        """
        Control the agent using the previously trained RL controller.
        """
        # Check if trained actor and critic modules are available and load them
        ckpt_dir = os.listdir(self.ckpt_path)
        if len(ckpt_dir) == 0:
            raise Exception("No training has been done yet.")
        else:
            restored_state = load_trained_modules(self.ckpt_path, self.ckpt_filename)
            nnx.update(self.agent.actor.mu_net, restored_state["actor_model"].mu_net)
            nnx.update(self.agent.critic.v_net, restored_state["critic_model"].v_net)

        start_time = time.time()
        states = self.env.get_states(self.reference_point)
        terminal = jnp.zeros(self.env.num_envs, dtype=bool)
        self.env.reset()
        while True:
            real_time = time.time() - start_time
            sim_time = self.env.mjx_batch.time[0]
            actions = self.agent.get_control_input(states)
            states = self.env.get_states(self.reference_point)
            _, terminal = self.env.transition(actions, states)
            if terminal.all():
                break
