import os
import time
import jax.numpy as jnp
from flax import nnx

from smallsat_sim.envs.vec_env import VecEnv
from smallsat_sim.planners.base_planner import BasePlanner
from smallsat_sim.controllers.rl.algorithms.vpg import VPG
from smallsat_sim.controllers.rl.algorithms.ppo import PPO
from smallsat_sim.controllers.rl.runners.runner_utils import load_trained_modules
from smallsat_sim.utils.helpers_rl import standardize


class RLController(object):
    """
    RL controller that uses the saved actor weights.
    """

    def __init__(self, env: VecEnv, planner: BasePlanner) -> None:
        # Initialize the environment and agent
        self.env = env
        self.planner = planner
        self.agent = PPO(self.env, planner)
        self.reference_points = planner.reference_points
        self.tracking_point_idx = 0

        # Path to save the checkpoints
        self.ckpt_dir = "smallsat_sim/controllers/rl/checkpoints/"
        self.ckpt_filename = "pretraining_state_l1.pkl"

    def control(
        self,
    ) -> None:
        """
        Control the agent using the previously trained RL controller.
        """
        # Check if trained actor and critic modules are available and load them
        file_path = os.path.join(self.ckpt_dir, self.ckpt_filename)
        if os.path.isfile(file_path):
            restored_state = load_trained_modules(self.ckpt_dir, self.ckpt_filename)
            nnx.update(self.agent.actor.mu_net, restored_state["actor_model"].mu_net)
            nnx.update(self.agent.critic.v_net, restored_state["critic_model"].v_net)
        else:
            raise Exception("No training has been done yet.")

        start_time = time.time()
        states = self.env.get_states(self.reference_points[self.tracking_point_idx])
        terminal = jnp.zeros(self.env.num_envs, dtype=bool)
        self.env.reset()
        while True:
            real_time = time.time() - start_time
            sim_time = self.env.mjx_batch.time[0]
            self.planner._visualize_renderer(self.reference_points)
            actions = self.agent.get_control_input(states)
            # actions = self.agent.actor.mu_net(states) # To test the pretrained network
            states = self.env.get_states(self.reference_points[self.tracking_point_idx])
            _, terminal = self.env.transition(actions, states)
            if terminal.all():
                if self.tracking_point_idx == (len(self.reference_points) - 1):
                    break
                elif self.tracking_point_idx == 10:
                    break
                else:
                    self.tracking_point_idx += 1
