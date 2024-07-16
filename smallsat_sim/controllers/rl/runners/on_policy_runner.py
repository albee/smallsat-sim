import time
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam

from smallsat_sim.envs.vec_env import VecEnv
from smallsat_sim.controllers.rl.algorithms.vpg import VPGAgent
from smallsat_sim.controllers.rl.storage.vpg_buffer import VPGBuffer


class OnPolicyRunner(object):
    """
    On-policy runner for training and evaluation.
    """
    def __init__(self, env: VecEnv) -> None:
        # Initialize the environment and agent
        self.env = env
        self.agent = VPGAgent(self.env)

    def learn(self, steps_per_epoch, epochs, max_epoch_len, gamma, lam, actor_lr, critic_lr):
        """
        Main training loop.
        """
        # Set up buffer
        buffer = VPGBuffer([self.env.obs_dim], [self.env.act_dim], steps_per_epoch, gamma, lam)

        # Initialize ADAM optimizers for the actor and critic networks
        actor_optimizer = Adam(self.agent.actor.parameters(), lr=actor_lr)
        critic_optimizer = Adam(self.agent.critic.parameters(), lr=critic_lr)

        # Initialize the environment
        states, ep_ret, ep_len = self.agent.env.reset(), 0, 0

        # Main training loop
        for _ in range(epochs):
            ep_returns = []
            for t in range(steps_per_epoch):
                a, v, logp = self.agent.act(states)

                next_states, r, terminal = self.agent.env.transition(a)
                ep_ret += r
                ep_len += 1

                # Log transition
                buffer.store(states, a, r, v, logp)

                # Update state
                states = next_states

                # Check if a timeout is appropriate
                timeout = (ep_len == max_epoch_len)
                epoch_ended = (t == steps_per_epoch - 1)

                if terminal or timeout or epoch_ended:
                    # If the trajectory didn't reach terminal state, bootstrap value target
                    if epoch_ended:
                        _, v, _ = self.agent.act(states)
                    else:
                        v = 0
                    
                    if timeout or terminal:
                        ep_returns.append(ep_ret)
                    
                    buffer.end_traj(v)

                    states, ep_ret, ep_len = self.agent.reset(), 0, 0

            # Get the data from the training loop
            data = buffer.get()

            obs = data['obs']
            actions = data['actions']
            tdres = data['tdres']
            ret = data['ret']

            # Policy gradient update
            actor_optimizer.zero_grad() # Reset gradient
            _, logp_a = self.agent.actor.forward(obs, actions)
            loss = -torch.sum(tdres * logp_a)
            loss.backward()
            actor_optimizer.step()

            # Value function updates
            for _ in range(100):
                critic_optimizer.zero_grad() # Reset gradient
                values = self.agent.critic.forward(obs)
                loss = nn.functional.mse_loss(values, ret)
                loss.backward()
                critic_optimizer.step()

    def evaluate(self, episode_len, n_evals) -> None:
        """
        Evaluate the agent.
        """
        returns = []

        for _ in range(n_evals):
            states = self.env.transition[0]
            cum_returns = 0
            terminal = False
            self.env.reset()
            for t in range(episode_len):
                actions = self.agent.get_control_input(states)
                states, rewards, terminal = self.env.transition(actions)
                cum_returns += rewards
                if terminal:
                    break
                returns.append(cum_returns)

    def control(self) -> None:
        """
        Control the agent using the previously trained RL controller.
        """
        start_time = time.time()
        states = self.env.transition[0]
        terminal = False
        self.env.reset()
        while True:
            real_time = time.time() - start_time
            sim_time = self.env.data.time
            actions = self.agent.get_control_input(states)
            states, _, terminal = self.env.transition(actions)
            if terminal:
                break
