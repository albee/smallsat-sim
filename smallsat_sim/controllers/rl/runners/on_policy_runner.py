import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam

from smallsat_sim.controllers.rl.envs.vec_env import VecEnv
from smallsat_sim.controllers.rl.algorithms.agent import RLAgent
from smallsat_sim.controllers.rl.storage.vpg_buffer import VPGBuffer


class OnPolicyRunner(object):
    """
    On-policy runner for training and evaluation.
    """
    def __init__(self, env: VecEnv) -> None:
        self.env = env

    def learn(self):
        """
        Main training loop.
        """

        # Initialize agent
        agent = RLAgent(self.env)

        # Training params
        steps_per_epoch = 3000
        epochs = 50
        max_epoch_len = 300
        gamma = 0.99
        lam = 0.97
        actor_lr = 3e-3
        critic_lr = 1e-3

        # Set up buffer
        buffer = VPGBuffer()

        # Initialize ADAM optimizers for the actor and critic networks
        actor_optimizer = Adam(agent.actor.parameters(), lr=actor_lr)
        critic_optimizer = Adam(agent.critic.parameters(), lr=critic_lr)

        # Initialize the environment
        states, ep_ret, ep_len = agent.env.reset(), 0, 0

        # Main training loop
        for epoch in range(epochs):
            ep_returns = []
            for t in range(steps_per_epoch):
                a, v, logp = agent.act(states)

                next_states, r, terminal = agent.env.transition(a)
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
                    # If the trajectory didn't reach terminal state, bootrsp value target
                    if epoch_ended:
                        _, v, _ = agent.act(states)
                    else:
                        v = 0
                    
                    if timeout or terminal:
                        ep_returns.append(ep_ret)
                    
                    buffer.end_traj(v)

                    states, ep_ret, ep_len = agent.reset(), 0, 0

            # Get the data from the training loop
            data = buffer.get()

            obs = data['obs']
            actions = data['actions']
            tdres = data['tdres']
            ret = data['ret']

            # Policy gradient update
            actor_optimizer.zero_grad() # Reset gradient
            _, logp_a = agent.actor.forward(obs, actions)
            loss = -torch.sum(tdres * logp_a)
            loss.backward()
            actor_optimizer.step()

            # Value function updates
            for _ in range(100):
                critic_optimizer.zero_grad() # Reset gradient
                values = agent.critic.forward(obs)
                loss = nn.functional.mse_loss(values, ret)
                loss.backward()
                critic_optimizer.step()

        return agent

    
    def evaluate(self) -> None:
        """
        Evaluate the agent.
        """
        agent = self.learn()

        episode_len = 300
        n_evals = 100
        returns = []

        for i in range(n_evals):
            states = self.env.transition[0]
            cum_returns = 0
            terminal = False
            self.env.reset()
            for t in range(episode_len):
                actions = agent.get_control_input(states)
                states, rewards, terminal = self.env.transition(actions)
                cum_returns += rewards
                if terminal:
                    break
                returns.append(cum_returns)
