import time
import jax.numpy as jnp
from flax import nnx
import optax

from smallsat_sim.envs.vec_env import VecEnv
from smallsat_sim.controllers.rl.algorithms.vpg import VPGAgent
from smallsat_sim.controllers.rl.storage.vpg_buffer import VPGBuffer


class OnPolicyRunner(object):
    """
    On-policy runner for training and evaluation.
    """
    def __init__(self, env: VecEnv, planner) -> None:
        # Initialize the environment and agent
        self.env = env
        self.agent = VPGAgent(self.env, planner)

    def learn(self, steps_per_epoch, epochs, max_epoch_len, gamma, lam, actor_lr, critic_lr):
        """
        Main training loop.
        """
        # Define the actor and critic loss
        @nnx.jit
        def actor_loss_fn(tdres: jnp.ndarray):
            _, logp_a = self.agent.actor.forward(obs, actions)
            return -jnp.sum(tdres * logp_a)
        
        @nnx.jit
        def critic_loss_fn(returns: jnp.ndarray):
            values = self.agent.critic.forward(obs)
            return jnp.mean((values - returns) ** 2) # MSE loss

        # Set up buffer
        buffer = VPGBuffer(self.env.n_envs, self.env.obs_dim, self.env.act_dim, steps_per_epoch, gamma, lam)

        # Initialize ADAM optimizers for the actor and critic networks
        actor_optimizer = nnx.Optimizer(self.agent.actor, optax.adam(learning_rate=actor_lr))
        critic_optimizer = nnx.Optimizer(self.agent.critic, optax.adam(learning_rate=critic_lr))

        # Initialize the environment
        states, ep_ret, ep_len = self.env.get_obs(), jnp.zeros(self.env.n_envs), 0

        # Main training loop
        for _ in range(epochs):
            ep_returns = jnp.zeros((self.env.n_envs, steps_per_epoch))
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

                if terminal.any() or timeout or epoch_ended:
                    # If the trajectory didn't reach terminal state, bootstrap value target
                    if epoch_ended:
                        _, v, _ = self.agent.act(states)
                    else:
                        v = jnp.zeros(self.env.n_envs)
                    
                    if timeout:
                        ep_returns[:, t] = ep_ret

                    if terminal.any():
                        true_indices = jnp.nonzero(terminal).squeeze()
                        for idx in true_indices:
                            ep_returns[idx, t] = ep_ret[idx]
                    
                    buffer.end_traj(v)

                    states, ep_ret, ep_len = self.env.get_obs(), jnp.zeros(self.env.n_envs), 0

            # Get the data from the training loop
            data = buffer.get()

            obs = data['obs']
            actions = data['act']
            tdres = data['tdres']
            returns = data['ret']

            # Policy gradient update
            # actor_optimizer.zero_grad() # Reset gradient # TODO
            loss, grads = nnx.value_and_grad(actor_loss_fn(tdres))(self.agent.actor)
            print(f'{loss = }')
            actor_optimizer.update(grads)
            # actor_optimizer.step() # TODO

            # Value function updates
            for _ in range(100):
                # critic_optimizer.zero_grad() # Reset gradient # TODO
                loss, grads = nnx.value_and_grad(critic_loss_fn(returns))(self.agent.critic)
                print(f'{loss = }')
                critic_optimizer.update(grads)
                # loss.backward() # TODO
                # critic_optimizer.step()

    def evaluate(self, episode_len, n_evals) -> None:
        """
        Evaluate the agent.
        """
        returns = []

        for _ in range(n_evals):
            states = self.env.get_obs()
            cum_returns = jnp.zeros(self.env.n_envs)
            terminal = jnp.zeros(self.env.n_envs, dtype=bool)
            self.env.reset()
            for _ in range(episode_len):
                actions = self.agent.get_control_input(states)
                states, rewards, terminal = self.env.transition(actions)
                cum_returns += rewards
                if terminal.all(): # TODO: abort environments that failed
                    break
                returns.append(cum_returns)

    def control(self) -> None:
        """
        Control the agent using the previously trained RL controller.
        """
        start_time = time.time()
        states = self.env.get_obs()
        terminal = jnp.zeros(self.env.n_envs, dtype=bool)
        self.env.reset()
        while True:
            real_time = time.time() - start_time
            sim_time = self.env.data.time
            actions = self.agent.get_control_input(states)
            states, _, terminal = self.env.transition(actions)
            if terminal.all():
                break
