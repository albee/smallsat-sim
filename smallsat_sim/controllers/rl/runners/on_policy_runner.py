import time
import jax.numpy as jnp
from flax import nnx
import optax

from smallsat_sim.envs.vec_env import VecEnv
from smallsat_sim.planners.base_planner import BasePlanner
from smallsat_sim.controllers.rl.algorithms.vpg import VPGAgent
from smallsat_sim.controllers.rl.storage.vpg_buffer import VPGBuffer


class OnPolicyRunner(object):
    """
    On-policy runner for training and evaluation. Inspired from https://spinningup.openai.com/en/latest/algorithms/vpg.html.
    """

    def __init__(self, env: VecEnv, planner: BasePlanner) -> None:
        # Initialize the environment and agent
        self.env = env
        self.agent = VPGAgent(self.env, planner)
        self.reference_point = planner.reference_points[8]
        self._load_rl_hyperparams()

    def learn(self):
        """
        Main training loop.
        """
        print("Training agent...")

        # Define the actor and critic loss
        @nnx.jit
        def actor_loss_fn(tdres: jnp.ndarray):
            _, logp_a = self.agent.actor.forward(obs, actions)
            return -jnp.sum(tdres * logp_a)

        @nnx.jit
        def critic_loss_fn(returns: jnp.ndarray):
            values = self.agent.critic.forward(obs)
            return jnp.mean((values - returns) ** 2)  # MSE loss

        # Set up buffer
        buffer = VPGBuffer(
            self.env.num_envs,
            self.env.obs_dim,
            self.env.act_dim,
            self.steps_per_epoch,
            self.gamma,
            self.lam,
        )

        # Initialize ADAM optimizers for the actor and critic networks
        actor_optimizer = nnx.Optimizer(
            self.agent.actor, optax.adam(learning_rate=self.actor_lr)
        )
        critic_optimizer = nnx.Optimizer(
            self.agent.critic, optax.adam(learning_rate=self.critic_lr)
        )

        # Initialize the environment
        self.env.reset()
        states, ep_ret, ep_len = (
            self.env.get_states(self.reference_point),
            jnp.zeros(self.env.num_envs),
            0,
        )

        # Main training loop
        for epoch in range(self.epochs):
            ep_returns = jnp.zeros((self.env.num_envs, self.steps_per_epoch))
            for t in range(self.steps_per_epoch):
                a, v, logp = self.agent.act(states)

                r, terminal = self.agent.env.transition(a, states)
                ep_ret += r
                ep_len += 1

                # Log transition
                buffer.store(states, a, r, v, logp)

                # Update state
                states = self.env.get_states(self.reference_point)

                # Check if a timeout is appropriate
                timeout = ep_len == self.max_epoch_len
                epoch_ended = t == self.steps_per_epoch - 1

                # N. B.: could also have a different ep_len for each env and consider timeout and terminal conditions
                # for each env individually, but this is not really necessary with the "carrot on a stick" approach
                if terminal.all() or timeout or epoch_ended:
                    # If the trajectory didn't reach terminal state, bootstrap value target
                    if epoch_ended:
                        _, v, _ = self.agent.act(states)
                    else:
                        v = jnp.zeros(self.env.num_envs)

                    if timeout:
                        ep_returns = ep_returns.at[:, t].set(ep_ret)

                    if terminal.all():
                        terminal_env_indices = jnp.nonzero(terminal)
                        ep_returns = ep_returns.at[terminal_env_indices, t].set(
                            ep_ret[terminal_env_indices]
                        )

                    buffer.end_traj(v)

                    self.env.reset()
                    states, ep_ret, ep_len = (
                        self.env.get_states(self.reference_point),
                        jnp.zeros(self.env.num_envs),
                        0,
                    )

            mean_return = jnp.mean(ep_returns) if len(ep_returns) > 0 else jnp.nan
            print(
                f"Epoch: {epoch+1}/{self.epochs}, mean return across all envs {mean_return}"
            )

            # Get the data from the training loop
            data = buffer.get()

            obs = data["obs"]
            actions = data["act"]
            tdres = data["tdres"]
            returns = data["ret"]

            # Policy gradient update
            loss, grads = nnx.value_and_grad(actor_loss_fn(tdres))(self.agent.actor)
            print(f"{loss = }")
            actor_optimizer.update(grads)

            # Value function updates
            for _ in range(100):
                loss, grads = nnx.value_and_grad(critic_loss_fn(returns))(
                    self.agent.critic
                )
                print(f"{loss = }")
                critic_optimizer.update(grads)

    def evaluate(self) -> None:
        """
        Evaluate the agent.
        """
        print("Evaluating agent...")

        returns = jnp.zeros((self.env.num_envs, self.n_evals))

        for eval in range(self.n_evals):
            print(f"Testing policy: episode {eval+1}/{self.n_evals}")
            states = self.env.get_states(self.reference_point)
            cum_returns = jnp.zeros(self.env.num_envs)
            terminal = jnp.zeros(self.env.num_envs, dtype=bool)
            self.env.reset()
            for _ in range(self.episode_len):
                actions = self.agent.get_control_input(states)
                states = self.env.get_states(self.reference_point)
                rewards, terminal = self.env.transition(actions, states)
                cum_returns += rewards
                if terminal.all():  # Abort if all environments terminated
                    break
            returns = returns.at[:, eval].set(cum_returns)
        print(f"Average return over all envs: {jnp.mean(cum_returns)}")

    def control(
        self,
    ) -> None:  # TODO: save the base and policy networks to be able to use them here
        """
        Control the agent using the previously trained RL controller.
        """
        start_time = time.time()
        states = self.env.get_states(self.reference_point)
        terminal = jnp.zeros(self.env.num_envs, dtype=bool)
        self.env.reset()
        while True:
            real_time = time.time() - start_time
            sim_time = self.env.data.time
            actions = self.agent.get_control_input(states)
            states = self.env.get_states(self.reference_point)
            _, terminal = self.env.transition(actions, states)
            if terminal.all():
                break

    def _load_rl_hyperparams(self) -> None:
        """
        Load the relevant hyperparams from the config file.
        """
        self.steps_per_epoch = self.env.env_cfg.control.RL.steps_per_epoch
        self.epochs = self.env.env_cfg.control.RL.epochs
        self.max_epoch_len = self.env.env_cfg.control.RL.max_epoch_len
        self.gamma = self.env.env_cfg.control.RL.gamma
        self.lam = self.env.env_cfg.control.RL.lam
        self.actor_lr = self.env.env_cfg.control.RL.actor_lr
        self.critic_lr = self.env.env_cfg.control.RL.critic_lr
        self.episode_len = self.env.env_cfg.control.RL.episode_len
        self.n_evals = self.env.env_cfg.control.RL.n_evals
