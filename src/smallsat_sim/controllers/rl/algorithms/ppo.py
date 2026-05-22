import math
import jax
import jax.numpy as jnp
from flax import nnx
import optax

from smallsat_sim.envs.vec_env import VecEnv
from smallsat_sim.controllers.rl.algorithms.base_agent import BaseAgent
from smallsat_sim.utils.helpers_jax import (
    calc_lateral_tracking_error,
    calc_attitude_error,
    calc_extrinsic_error,
)


from smallsat_sim.controllers.rl.algorithms.ppo_updates import (
    _actor_critic_epochs_jit,
    _actor_epochs_jit,
    _critic_epochs_jit,
    _diag_gaussian_kl,
    actor_loss_fn,
    critic_loss_fn,
)


class PPO(BaseAgent):
    """
    Proximal Policy Optimization (PPO) agent.
    """

    def __init__(
        self,
        env,
        planner,
        rng_key: jnp.ndarray,
        activation=nnx.tanh,
    ) -> None:
        super().__init__(env, planner, rng_key=rng_key, activation=activation)

        # Load hyperparams
        self._load_ppo_hyperparams()

        # Total batch size
        self.batch_size = self.env.num_envs * self.steps_per_epoch

        # Define the size of each mini-batch based on config, capped by batch size.
        self.num_minibatches = max(1, min(self.num_minibatches, self.batch_size))

        # Ensure fixed-size minibatches to avoid recompilation and enable JIT friendliness
        desired_minibatches = self.num_minibatches
        gcd = math.gcd(self.batch_size, desired_minibatches)
        if gcd != desired_minibatches:
            self.num_minibatches = gcd if gcd > 0 else 1
        self.minibatch_size = self.batch_size // self.num_minibatches
        self.effective_batch_size = self.minibatch_size * self.num_minibatches
        self._dropped_samples = self.batch_size - self.effective_batch_size
        if self._dropped_samples > 0:
            print(
                f"[PPO] Dropping {self._dropped_samples} samples per epoch to keep minibatch size {self.minibatch_size}."
            )

        # Compute the total number of steps
        actor_total_steps = int(
            self.epochs * self.actor_training_epochs * self.num_minibatches
        )  # Upper bound because of early stopping
        critic_total_steps = int(
            self.epochs * self.critic_training_epochs * self.num_minibatches
        )

        # Initialize ADAM optimizers for the actor and critic networks with linear lr decay
        self.actor_optimizer = nnx.Optimizer(
            self.actor,
            optax.adam(
                learning_rate=optax.schedules.linear_schedule(
                    self.actor_lr, self.actor_lr * 0.01, actor_total_steps
                ),
                eps=1e-5,
            ),
        )
        self.critic_optimizer = nnx.Optimizer(
            self.critic,
            optax.adam(
                learning_rate=optax.schedules.linear_schedule(
                    self.critic_lr, self.critic_lr * 0.01, critic_total_steps
                ),
                eps=1e-5,
            ),
        )

        self._jit_warmup_done = False

    def update_policy_gradient(
        self,
        key,
        obs_residuals: jnp.ndarray,
        actions: jnp.ndarray,
        tdres: jnp.ndarray,
        logp: jnp.ndarray,
        minibatch: bool = True,
    ) -> jnp.ndarray:
        """
        Update the policy gradient.
        """
        tdres_mean = jnp.mean(tdres)
        tdres_std = jnp.std(tdres)
        tdres = (tdres - tdres_mean) / (tdres_std + 1e-8)

        old_actor_state = nnx.state(self.actor)
        if minibatch:
            epoch_keys = jax.random.split(key, self.actor_training_epochs)
            actor_graphdef, actor_state = nnx.split((self.actor, self.actor_optimizer))
            (
                actor_loss,
                _,
                _,
                _,
                actor_state,
            ) = _actor_epochs_jit(
                actor_graphdef,
                actor_state,
                self._actor_graphdef,
                old_actor_state,
                epoch_keys,
                obs_residuals,
                actions,
                tdres,
                logp,
                self.clip_ratio,
                self.entropy_coef,
                self.target_kl,
                self.num_minibatches,
                self.minibatch_size,
            )
            nnx.update((self.actor, self.actor_optimizer), actor_state)
        else:
            actor_loss = jnp.array(0.0, dtype=tdres.dtype)
            for _ in range(self.actor_training_epochs):
                actor, optimizer = self.actor, self.actor_optimizer

                def loss_fn(model):
                    return actor_loss_fn(
                        model,
                        tdres,
                        obs_residuals,
                        actions,
                        logp,
                        self.clip_ratio,
                        self.entropy_coef,
                    )

                actor_loss, grads = nnx.value_and_grad(loss_fn)(actor)

                optimizer.update(grads)

                actor_old = nnx.merge(self._actor_graphdef, old_actor_state)
                pi_old, _ = actor_old.forward(obs_residuals, actions)
                pi_new, _ = self.actor.forward(obs_residuals, actions)
                kl = jnp.mean(
                    _diag_gaussian_kl(
                        pi_old.mean(),
                        pi_old.stddev(),
                        pi_new.mean(),
                        pi_new.stddev(),
                    )
                )
                if kl > 1.5 * self.target_kl:
                    break

        return actor_loss

    def update_value_function(
        self,
        key,
        obs_residuals: jnp.ndarray,
        returns: jnp.ndarray,
        minibatch: bool = True,
    ) -> jnp.ndarray:
        """
        Update the value function.
        """

        critic_loss = jnp.array(0.0, dtype=returns.dtype)

        if minibatch:
            epoch_keys = jax.random.split(key, self.critic_training_epochs)
            critic_graphdef, critic_state = nnx.split(
                (self.critic, self.critic_optimizer)
            )
            # Use the live critic instance to compute current predictions as
            # the "old" values for clipping fallback. We avoid merging graphdef
            # + state here which may return different structures depending on
            # how nnx.split was called.
            values_old_arg = self.critic.forward(obs_residuals)

            critic_loss, critic_state = _critic_epochs_jit(
                critic_graphdef,
                critic_state,
                epoch_keys,
                obs_residuals,
                returns,
                values_old_arg,
                self.use_value_clip,
                self.value_clip_coef,
                self.debug_prints,
                self.num_minibatches,
                self.minibatch_size,
            )
            nnx.update((self.critic, self.critic_optimizer), critic_state)
        else:
            for _ in range(self.critic_training_epochs):
                critic, optimizer = self.critic, self.critic_optimizer

                def loss_fn(model):
                    return critic_loss_fn(model, returns, obs_residuals)

                critic_loss, grads = nnx.value_and_grad(loss_fn)(critic)
                optimizer.update(grads)

        return critic_loss

    def update_actor_critic_minibatch(
        self,
        key,
        obs_residuals: jnp.ndarray,
        actions: jnp.ndarray,
        tdres: jnp.ndarray,
        logp: jnp.ndarray,
        returns: jnp.ndarray,
        values_old: jnp.ndarray | None = None,
        critic_grad_scale: float = 1.0,
    ) -> tuple:
        """
        Update the policy gradient and the value function (both at each minibatch, as opposed to doing it sequentially).
        """
        tdres_mean = jnp.mean(tdres)
        tdres_std = jnp.std(tdres)
        tdres = (tdres - tdres_mean) / (tdres_std + 1e-8)

        old_actor_state = nnx.state(self.actor)
        epoch_keys = jax.random.split(key, self.actor_critic_training_epochs)
        joint_graphdef, joint_state = nnx.split(
            (
                self.actor,
                self.actor_optimizer,
                self.critic,
                self.critic_optimizer,
            )
        )

        # Use provided stored (old) values if available, otherwise compute
        # predictions from the current critic. Using the stored rollout-time
        # values is preferred for PPO value clipping.
        if values_old is None:
            values_old_arg = self.critic.forward(obs_residuals)
        else:
            values_old_arg = values_old

        (
            last_actor_loss,
            last_critic_loss,
            mean_actor_loss,
            mean_critic_loss,
            last_kl,
            mean_kl,
            mean_clip_frac,
            joint_state,
        ) = _actor_critic_epochs_jit(
            joint_graphdef,
            joint_state,
            self._actor_graphdef,
            old_actor_state,
            epoch_keys,
            obs_residuals,
            actions,
            tdres,
            logp,
            returns,
            values_old_arg,
            self.clip_ratio,
            self.entropy_coef,
            self.target_kl,
            self.use_value_clip,
            self.value_clip_coef,
            self.debug_prints,
            self.num_minibatches,
            self.minibatch_size,
            critic_grad_scale,
        )

        nnx.update(
            (
                self.actor,
                self.actor_optimizer,
                self.critic,
                self.critic_optimizer,
            ),
            joint_state,
        )

        return (
            last_actor_loss,
            last_critic_loss,
            mean_actor_loss,
            mean_critic_loss,
            last_kl,
            mean_kl,
            mean_clip_frac,
        )

    def warmup_jit(self) -> None:
        """
        Trigger compilation of the jitted PPO update without mutating live parameters.
        """
        if getattr(self, "_jit_warmup_done", False):
            return

        obs_dim_total = self.env.obs_dim + self.env.res_dim
        batch = self.batch_size
        dummy_obs = jnp.zeros((batch, obs_dim_total), dtype=jnp.float32)
        dummy_actions = jnp.zeros((batch, self.env.act_dim), dtype=jnp.float32)
        dummy_tdres = jnp.zeros((batch,), dtype=jnp.float32)
        dummy_logp = jnp.zeros((batch,), dtype=jnp.float32)
        dummy_returns = jnp.zeros((batch,), dtype=jnp.float32)

        joint_graphdef, joint_state = nnx.split(
            (
                self.actor,
                self.actor_optimizer,
                self.critic,
                self.critic_optimizer,
            )
        )
        old_actor_state = nnx.state(self.actor)
        warmup_keys = jax.random.split(
            jax.random.PRNGKey(0), self.actor_critic_training_epochs
        )
        dummy_old_values = jnp.zeros((batch,), dtype=jnp.float32)
        warmup_result = _actor_critic_epochs_jit(
            joint_graphdef,
            joint_state,
            self._actor_graphdef,
            old_actor_state,
            warmup_keys,
            dummy_obs,
            dummy_actions,
            dummy_tdres,
            dummy_logp,
            dummy_returns,
            dummy_old_values,
            self.clip_ratio,
            self.entropy_coef,
            self.target_kl,
            self.use_value_clip,
            self.value_clip_coef,
            self.debug_prints,
            self.num_minibatches,
            self.minibatch_size,
            1.0,
        )
        jax.block_until_ready(warmup_result[0])
        self._jit_warmup_done = True

    def _load_ppo_hyperparams(self) -> None:
        """
        Load PPO-specific hyperparams.
        """
        cfg = self.env.env_cfg.control.RL.PPO
        self.steps_per_epoch = cfg.steps_per_epoch
        self.epochs = cfg.epochs
        self.max_ep_len = cfg.max_ep_len
        self.gamma = cfg.gamma
        self.lam = cfg.lam
        self.actor_lr = cfg.actor_lr
        self.critic_lr = cfg.critic_lr
        self.entropy_coef = cfg.entropy_coef
        self.actor_training_epochs = int(cfg.actor_training_epochs)
        self.critic_training_epochs = int(cfg.critic_training_epochs)
        self.actor_critic_training_epochs = int(cfg.actor_critic_training_epochs)
        self.num_minibatches = int(cfg.num_minibatches)
        self.clip_ratio = float(cfg.clip_ratio)
        self.target_kl = float(cfg.target_kl)
        self.use_value_clip = bool(cfg.use_value_clip)
        self.value_clip_coef = float(cfg.value_clip_coef)
        self.debug_prints = bool(getattr(cfg, "debug_prints", False))

    def _log(
        self,
        run_id: int,
        timestamp: float,
        stage: str,
        env: VecEnv,
        actual_ext: jnp.ndarray,
        estimated_ext: jnp.ndarray,
    ) -> None:
        """
        Logs desired quantities if flag is enabled
        """
        if self.has_logger:
            tracking_error = calc_lateral_tracking_error(env.get_obs(), self.planner)
            angle_error = jnp.degrees(calc_attitude_error(env.get_obs()))
            if self.env.use_adaptive_approach:
                estimated_wrench = estimated_ext
                if estimated_wrench.shape[-1] != actual_ext.shape[-1]:
                    estimated_wrench = estimated_wrench[..., : actual_ext.shape[-1]]
                extrinsic_error = calc_extrinsic_error(actual_ext, estimated_wrench)
            else:
                extrinsic_error = jnp.zeros(env.num_envs)

            self.logger.log(
                run_id,
                timestamp,
                run_name=env.run_name,
                stage=stage,
                mean_lateral_error=tracking_error.mean(),
                mean_angle_error=angle_error.mean(),
                mean_extrinsic_error=extrinsic_error.mean(),
            )
