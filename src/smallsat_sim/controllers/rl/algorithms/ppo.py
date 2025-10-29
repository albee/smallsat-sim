import math
from functools import partial

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


# Define the actor loss
def actor_loss_fn(
    actor_model,
    tdres: jnp.ndarray,
    obs_extrinsics: jnp.ndarray,
    actions: jnp.ndarray,
    logp: jnp.ndarray,
    clip_ratio: float,
    entropy_coef: float,
):
    pi, logp_a = actor_model.forward(obs_extrinsics, actions)
    ratio = jnp.exp(logp_a - logp)
    clipped_ratio = jax.lax.clamp(1 - clip_ratio, ratio, 1 + clip_ratio)
    surrogate = jax.lax.min(ratio * tdres, clipped_ratio * tdres)
    entropy_scale = jnp.asarray(entropy_coef, dtype=tdres.dtype)
    entropy_bonus = entropy_scale * pi.entropy()
    return -(surrogate.mean() + entropy_bonus.mean())


# Define the critic loss
def critic_loss_fn(
    critic_model,
    returns: jnp.ndarray,
    obs_extrinsics: jnp.ndarray,
):
    values = critic_model.forward(obs_extrinsics)
    return jnp.mean((values - returns) ** 2)  # MSE loss


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

        # Set the number of training epochs
        self.actor_training_epochs = 100
        self.critic_training_epochs = 100
        self.actor_critic_training_epochs = 100

        # Total batch size
        self.batch_size = self.env.num_envs * self.steps_per_epoch

        # Define the size of each mini-batch
        self.num_minibatches = min(16, self.batch_size)

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

        # Set the clip ratio and the target kl divergence
        self.clip_ratio = 0.2
        self.target_kl = 0.01

    def update_policy_gradient(
        self,
        key,
        obs_extrinsics: jnp.ndarray,
        actions: jnp.ndarray,
        tdres: jnp.ndarray,
        logp: jnp.ndarray,
        minibatch: bool = True,
    ) -> jnp.ndarray:
        """
        Update the policy gradient.
        """
        if minibatch:
            epoch_keys = jax.random.split(key, self.actor_training_epochs)
            actor_graphdef, actor_state = nnx.split((self.actor, self.actor_optimizer))
                actor_loss, actor_state = _actor_epochs_jit(
                    actor_graphdef,
                    actor_state,
                    epoch_keys,
                    obs_extrinsics,
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
                        obs_extrinsics,
                        actions,
                        logp,
                        self.clip_ratio,
                        self.entropy_coef,
                    )

                actor_loss, grads = nnx.value_and_grad(loss_fn)(actor)

                optimizer.update(grads)

                _, logp_a = self.actor.forward(obs_extrinsics, actions)
                kl = (logp - logp_a).mean()
                if kl > 1.5 * self.target_kl:
                    break

        return actor_loss

    def update_value_function(
        self,
        key,
        obs_extrinsics: jnp.ndarray,
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
            critic_loss, critic_state = _critic_epochs_jit(
                critic_graphdef,
                critic_state,
                epoch_keys,
                obs_extrinsics,
                returns,
                self.num_minibatches,
                self.minibatch_size,
            )
            nnx.update((self.critic, self.critic_optimizer), critic_state)
        else:
            for _ in range(self.critic_training_epochs):
                critic, optimizer = self.critic, self.critic_optimizer

                def loss_fn(model):
                    return critic_loss_fn(model, returns, obs_extrinsics)

                critic_loss, grads = nnx.value_and_grad(loss_fn)(critic)

                optimizer.update(grads)

        return critic_loss

    def update_actor_critic_minibatch(
        self,
        key,
        obs_extrinsics: jnp.ndarray,
        actions: jnp.ndarray,
        tdres: jnp.ndarray,
        logp: jnp.ndarray,
        returns: jnp.ndarray,
    ) -> tuple:
        """
        Update the policy gradient and the value function (both at each minibatch, as opposed to doing it sequentially).
        """
        epoch_keys = jax.random.split(key, self.actor_critic_training_epochs)
        joint_graphdef, joint_state = nnx.split(
            (
                self.actor,
                self.actor_optimizer,
                self.critic,
                self.critic_optimizer,
            )
        )
        (
            actor_loss,
            critic_loss,
            joint_state,
        ) = _actor_critic_epochs_jit(
            joint_graphdef,
            joint_state,
            epoch_keys,
            obs_extrinsics,
            actions,
            tdres,
            logp,
            returns,
            self.clip_ratio,
            self.entropy_coef,
            self.target_kl,
            self.num_minibatches,
            self.minibatch_size,
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

        return actor_loss, critic_loss

    def _load_ppo_hyperparams(self) -> None:
        """
        Load PPO-specific hyperparams.
        """
        self.steps_per_epoch = self.env.env_cfg.control.RL.PPO.steps_per_epoch
        self.epochs = self.env.env_cfg.control.RL.PPO.epochs
        self.max_ep_len = self.env.env_cfg.control.RL.PPO.max_ep_len
        self.gamma = self.env.env_cfg.control.RL.PPO.gamma
        self.lam = self.env.env_cfg.control.RL.PPO.lam
        self.actor_lr = self.env.env_cfg.control.RL.PPO.actor_lr
        self.critic_lr = self.env.env_cfg.control.RL.PPO.critic_lr
        self.entropy_coef = self.env.env_cfg.control.RL.PPO.entropy_coef

    def _log(
        self,
        run_id: int,
        timestamp: float,
        stage: str,
        env: VecEnv,
        ctrl_input: jnp.ndarray,
        obs_extrinsics: jnp.ndarray,
    ) -> None:
        """
        Logs desired quantities if flag is enabled
        """
        if self.has_logger:
            tracking_error = calc_lateral_tracking_error(env.get_obs(), self.planner)
            angle_error = jnp.degrees(calc_attitude_error(env.get_obs()))

            if env.ext_dim != 0:
                # If the extrinsics are included in the observations
                extrinsics = obs_extrinsics[:, env.obs_dim :]
            else:
                # If the extrinsics are not included in the observations
                # NOTE: they are still logged but not need to plot them
                extrinsics = jnp.zeros((env.num_envs, env.act_dim))

            extrinsic_error = calc_extrinsic_error(
                extrinsics,
                env.mjx_batch.ctrl
                / (ctrl_input + 1e-8 * jnp.ones_like(env.mjx_batch.ctrl)),
            )
            self.logger.log(
                run_id,
                timestamp,
                run_name=env.run_name,
                stage=stage,
                mean_tracking_error=tracking_error.mean(),
                mean_angle_error=angle_error.mean(),
                mean_extrinsic_error=extrinsic_error.mean(),
            )


def _shuffle_and_batch(
    array: jnp.ndarray,
    permutation: jnp.ndarray,
    num_minibatches: int,
    minibatch_size: int,
) -> jnp.ndarray:
    effective_batch_size = num_minibatches * minibatch_size
    shuffled = array[permutation]
    trimmed = shuffled[:effective_batch_size]
    return trimmed.reshape((num_minibatches, minibatch_size) + trimmed.shape[1:])


@partial(jax.jit, static_argnums=(10, 11))
def _actor_epochs_jit(
    graphdef,
    state,
    epoch_keys: jnp.ndarray,
    obs: jnp.ndarray,
    actions: jnp.ndarray,
    advantages: jnp.ndarray,
    logp: jnp.ndarray,
    clip_ratio: float,
    entropy_coef: float,
    target_kl: float,
    num_minibatches: int,
    minibatch_size: int,
):
    batch_size = obs.shape[0]
    zero_loss = jnp.zeros((), dtype=advantages.dtype)
    zero_kl = jnp.zeros((), dtype=logp.dtype)

    def epoch_body(i, carry):
        state, stop, last_loss, last_kl = carry
        key = epoch_keys[i]

        def run_epoch(args):
            state, stop, last_loss, last_kl, epoch_key = args
            permutation = jax.random.permutation(epoch_key, batch_size)
            obs_batches = _shuffle_and_batch(
                obs, permutation, num_minibatches, minibatch_size
            )
            act_batches = _shuffle_and_batch(
                actions, permutation, num_minibatches, minibatch_size
            )
            adv_batches = _shuffle_and_batch(
                advantages, permutation, num_minibatches, minibatch_size
            )
            logp_batches = _shuffle_and_batch(
                logp, permutation, num_minibatches, minibatch_size
            )

            def minibatch_body(mb, carry):
                state_mb, stop_mb, last_loss_mb, last_kl_mb = carry
                operand = (
                    state_mb,
                    stop_mb,
                    last_loss_mb,
                    last_kl_mb,
                    obs_batches[mb],
                    act_batches[mb],
                    adv_batches[mb],
                    logp_batches[mb],
                )

                def skip_fn(op):
                    state_skip, stop_skip, loss_skip, kl_skip, *_ = op
                    return state_skip, stop_skip, loss_skip, kl_skip

                def update_fn(op):
                    (
                        state_upd,
                        stop_upd,
                        _,
                        _,
                        obs_mb,
                        act_mb,
                        adv_mb,
                        logp_mb,
                    ) = op
                    actor, optimizer = nnx.merge(graphdef, state_upd)

                    def loss_fn(model):
                        return actor_loss_fn(
                            model,
                            adv_mb,
                            obs_mb,
                            act_mb,
                            logp_mb,
                            clip_ratio,
                            entropy_coef,
                        )

                    loss, grads = nnx.value_and_grad(loss_fn)(actor)
                    optimizer.update(grads)
                    _, logp_a = actor.forward(obs_mb, act_mb)
                    kl = jnp.mean(logp_mb - logp_a)
                    new_stop = jnp.logical_or(stop_upd, kl > 1.5 * target_kl)
                    new_state = nnx.state((actor, optimizer))
                    return new_state, new_stop, loss, kl

                return jax.lax.cond(stop_mb, skip_fn, update_fn, operand)

            return jax.lax.fori_loop(
                0,
                num_minibatches,
                minibatch_body,
                (state, stop, last_loss, last_kl),
            )

        return jax.lax.cond(
            stop,
            lambda op: op[:4],
            run_epoch,
            (state, stop, last_loss, last_kl, key),
        )

    state, stop, last_loss, last_kl = jax.lax.fori_loop(
        0,
        epoch_keys.shape[0],
        epoch_body,
        (state, False, zero_loss, zero_kl),
    )

    return last_loss, state


@partial(jax.jit, static_argnums=(5, 6))
def _critic_epochs_jit(
    graphdef,
    state,
    epoch_keys: jnp.ndarray,
    obs: jnp.ndarray,
    returns: jnp.ndarray,
    num_minibatches: int,
    minibatch_size: int,
):
    batch_size = obs.shape[0]
    zero_loss = jnp.zeros((), dtype=returns.dtype)

    def epoch_body(i, carry):
        state, last_loss = carry
        key = epoch_keys[i]
        permutation = jax.random.permutation(key, batch_size)
        obs_batches = _shuffle_and_batch(
            obs, permutation, num_minibatches, minibatch_size
        )
        returns_batches = _shuffle_and_batch(
            returns, permutation, num_minibatches, minibatch_size
        )

        def minibatch_body(mb, carry):
            state_mb, _ = carry
            critic, optimizer = nnx.merge(graphdef, state_mb)

            def loss_fn(model):
                return critic_loss_fn(model, returns_batches[mb], obs_batches[mb])

            loss, grads = nnx.value_and_grad(loss_fn)(critic)
            optimizer.update(grads)
            new_state = nnx.state((critic, optimizer))
            return new_state, loss

        return jax.lax.fori_loop(0, num_minibatches, minibatch_body, (state, last_loss))

    state, last_loss = jax.lax.fori_loop(
        0, epoch_keys.shape[0], epoch_body, (state, zero_loss)
    )

    return last_loss, state


@partial(jax.jit, static_argnums=(11, 12))
def _actor_critic_epochs_jit(
    graphdef,
    state,
    epoch_keys: jnp.ndarray,
    obs: jnp.ndarray,
    actions: jnp.ndarray,
    advantages: jnp.ndarray,
    logp: jnp.ndarray,
    returns: jnp.ndarray,
    clip_ratio: float,
    entropy_coef: float,
    target_kl: float,
    num_minibatches: int,
    minibatch_size: int,
):
    batch_size = obs.shape[0]
    zero_actor_loss = jnp.zeros((), dtype=advantages.dtype)
    zero_critic_loss = jnp.zeros((), dtype=returns.dtype)
    zero_kl = jnp.zeros((), dtype=logp.dtype)

    def epoch_body(i, carry):
        state, stop, last_actor_loss, last_critic_loss, last_kl = carry
        key = epoch_keys[i]

        def run_epoch(args):
            state, stop, last_actor_loss, last_critic_loss, last_kl, epoch_key = args
            permutation = jax.random.permutation(epoch_key, batch_size)
            obs_batches = _shuffle_and_batch(
                obs, permutation, num_minibatches, minibatch_size
            )
            act_batches = _shuffle_and_batch(
                actions, permutation, num_minibatches, minibatch_size
            )
            adv_batches = _shuffle_and_batch(
                advantages, permutation, num_minibatches, minibatch_size
            )
            logp_batches = _shuffle_and_batch(
                logp, permutation, num_minibatches, minibatch_size
            )
            returns_batches = _shuffle_and_batch(
                returns, permutation, num_minibatches, minibatch_size
            )

            def minibatch_body(mb, carry):
                (
                    state_mb,
                    stop_mb,
                    last_actor_loss_mb,
                    last_critic_loss_mb,
                    last_kl_mb,
                ) = carry
                operand = (
                    state_mb,
                    stop_mb,
                    last_actor_loss_mb,
                    last_critic_loss_mb,
                    last_kl_mb,
                    obs_batches[mb],
                    act_batches[mb],
                    adv_batches[mb],
                    logp_batches[mb],
                    returns_batches[mb],
                )

                def skip_fn(op):
                    (
                        state_skip,
                        stop_skip,
                        actor_loss_skip,
                        critic_loss_skip,
                        kl_skip,
                        *_,
                    ) = op
                    return (
                        state_skip,
                        stop_skip,
                        actor_loss_skip,
                        critic_loss_skip,
                        kl_skip,
                    )

                def update_fn(op):
                    (
                        state_upd,
                        stop_upd,
                        _,
                        _,
                        _,
                        obs_mb,
                        act_mb,
                        adv_mb,
                        logp_mb,
                        returns_mb,
                    ) = op
                    actor, actor_opt, critic, critic_opt = nnx.merge(
                        graphdef, state_upd
                    )

                    def actor_loss_local(model):
                        return actor_loss_fn(
                            model,
                            adv_mb,
                            obs_mb,
                            act_mb,
                            logp_mb,
                            clip_ratio,
                            entropy_coef,
                        )

                    actor_loss, actor_grads = nnx.value_and_grad(actor_loss_local)(
                        actor
                    )
                    actor_opt.update(actor_grads)

                    def critic_loss_local(model):
                        return critic_loss_fn(model, returns_mb, obs_mb)

                    critic_loss, critic_grads = nnx.value_and_grad(critic_loss_local)(
                        critic
                    )
                    critic_opt.update(critic_grads)

                    _, logp_a = actor.forward(obs_mb, act_mb)
                    kl = jnp.mean(logp_mb - logp_a)
                    new_stop = jnp.logical_or(stop_upd, kl > 1.5 * target_kl)
                    new_state = nnx.state((actor, actor_opt, critic, critic_opt))
                    return new_state, new_stop, actor_loss, critic_loss, kl

                return jax.lax.cond(stop_mb, skip_fn, update_fn, operand)

            return jax.lax.fori_loop(
                0,
                num_minibatches,
                minibatch_body,
                (state, stop, last_actor_loss, last_critic_loss, last_kl),
            )

        return jax.lax.cond(
            stop,
            lambda op: op[:5],
            run_epoch,
            (state, stop, last_actor_loss, last_critic_loss, last_kl, key),
        )

    state, stop, last_actor_loss, last_critic_loss, last_kl = jax.lax.fori_loop(
        0,
        epoch_keys.shape[0],
        epoch_body,
        (state, False, zero_actor_loss, zero_critic_loss, zero_kl),
    )

    return last_actor_loss, last_critic_loss, state
