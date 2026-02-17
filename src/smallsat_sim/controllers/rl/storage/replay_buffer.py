from __future__ import annotations

from collections.abc import Sequence

import jax
import jax.numpy as jnp

from smallsat_sim.utils.helpers_jax import discount_cumsum


class ReplayBuffer(object):
    """
    Device-friendly rollout buffer. Stores entire trajectories in preallocated arrays
    so data never leaves device memory until the epoch is complete.
    """

    def __init__(
        self,
        num_envs,
        obs_dim,
        act_dim,
        res_dim,
        size,
        gamma,
        lam,
    ) -> None:
        self.num_envs = num_envs
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.res_dim = res_dim
        self.gamma = gamma
        self.lam = lam
        self.max_size = size

        self.ptr = 0
        self.path_start_idx = 0

        self._reset_storage()

    def store(
        self,
        obs: jnp.ndarray,
        act: jnp.ndarray,
        rew: jnp.ndarray,
        val: jnp.ndarray,
        logp: jnp.ndarray,
        residuals: jnp.ndarray,
    ):
        """
        Append a single timestep to the buffer at each environment update in each environment.
        """
        if residuals.size == 0:
            residuals = residuals.reshape((residuals.shape[0], 0))

        if self.ptr >= self.max_size:
            raise RuntimeError(
                f"ReplayBuffer overflow: requested index {self.ptr} with max size {self.max_size}"
            )

        idx = self.ptr
        self.obs_buf = self.obs_buf.at[idx].set(obs)
        self.act_buf = self.act_buf.at[idx].set(act)
        self.rew_buf = self.rew_buf.at[idx].set(rew)
        self.val_buf = self.val_buf.at[idx].set(val)
        self.logp_buf = self.logp_buf.at[idx].set(logp)
        self.residuals_buf = self.residuals_buf.at[idx].set(residuals)
        self.tdres_filled = self.tdres_filled.at[idx].set(False)
        self.ret_filled = self.ret_filled.at[idx].set(False)

        self.ptr += 1

    def store_batch(
        self,
        obs: jnp.ndarray,
        act: jnp.ndarray,
        rew: jnp.ndarray,
        val: jnp.ndarray,
        logp: jnp.ndarray,
        residuals: jnp.ndarray,
    ):
        """
        Append a batch of timesteps at once. Each array should have leading dimension T (time)
        followed by the per-environment axes.
        """
        length = obs.shape[0]
        if length == 0:
            return
        end_ptr = self.ptr + length
        if end_ptr > self.max_size:
            raise RuntimeError(
                f"ReplayBuffer overflow: requested indices {self.ptr}:{end_ptr} with max size {self.max_size}"
            )

        idx_slice = slice(self.ptr, end_ptr)
        self.obs_buf = self.obs_buf.at[idx_slice].set(obs)
        self.act_buf = self.act_buf.at[idx_slice].set(act)
        self.rew_buf = self.rew_buf.at[idx_slice].set(rew)
        self.val_buf = self.val_buf.at[idx_slice].set(val)
        self.logp_buf = self.logp_buf.at[idx_slice].set(logp)
        self.residuals_buf = self.residuals_buf.at[idx_slice].set(residuals)
        self.tdres_filled = self.tdres_filled.at[idx_slice].set(False)
        self.ret_filled = self.ret_filled.at[idx_slice].set(False)

        self.ptr = end_ptr

    def end_traj(self, last_vals: jnp.ndarray):
        """
        Return the discounted rewards-to-go and TD residuals.
        """
        self._finalize_slice(self.path_start_idx, self.ptr, last_vals)
        self.path_start_idx = self.ptr

    def end_traj_batch(
        self,
        end_indices: Sequence[int],
        last_vals_batch: jnp.ndarray,
    ):
        """
        Vectorised trajectory finalisation. ``end_indices`` should be the absolute indices
        (within the buffer) marking the end of each trajectory segment (exclusive).
        """
        if not end_indices:
            return
        if len(end_indices) != len(last_vals_batch):
            raise ValueError(
                "end_traj_batch expects the same number of end indices and last values."
            )

        start = self.path_start_idx
        last_vals_batch = jnp.asarray(last_vals_batch)
        for idx, end in enumerate(end_indices):
            if end < start or end > self.ptr:
                raise ValueError(
                    f"Invalid trajectory end index {end}; start={start}, ptr={self.ptr}"
                )
            self._finalize_slice(start, end, last_vals_batch[idx])
            start = end

        self.path_start_idx = start

    def _finalize_slice(
        self, start_idx: int, end_idx: int, last_vals: jnp.ndarray
    ) -> None:
        """
        Compute discounted returns and advantages for the slice [start_idx, end_idx).
        """
        if start_idx == end_idx:
            return

        rew_seq = self.rew_buf[start_idx:end_idx]
        val_seq = self.val_buf[start_idx:end_idx]
        last_vals = jnp.asarray(last_vals)

        rews = jnp.concatenate([rew_seq, last_vals.reshape((1, -1))])
        vals = jnp.concatenate([val_seq, last_vals.reshape((1, -1))])

        deltas = rews[:-1] - vals[:-1] + self.gamma * vals[1:]
        advantages = discount_cumsum(deltas, self.gamma * self.lam)
        returns = discount_cumsum(rews, self.gamma)[:-1]

        self.tdres_buf = self.tdres_buf.at[start_idx:end_idx].set(advantages)
        self.tdres_filled = self.tdres_filled.at[start_idx:end_idx].set(True)
        self.ret_buf = self.ret_buf.at[start_idx:end_idx].set(returns)
        self.ret_filled = self.ret_filled.at[start_idx:end_idx].set(True)

    def finalize_with_masks(
        self,
        done_masks: jnp.ndarray,
        bootstrap_values: jnp.ndarray,
        terminated_masks: jnp.ndarray | None = None,
        truncated_masks: jnp.ndarray | None = None,
    ) -> None:
        """
        Compute returns/advantages for a fixed-horizon rollout using per-env done
        masks and per-step bootstrap values.
        """
        if self.ptr == 0:
            return

        done_masks = jnp.asarray(done_masks, dtype=bool)
        bootstrap_values = jnp.asarray(bootstrap_values)
        expected_shape = (self.ptr, self.num_envs)
        if done_masks.shape != expected_shape:
            raise ValueError(
                f"done_masks must have shape {expected_shape}, got {done_masks.shape}"
            )
        if bootstrap_values.shape != expected_shape:
            raise ValueError(
                f"bootstrap_values must have shape {expected_shape}, got {bootstrap_values.shape}"
            )

        if terminated_masks is None:
            terminated_masks = jnp.zeros_like(done_masks)
        else:
            terminated_masks = jnp.asarray(terminated_masks, dtype=bool)
            if terminated_masks.shape != expected_shape:
                raise ValueError(
                    f"terminated_masks must have shape {expected_shape}, got {terminated_masks.shape}"
                )

        if truncated_masks is None:
            truncated_masks = jnp.zeros_like(done_masks)
        else:
            truncated_masks = jnp.asarray(truncated_masks, dtype=bool)
            if truncated_masks.shape != expected_shape:
                raise ValueError(
                    f"truncated_masks must have shape {expected_shape}, got {truncated_masks.shape}"
                )

        rews = self.rew_buf[: self.ptr]
        vals = self.val_buf[: self.ptr]
        gamma = jnp.asarray(self.gamma, dtype=vals.dtype)
        lam = jnp.asarray(self.lam, dtype=vals.dtype)
        bootstrap_last = bootstrap_values[-1]

        def _scan_step(carry, inputs):
            next_adv, next_val = carry
            rew_t, val_t, done_t, boot_t = inputs
            next_value = jnp.where(done_t, boot_t, next_val)
            delta = rew_t + gamma * next_value - val_t
            adv_t = delta + gamma * lam * (1.0 - done_t.astype(vals.dtype)) * next_adv
            ret_t = adv_t + val_t
            return (adv_t, val_t), (adv_t, ret_t)

        (_, _), (advantages_rev, returns_rev) = jax.lax.scan(
            _scan_step,
            (jnp.zeros((self.num_envs,), dtype=vals.dtype), bootstrap_last),
            (
                rews[::-1],
                vals[::-1],
                done_masks[::-1],
                bootstrap_values[::-1],
            ),
        )
        advantages = advantages_rev[::-1]
        returns = returns_rev[::-1]

        self.tdres_buf = self.tdres_buf.at[: self.ptr].set(advantages)
        self.ret_buf = self.ret_buf.at[: self.ptr].set(returns)
        self.tdres_filled = self.tdres_filled.at[: self.ptr].set(True)
        self.ret_filled = self.ret_filled.at[: self.ptr].set(True)
        self.done_buf = self.done_buf.at[: self.ptr].set(done_masks)
        self.terminated_buf = self.terminated_buf.at[: self.ptr].set(terminated_masks)
        self.truncated_buf = self.truncated_buf.at[: self.ptr].set(truncated_masks)
        self.bootstrap_buf = self.bootstrap_buf.at[: self.ptr].set(bootstrap_values)

    def get(self):
        """
        Return all the data from the buffer. Reset pointers in the buffer.
        """
        if self.ptr != self.max_size:
            raise RuntimeError(
                f"ReplayBuffer incomplete: collected {self.ptr} steps but expected {self.max_size}"
            )

        if not jnp.all(self.tdres_filled):
            missing = jnp.where(~self.tdres_filled)[0]
            raise RuntimeError(
                f"ReplayBuffer has unset entries for tdres at indices {missing}"
            )

        if not jnp.all(self.ret_filled):
            missing = jnp.where(~self.ret_filled)[0]
            raise RuntimeError(
                f"ReplayBuffer has unset entries for ret at indices {missing}"
            )

        obs = self.obs_buf
        act = self.act_buf
        rews = self.rew_buf
        tdres = self.tdres_buf
        ret = self.ret_buf
        logp = self.logp_buf
        residuals = self.residuals_buf

        data = dict(
            obs=obs,
            act=act,
            rews=rews,
            ret=ret,
            tdres=tdres,
            logp=logp,
            residuals=residuals,
            vals=self.val_buf,
            done=self.done_buf,
            terminated=self.terminated_buf,
            truncated=self.truncated_buf,
            bootstrap=self.bootstrap_buf,
        )

        self._reset_storage()

        return data

    def _reset_storage(self):
        obs_shape = (self.max_size, self.num_envs, self.obs_dim)
        act_shape = (self.max_size, self.num_envs, self.act_dim)
        rew_shape = (self.max_size, self.num_envs)
        residuals_shape = (self.max_size, self.num_envs, self.res_dim)

        self.obs_buf = jnp.zeros(obs_shape)
        self.act_buf = jnp.zeros(act_shape)
        self.rew_buf = jnp.zeros(rew_shape)
        self.val_buf = jnp.zeros(rew_shape)
        self.logp_buf = jnp.zeros(rew_shape)
        self.residuals_buf = jnp.zeros(residuals_shape)
        self.tdres_buf = jnp.zeros(rew_shape)
        self.ret_buf = jnp.zeros(rew_shape)
        self.done_buf = jnp.zeros(rew_shape, dtype=bool)
        self.terminated_buf = jnp.zeros(rew_shape, dtype=bool)
        self.truncated_buf = jnp.zeros(rew_shape, dtype=bool)
        self.bootstrap_buf = jnp.zeros(rew_shape)
        self.tdres_filled = jnp.zeros((self.max_size,), dtype=bool)
        self.ret_filled = jnp.zeros((self.max_size,), dtype=bool)

        self.ptr = 0
        self.path_start_idx = 0
