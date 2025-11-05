from __future__ import annotations

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

    def end_traj(self, last_vals: jnp.ndarray):
        """
        Return the discounted rewards-to-go and TD residuals.
        """
        path_slice = slice(self.path_start_idx, self.ptr)

        rew_seq = self.rew_buf[path_slice]
        val_seq = self.val_buf[path_slice]
        last_vals = jnp.asarray(last_vals)

        rews = jnp.concatenate([rew_seq, last_vals.reshape((1, -1))])
        vals = jnp.concatenate([val_seq, last_vals.reshape((1, -1))])

        deltas = rews[:-1] - vals[:-1] + self.gamma * vals[1:]
        advantages = discount_cumsum(deltas, self.gamma * self.lam)
        for idx, adv in enumerate(advantages):
            self.tdres_buf = self.tdres_buf.at[self.path_start_idx + idx].set(adv)
            self.tdres_filled = self.tdres_filled.at[self.path_start_idx + idx].set(
                True
            )

        returns = discount_cumsum(rews, self.gamma)[:-1]
        for idx, ret in enumerate(returns):
            self.ret_buf = self.ret_buf.at[self.path_start_idx + idx].set(ret)
            self.ret_filled = self.ret_filled.at[self.path_start_idx + idx].set(True)

        self.path_start_idx = self.ptr

    def get(self):
        """
        Return all the data from the buffer (with advantages normalized). Reset pointers in the buffer.
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

        tdres_mean = jnp.mean(tdres, axis=0)
        tdres_std = jnp.std(tdres, axis=0)
        tdres = (tdres - tdres_mean) / tdres_std

        data = dict(
            obs=obs,
            act=act,
            rews=rews,
            ret=ret,
            tdres=tdres,
            logp=logp,
            residuals=residuals,
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
        self.tdres_filled = jnp.zeros((self.max_size,), dtype=bool)
        self.ret_filled = jnp.zeros((self.max_size,), dtype=bool)

        self.ptr = 0
        self.path_start_idx = 0
