from __future__ import annotations

from typing import List, Optional

import jax.numpy as jnp

from smallsat_sim.utils.helpers_jax import discount_cumsum


class ReplayBuffer(object):
    """
    Replay buffer to store trajectories. Inspired from https://spinningup.openai.com/en/latest/algorithms/vpg.html.
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
        # Make sure that the buffer still has room
        if self.ptr >= self.max_size:
            raise RuntimeError(
                f"ReplayBuffer overflow: requested index {self.ptr} with max size {self.max_size}"
            )

        # Cache the data as-is (keeps them on device until we consolidate at the end).
        self.obs_buf.append(obs)
        self.act_buf.append(act)
        self.rew_buf.append(rew)
        self.val_buf.append(val)
        self.logp_buf.append(logp)
        self.residuals_buf.append(residuals)
        self.tdres_buf.append(None)
        self.ret_buf.append(None)

        # Update pointer
        self.ptr += 1

    def end_traj(self, last_vals: jnp.ndarray):
        """
        Return the discounted rewards-to-go and TD residuals.
        """
        # Get the indices where the TD residuals and discounted reward-to-go are stored
        path_slice = slice(self.path_start_idx, self.ptr)

        rew_seq = jnp.stack(self.rew_buf[path_slice], axis=0)
        val_seq = jnp.stack(self.val_buf[path_slice], axis=0)
        last_vals = jnp.asarray(last_vals)

        rews = jnp.concatenate([rew_seq, last_vals.reshape((1, -1))])
        vals = jnp.concatenate([val_seq, last_vals.reshape((1, -1))])

        # TD residual calculation with bootstrap
        deltas = rews[:-1] - vals[:-1] + self.gamma * vals[1:]
        advantages = discount_cumsum(deltas, self.gamma * self.lam)
        for idx, adv in enumerate(advantages):
            self.tdres_buf[self.path_start_idx + idx] = adv

        # Discounted rewards-to-go include bootstrap value (drop last entry)
        returns = discount_cumsum(rews, self.gamma)[:-1]
        for idx, ret in enumerate(returns):
            self.ret_buf[self.path_start_idx + idx] = ret

        # Update path start index
        self.path_start_idx = self.ptr

    def get(self):
        """
        Return all the data from the buffer (with advantages normalized). Reset pointers in the buffer.
        """
        # Make sure that the buffer is full before getting something from it
        if self.ptr != self.max_size:
            raise RuntimeError(
                f"ReplayBuffer incomplete: collected {self.ptr} steps but expected {self.max_size}"
            )

        def _stack_list(name: str, values: List[Optional[jnp.ndarray]]):
            if any(v is None for v in values):
                missing = [idx for idx, v in enumerate(values) if v is None]
                raise RuntimeError(
                    f"ReplayBuffer has unset entries for {name} at indices {missing}"
                )
            return jnp.stack(values, axis=0)

        obs = jnp.stack(self.obs_buf, axis=0)
        act = jnp.stack(self.act_buf, axis=0)
        rews = jnp.stack(self.rew_buf, axis=0)
        tdres = _stack_list("tdres", self.tdres_buf)
        ret = _stack_list("ret", self.ret_buf)
        logp = jnp.stack(self.logp_buf, axis=0)
        residuals = jnp.stack(self.residuals_buf, axis=0)

        # Normalize the TD residuals (could instead also normalize on minibatch-level)
        tdres_mean = jnp.mean(tdres, axis=0)
        tdres_std = jnp.std(tdres, axis=0)
        tdres = (tdres - tdres_mean) / tdres_std

        # Save the data in a dict
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

        return {k: v for k, v in data.items()}

    def _reset_storage(self):
        self.obs_buf: List[jnp.ndarray] = []
        self.act_buf: List[jnp.ndarray] = []
        self.rew_buf: List[jnp.ndarray] = []
        self.val_buf: List[jnp.ndarray] = []
        self.logp_buf: List[jnp.ndarray] = []
        self.residuals_buf: List[jnp.ndarray] = []
        self.tdres_buf: List[Optional[jnp.ndarray]] = []
        self.ret_buf: List[Optional[jnp.ndarray]] = []

        self.ptr = 0
        self.path_start_idx = 0
