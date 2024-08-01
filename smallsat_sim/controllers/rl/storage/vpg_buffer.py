import jax.numpy as jnp

from smallsat_sim.utils.helpers import discount_cumsum, combined_shape


class VPGBuffer(object):
    """
    Vanilla Policy Gradient buffer to store trajectories. Inspired from https://spinningup.openai.com/en/latest/algorithms/vpg.html.
    """

    def __init__(self, num_envs, obs_dim, act_dim, size, gamma, lam) -> None:
        self.obs_buf = jnp.zeros(combined_shape(size, (num_envs, obs_dim)))
        self.act_buf = jnp.zeros(combined_shape(size, (num_envs, act_dim)))
        self.tdres_buf = jnp.zeros((size, num_envs))
        self.rew_buf = jnp.zeros((size, num_envs))
        self.ret_buf = jnp.zeros((size, num_envs))
        self.val_buf = jnp.zeros((size, num_envs))
        self.logp_buf = jnp.zeros((size, num_envs))
        self.gamma = gamma
        self.lam = lam
        self.ptr = 0
        self.path_start_idx = 0
        self.max_size = size

    def store(
        self,
        obs: jnp.ndarray,
        act: jnp.ndarray,
        rew: jnp.ndarray,
        val: jnp.ndarray,
        logp: jnp.ndarray,
    ):
        """
        Append a single timestep to the buffer at each environment update in each environment.
        """
        # Make sure that the buffer still has room
        assert self.ptr < self.max_size

        # Store new data in the respective buffers
        self.obs_buf = self.obs_buf.at[self.ptr].set(obs)
        self.act_buf = self.act_buf.at[self.ptr].set(act)
        self.rew_buf = self.rew_buf.at[self.ptr].set(rew)
        self.val_buf = self.val_buf.at[self.ptr].set(val)
        self.logp_buf = self.logp_buf.at[self.ptr].set(logp)

        # Update pointer
        self.ptr += 1

    def end_traj(self, last_val=0):  # TODO: double-check this function
        """
        Return the discounted rewards-to-go and TD residuals.
        """
        # Get the indices where the TD residuals and discounted reward-to-go are stored
        path_slice = slice(self.path_start_idx, self.ptr)

        rews = jnp.concatenate(self.rew_buf[path_slice], last_val)
        vals = jnp.concatenatet(self.val_buf[path_slice], last_val)
        run_len = self.ptr - self.path_start_idx

        self.ret_buf = self.ret_buf.at[self.ptr : self.path_start_idx].set(
            jnp.cumsum(self.rew_buf[self.ptr : self.path_start_idx][::-1])[::-1]
        )

        # TD residual calculation
        deltas = (
            rews[:-1] - vals[:-1] + self.gamma * jnp.concatenate(vals[1:-1], vals[-1])
        )
        self.tdres_buf = self.tdres_buf.at[path_slice].set(
            jnp.ndarray(
                [
                    discount_cumsum(deltas[t:run_len], self.gamma * self.lam)[0]
                    for t in range(run_len)
                ]
            )
        )

        # Discounted rewards-to-go calculation
        self.ret_buf = self.ret_buf.at[path_slice].set(
            jnp.ndarray(
                [
                    discount_cumsum(rews[t:run_len], self.gamma)[0]
                    for t in range(run_len)
                ]
            )
        )

        # Update path start index
        self.path_start_idx = self.ptr

    def get(self):  # TODO: double-check this function
        """
        Return all the data from the buffer (with advantages normalized). Reset pointers in the buffer.
        """
        # Make sure that the buffer is full before getting something from it
        assert self.ptr == self.max_size
        self.ptr, self.path_start_idx = 0, 0

        # Normalize the TD residuals
        tdres_mean = jnp.mean(self.tdres_buf)
        tdres_std = jnp.std(self.tdres_buf)
        self.tdres_buf = (self.tdres_buf - tdres_mean) / tdres_std

        # Save the data in a dict
        data = dict(
            obs=self.obs_buf,
            act=self.act_buf,
            ret=self.ret_buf,
            tdres=self.tdres_buf,
            logp=self.logp_buf,
        )

        return {k: v for k, v in data.items()}
