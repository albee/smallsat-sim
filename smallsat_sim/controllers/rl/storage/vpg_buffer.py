import torch

from smallsat_sim.utils.helpers import discount_cumsum, combined_shape


class VPGBuffer(object):
    """
    Vanilla Policy Gradient buffer to store trajectories. Inspired from https://spinningup.openai.com/en/latest/algorithms/vpg.html.
    """
    def __init__(self, n_envs, obs_dim, act_dim, size, gamma, lam, device) -> None:
        self.obs_buf = torch.zeros(combined_shape(size, (n_envs, obs_dim)), dtype=torch.float32, device=device)
        self.act_buf = torch.zeros(combined_shape(size, (n_envs, act_dim)), dtype=torch.float32, device=device)
        self.tdres_buf = torch.zeros((size, n_envs), dtype=torch.float32, device=device)
        self.rew_buf = torch.zeros((size, n_envs), dtype=torch.float32, device=device)
        self.ret_buf = torch.zeros((size, n_envs), dtype=torch.float32, device=device)
        self.val_buf = torch.zeros((size, n_envs), dtype=torch.float32, device=device)
        self.logp_buf = torch.zeros((size, n_envs), dtype=torch.float32, device=device)
        self.gamma = gamma
        self.lam = lam
        self.ptr = 0
        self.path_start_idx = 0
        self.max_size = size
        self.device = device

    def store(self, obs: torch.tensor, act: torch.tensor, rew: torch.tensor, val: torch.tensor, logp: torch.tensor):
        """
        Append a single timestep to the buffer at each environment update in each environment.
        """
        # Make sure that the buffer still has room
        assert self.ptr < self.max_size

        # Store new data in the respective buffers
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.val_buf[self.ptr] = val
        self.logp_buf[self.ptr] = logp

        # Update pointer
        self.ptr += 1
    
    def end_traj(self, last_val=0): # TODO: double-check this function
        """
        Return the discounted rewards-to-go and TD residuals.
        """
        # Get the indices where the TD residuals and discounted reward-to-go are stored
        path_slice = slice(self.path_start_idx, self.ptr)

        rews = torch.cat(self.rew_buf[path_slice], last_val)
        vals = torch.cat(self.val_buf[path_slice], last_val)
        run_len = self.ptr - self.path_start_idx

        self.ret_buf[self.ptr:self.path_start_idx] = (torch.cumsum(self.rew_buf[self.ptr:self.path_start_idx][::-1])[::-1])

        # TD residual calculation
        deltas = rews[:-1] - vals[:-1] + self.gamma * torch.cat(vals[1:-1], vals[-1])
        self.tdres_buf[path_slice] = torch.tensor([discount_cumsum(deltas[t:run_len], self.gamma * self.lam)[0] for t in range(run_len)])

        # Discounted rewards-to-go calculation
        self.ret_buf[path_slice] = torch.tensor([discount_cumsum(rews[t:run_len], self.gamma)[0] for t in range(run_len)])

        # Update path start index
        self.path_start_idx = self.ptr
    
    def get(self): # TODO: double-check this function
        """
        Return all the data from the buffer (with advantages normalized). Reset pointers in the buffer.
        """
        # Make sure that the buffer is full before getting something from it
        assert self.ptr == self.max_size
        self.ptr, self.path_start_idx = 0, 0

        # Normalize the TD residuals
        tdres_mean = torch.mean(self.tdres_buf)
        tdres_std = torch.std(self.tdres_buf)
        self.tdres_buf = (self.tdres_buf - tdres_mean) / tdres_std

        # Save the data in a dict
        data = dict(obs=self.obs_buf, act=self.act_buf, ret=self.ret_buf, tdres=self.tdres_buf, logp=self.logp_buf)

        return {k: torch.as_tensor(v, dtype=torch.float32, device=self.device) for k, v in data.items()}
