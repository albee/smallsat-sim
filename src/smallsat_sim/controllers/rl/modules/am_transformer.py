import jax
import jax.numpy as jnp
from flax import nnx


def _causal_mask(length: int) -> jnp.ndarray:
    """
    Return a causal mask with zeros on/under the diagonal and large negative
    values elsewhere so it can be added to attention logits.
    """
    mask = jnp.tril(jnp.ones((length, length), dtype=jnp.float32))
    return jnp.where(mask > 0, 0.0, -1e9)


def _sinusoidal_encoding(length: int, dim: int) -> jnp.ndarray:
    """
    Non-trainable sinusoidal positional encoding matching the original Transformer.
    """
    positions = jnp.arange(length, dtype=jnp.float32)[:, None]
    div_term = jnp.exp(
        jnp.arange(0, dim, 2, dtype=jnp.float32) * -(jnp.log(10000.0) / dim)
    )
    pe = jnp.zeros((length, dim), dtype=jnp.float32)
    pe = pe.at[:, 0::2].set(jnp.sin(positions * div_term))
    pe = pe.at[:, 1::2].set(jnp.cos(positions * div_term))
    return pe


class LayerNorm(nnx.Module):
    def __init__(self, features: int):
        self.gamma = nnx.Param(jnp.ones((features,), dtype=jnp.float32))
        self.beta = nnx.Param(jnp.zeros((features,), dtype=jnp.float32))
        self.eps = 1e-6

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        mean = x.mean(axis=-1, keepdims=True)
        var = x.var(axis=-1, keepdims=True)
        x_hat = (x - mean) / jnp.sqrt(var + self.eps)
        return self.gamma * x_hat + self.beta


class FeedForward(nnx.Module):
    def __init__(self, d_model: int, hidden_dim: int):
        self.fc1 = nnx.Linear(d_model, hidden_dim, rngs=nnx.Rngs(params=0))
        self.fc2 = nnx.Linear(hidden_dim, d_model, rngs=nnx.Rngs(params=0))

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = jax.nn.gelu(self.fc1(x))
        return self.fc2(x)


class MultiHeadSelfAttention(nnx.Module):
    def __init__(self, d_model: int, n_heads: int):
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.q_proj = nnx.Linear(d_model, d_model, rngs=nnx.Rngs(params=0))
        self.k_proj = nnx.Linear(d_model, d_model, rngs=nnx.Rngs(params=0))
        self.v_proj = nnx.Linear(d_model, d_model, rngs=nnx.Rngs(params=0))
        self.out_proj = nnx.Linear(d_model, d_model, rngs=nnx.Rngs(params=0))

    def __call__(self, x: jnp.ndarray, mask: jnp.ndarray) -> jnp.ndarray:
        """
        Args:
            x: [B, L, D] tensor
            mask: [L, L] causal mask
        """
        batch_size, seq_len, _ = x.shape
        q = (
            self.q_proj(x)
            .reshape(batch_size, seq_len, self.n_heads, self.head_dim)
            .transpose(0, 2, 1, 3)
        )
        k = (
            self.k_proj(x)
            .reshape(batch_size, seq_len, self.n_heads, self.head_dim)
            .transpose(0, 2, 1, 3)
        )
        v = (
            self.v_proj(x)
            .reshape(batch_size, seq_len, self.n_heads, self.head_dim)
            .transpose(0, 2, 1, 3)
        )

        scale = 1.0 / jnp.sqrt(self.head_dim)
        logits = jnp.einsum("bhid,bhjd->bhij", q, k) * scale
        logits = logits + mask[None, None, :, :]
        weights = jax.nn.softmax(logits, axis=-1)
        attn = jnp.einsum("bhij,bhjd->bhid", weights, v)
        attn = attn.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, self.d_model)
        return self.out_proj(attn)


class TransformerBlock(nnx.Module):
    def __init__(self, d_model: int, n_heads: int, mlp_dim: int):
        self.norm1 = LayerNorm(d_model)
        self.attn = MultiHeadSelfAttention(d_model, n_heads)
        self.norm2 = LayerNorm(d_model)
        self.ff = FeedForward(d_model, mlp_dim)

    def __call__(self, x: jnp.ndarray, mask: jnp.ndarray) -> jnp.ndarray:
        attn_out = self.attn(self.norm1(x), mask)
        x = x + attn_out
        ff_out = self.ff(self.norm2(x))
        return x + ff_out


class TransformerAdaptationModule(nnx.Module):
    """
    Transformer-based adaptation module mirroring the interface of AdaptationModule. 
    Accepts a history of concatenated state-action vectors with shape [T, state_action_dim]
    and returns the predicted extrinsics for the most recent timestep. Optionally exposes a
    probabilistic head (mean and log standard deviation) when ``return_stats`` is set to True.
    """

    def __init__(
        self,
        n_steps: int,
        state_action_dim: int,
        ext_dim: int,
        d_model: int = 128,
        n_heads: int = 4,
        mlp_dim: int = 256,
        n_layers: int = 3,
    ):
        super().__init__()
        self.n_steps = n_steps
        self.state_action_dim = state_action_dim
        self.ext_dim = max(1, ext_dim)
        self.d_model = d_model
        self.input_proj = nnx.Linear(state_action_dim, d_model, rngs=nnx.Rngs(params=0))
        self.blocks = [
            TransformerBlock(d_model, n_heads, mlp_dim) for _ in range(n_layers)
        ]
        self.norm = LayerNorm(d_model)
        self.mu_head = nnx.Linear(d_model, self.ext_dim, rngs=nnx.Rngs(params=0))
        self.log_sigma_head = nnx.Linear(d_model, self.ext_dim, rngs=nnx.Rngs(params=0))

    def __call__(
        self,
        history: jnp.ndarray,
        *,
        return_stats: bool = False,
    ) -> jnp.ndarray | tuple[jnp.ndarray, jnp.ndarray]:
        """
        Args:
            history: [T, state_action_dim] with T == n_steps and the most
                     recent transition at index -1.
        Returns:
            extrinsic prediction for the most recent timestep, shape [ext_dim].
        """
        if history.ndim != 2:
            raise ValueError(
                "history must be a 2D array of shape [T, state_action_dim]"
            )
        if history.shape[0] != self.n_steps:
            raise ValueError(f"expected {self.n_steps} steps, got {history.shape[0]}")
        if history.shape[1] != self.state_action_dim:
            raise ValueError(
                f"expected feature dim {self.state_action_dim}, "
                f"got {history.shape[1]}"
            )

        # [1, T, d_model] after projection and positional encoding
        x = self.input_proj(history)
        pos = _sinusoidal_encoding(self.n_steps, self.d_model)
        x = x + pos
        x = jnp.expand_dims(x, axis=0)
        mask = _causal_mask(self.n_steps)

        for block in self.blocks:
            x = block(x, mask)

        x = self.norm(x)
        last_token = x[:, -1, :]  # [1, d_model]
        mu = self.mu_head(last_token)
        log_sigma = self.log_sigma_head(last_token)
        mu = jnp.squeeze(mu, axis=0)
        log_sigma = jnp.squeeze(log_sigma, axis=0)
        if return_stats:
            return mu, log_sigma
        return mu
