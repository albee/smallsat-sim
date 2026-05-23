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
    def __init__(
        self,
        d_model: int,
        hidden_dim: int,
        dropout_rate: float,
        rngs: nnx.Rngs,
    ):
        self.fc1 = nnx.Linear(d_model, hidden_dim, rngs=rngs)
        self.fc2 = nnx.Linear(hidden_dim, d_model, rngs=rngs)
        self.dropout = nnx.Dropout(rate=dropout_rate, rngs=rngs)

    def __call__(self, x: jnp.ndarray, *, training: bool = False) -> jnp.ndarray:
        x = jax.nn.gelu(self.fc1(x))
        x = self.dropout(x, deterministic=not training)
        return self.fc2(x)


class MultiHeadSelfAttention(nnx.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout_rate: float,
        rngs: nnx.Rngs,
    ):
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.q_proj = nnx.Linear(d_model, d_model, rngs=rngs)
        self.k_proj = nnx.Linear(d_model, d_model, rngs=rngs)
        self.v_proj = nnx.Linear(d_model, d_model, rngs=rngs)
        self.out_proj = nnx.Linear(d_model, d_model, rngs=rngs)
        self.dropout = nnx.Dropout(rate=dropout_rate, rngs=rngs)

    def __call__(
        self,
        x: jnp.ndarray,
        mask: jnp.ndarray,
        *,
        training: bool = False,
    ) -> jnp.ndarray:
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
        weights = self.dropout(weights, deterministic=not training)
        attn = jnp.einsum("bhij,bhjd->bhid", weights, v)
        attn = attn.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, self.d_model)
        return self.out_proj(attn)


class MultiHeadCrossAttention(nnx.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout_rate: float,
        rngs: nnx.Rngs,
    ):
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.q_proj = nnx.Linear(d_model, d_model, rngs=rngs)
        self.k_proj = nnx.Linear(d_model, d_model, rngs=rngs)
        self.v_proj = nnx.Linear(d_model, d_model, rngs=rngs)
        self.out_proj = nnx.Linear(d_model, d_model, rngs=rngs)
        self.dropout = nnx.Dropout(rate=dropout_rate, rngs=rngs)

    def __call__(
        self,
        query: jnp.ndarray,
        context: jnp.ndarray,
        *,
        training: bool = False,
        return_weights: bool = False,
    ) -> jnp.ndarray | tuple[jnp.ndarray, jnp.ndarray]:
        """
        Attend from one or more query tokens to encoded history tokens.

        Args:
            query: [B, Q, D]
            context: [B, T, D]
        """
        batch_size, query_len, _ = query.shape
        context_len = context.shape[1]
        q = (
            self.q_proj(query)
            .reshape(batch_size, query_len, self.n_heads, self.head_dim)
            .transpose(0, 2, 1, 3)
        )
        k = (
            self.k_proj(context)
            .reshape(batch_size, context_len, self.n_heads, self.head_dim)
            .transpose(0, 2, 1, 3)
        )
        v = (
            self.v_proj(context)
            .reshape(batch_size, context_len, self.n_heads, self.head_dim)
            .transpose(0, 2, 1, 3)
        )
        scale = 1.0 / jnp.sqrt(self.head_dim)
        logits = jnp.einsum("bhqd,bhtd->bhqt", q, k) * scale
        weights = jax.nn.softmax(logits, axis=-1)
        dropped_weights = self.dropout(weights, deterministic=not training)
        attn = jnp.einsum("bhqt,bhtd->bhqd", dropped_weights, v)
        attn = attn.transpose(0, 2, 1, 3).reshape(
            batch_size, query_len, self.d_model
        )
        output = self.out_proj(attn)
        if return_weights:
            return output, weights
        return output


class TransformerBlock(nnx.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        mlp_dim: int,
        dropout_rate: float,
        rngs: nnx.Rngs,
    ):
        self.norm1 = LayerNorm(d_model)
        self.attn = MultiHeadSelfAttention(d_model, n_heads, dropout_rate, rngs)
        self.norm2 = LayerNorm(d_model)
        self.ff = FeedForward(d_model, mlp_dim, dropout_rate, rngs)
        self.dropout = nnx.Dropout(rate=dropout_rate, rngs=rngs)

    def __call__(
        self,
        x: jnp.ndarray,
        mask: jnp.ndarray,
        *,
        training: bool = False,
    ) -> jnp.ndarray:
        attn_out = self.attn(self.norm1(x), mask, training=training)
        attn_out = self.dropout(attn_out, deterministic=not training)
        x = x + attn_out
        ff_out = self.ff(self.norm2(x), training=training)
        ff_out = self.dropout(ff_out, deterministic=not training)
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
        dropout_rate: float = 0.05,
        query_dim: int = 0,
        predict_delta_dim: int = 0,
        predict_tracking: bool = False,
        rngs: nnx.Rngs | None = None,
    ):
        super().__init__()
        if rngs is None:
            rngs = nnx.Rngs(params=0, dropout=1)
        self.n_steps = n_steps
        self.state_action_dim = state_action_dim
        self.ext_dim = max(1, ext_dim)
        self.query_dim = int(query_dim)
        self.predict_delta_dim = int(predict_delta_dim)
        self.predict_tracking = bool(predict_tracking)
        self.d_model = d_model
        self.input_proj = nnx.Linear(state_action_dim, d_model, rngs=rngs)
        self.query_proj = (
            nnx.Linear(self.query_dim, d_model, rngs=rngs)
            if self.query_dim > 0
            else None
        )
        self.pos_embedding = nnx.Param(
            0.02 * jax.random.normal(rngs.params(), (n_steps, d_model))
        )
        self.input_dropout = nnx.Dropout(rate=dropout_rate, rngs=rngs)
        self.blocks = [
            TransformerBlock(d_model, n_heads, mlp_dim, dropout_rate, rngs)
            for _ in range(n_layers)
        ]
        self.norm = LayerNorm(d_model)
        self.mu_head = nnx.Linear(d_model, self.ext_dim, rngs=rngs)
        self.log_sigma_head = nnx.Linear(d_model, self.ext_dim, rngs=rngs)
        self.delta_head = (
            nnx.Linear(d_model, self.predict_delta_dim, rngs=rngs)
            if self.predict_delta_dim > 0
            else None
        )
        self.tracking_head = (
            nnx.Linear(d_model, 1, rngs=rngs) if self.predict_tracking else None
        )

    def __call__(
        self,
        history: jnp.ndarray,
        query: jnp.ndarray | None = None,
        *,
        return_stats: bool = False,
        return_predictions: bool = False,
        training: bool = False,
    ) -> jnp.ndarray | tuple[jnp.ndarray, ...]:
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

        # [1, T, d_model] after projection and learned positional encoding.
        x = self.input_proj(history)
        x = x + self.pos_embedding.value
        x = self.input_dropout(x, deterministic=not training)
        x = jnp.expand_dims(x, axis=0)
        mask = _causal_mask(self.n_steps)

        for block in self.blocks:
            x = block(x, mask, training=training)

        x = self.norm(x)
        last_token = x[:, -1, :]  # [1, d_model]
        if self.query_proj is not None:
            if query is None:
                query = jnp.zeros((self.query_dim,), dtype=last_token.dtype)
            if query.shape[-1] != self.query_dim:
                raise ValueError(
                    f"expected query dim {self.query_dim}, got {query.shape[-1]}"
                )
            last_token = last_token + self.query_proj(jnp.expand_dims(query, axis=0))
        mu = self.mu_head(last_token)
        log_sigma = self.log_sigma_head(last_token)
        mu = jnp.squeeze(mu, axis=0)
        log_sigma = jnp.squeeze(log_sigma, axis=0)
        outputs = [mu]
        if return_stats:
            outputs.append(log_sigma)
        if return_predictions:
            if self.delta_head is not None:
                delta = jnp.squeeze(self.delta_head(last_token), axis=0)
            else:
                delta = jnp.zeros((0,), dtype=mu.dtype)
            if self.tracking_head is not None:
                tracking = jnp.squeeze(self.tracking_head(last_token), axis=0)
            else:
                tracking = jnp.zeros((1,), dtype=mu.dtype)
            outputs.extend([delta, tracking])
        if len(outputs) == 1:
            return outputs[0]
        return tuple(outputs)


class CrossAttentionAdaptationModule(nnx.Module):
    """
    Demand-conditioned transformer adaptation module.

    The history is first encoded with causal self-attention. A task/current-state
    query then cross-attends to the encoded history, so the latent can select
    history tokens relevant to the current desired wrench/control demand.
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
        dropout_rate: float = 0.05,
        query_dim: int = 0,
        predict_delta_dim: int = 0,
        predict_tracking: bool = False,
        rngs: nnx.Rngs | None = None,
    ):
        super().__init__()
        if rngs is None:
            rngs = nnx.Rngs(params=0, dropout=1)
        self.n_steps = n_steps
        self.state_action_dim = state_action_dim
        self.ext_dim = max(1, ext_dim)
        self.query_dim = int(query_dim)
        self.predict_delta_dim = int(predict_delta_dim)
        self.predict_tracking = bool(predict_tracking)
        self.d_model = d_model
        self.input_proj = nnx.Linear(state_action_dim, d_model, rngs=rngs)
        self.query_proj = nnx.Linear(max(1, self.query_dim), d_model, rngs=rngs)
        self.pos_embedding = nnx.Param(
            0.02 * jax.random.normal(rngs.params(), (n_steps, d_model))
        )
        self.input_dropout = nnx.Dropout(rate=dropout_rate, rngs=rngs)
        self.blocks = [
            TransformerBlock(d_model, n_heads, mlp_dim, dropout_rate, rngs)
            for _ in range(n_layers)
        ]
        self.history_norm = LayerNorm(d_model)
        self.query_norm = LayerNorm(d_model)
        self.cross_attn = MultiHeadCrossAttention(d_model, n_heads, dropout_rate, rngs)
        self.cross_dropout = nnx.Dropout(rate=dropout_rate, rngs=rngs)
        self.post_norm = LayerNorm(d_model)
        self.post_ff = FeedForward(d_model, mlp_dim, dropout_rate, rngs)
        self.mu_head = nnx.Linear(d_model, self.ext_dim, rngs=rngs)
        self.log_sigma_head = nnx.Linear(d_model, self.ext_dim, rngs=rngs)
        self.delta_head = (
            nnx.Linear(d_model, self.predict_delta_dim, rngs=rngs)
            if self.predict_delta_dim > 0
            else None
        )
        self.tracking_head = (
            nnx.Linear(d_model, 1, rngs=rngs) if self.predict_tracking else None
        )

    def __call__(
        self,
        history: jnp.ndarray,
        query: jnp.ndarray | None = None,
        *,
        return_stats: bool = False,
        return_predictions: bool = False,
        return_attention: bool = False,
        training: bool = False,
    ) -> jnp.ndarray | tuple[jnp.ndarray, ...]:
        if history.ndim != 2:
            raise ValueError(
                "history must be a 2D array of shape [T, state_action_dim]"
            )
        if history.shape[0] != self.n_steps:
            raise ValueError(f"expected {self.n_steps} steps, got {history.shape[0]}")
        if history.shape[1] != self.state_action_dim:
            raise ValueError(
                f"expected feature dim {self.state_action_dim}, got {history.shape[1]}"
            )

        x = self.input_proj(history)
        x = x + self.pos_embedding.value
        x = self.input_dropout(x, deterministic=not training)
        x = jnp.expand_dims(x, axis=0)
        mask = _causal_mask(self.n_steps)
        for block in self.blocks:
            x = block(x, mask, training=training)
        x = self.history_norm(x)

        if self.query_dim > 0:
            if query is None:
                query = jnp.zeros((self.query_dim,), dtype=x.dtype)
            if query.shape[-1] != self.query_dim:
                raise ValueError(
                    f"expected query dim {self.query_dim}, got {query.shape[-1]}"
                )
            query_token = self.query_proj(jnp.expand_dims(query, axis=0))
        else:
            query_token = self.query_proj(jnp.zeros((1, 1), dtype=x.dtype))
        query_token = jnp.expand_dims(query_token, axis=1)
        query_token = self.query_norm(query_token)

        attended, attn_weights = self.cross_attn(
            query_token,
            x,
            training=training,
            return_weights=True,
        )
        token = query_token + self.cross_dropout(attended, deterministic=not training)
        token = token + self.post_ff(self.post_norm(token), training=training)
        token = jnp.squeeze(token, axis=1)

        mu = jnp.squeeze(self.mu_head(token), axis=0)
        log_sigma = jnp.squeeze(self.log_sigma_head(token), axis=0)
        outputs = [mu]
        if return_stats:
            outputs.append(log_sigma)
        if return_predictions:
            if self.delta_head is not None:
                delta = jnp.squeeze(self.delta_head(token), axis=0)
            else:
                delta = jnp.zeros((0,), dtype=mu.dtype)
            if self.tracking_head is not None:
                tracking = jnp.squeeze(self.tracking_head(token), axis=0)
            else:
                tracking = jnp.zeros((1,), dtype=mu.dtype)
            outputs.extend([delta, tracking])
        if return_attention:
            # [heads, steps] averaged by caller if needed.
            outputs.append(jnp.squeeze(attn_weights, axis=(0, 2)))
        if len(outputs) == 1:
            return outputs[0]
        return tuple(outputs)
