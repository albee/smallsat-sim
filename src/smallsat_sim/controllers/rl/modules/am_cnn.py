from flax import nnx
import jax.numpy as jnp


class CNNAdaptationModule(nnx.Module):
    """
    Returns the extrinsics given the history of states and actions. Uses a 1-D CNN based to capture temporal correlations.
    """

    def __init__(
        self,
        n_steps: int,
        state_action_dim: int,
        ext_dim: int,
        query_dim: int = 0,
        predict_delta_dim: int = 0,
        predict_tracking: bool = False,
        predict_authority_dim: int = 0,
        rngs: nnx.Rngs | None = None,
    ):
        super().__init__()
        if rngs is None:
            rngs = nnx.Rngs(params=0)
        self.query_dim = int(query_dim)
        self.predict_delta_dim = int(predict_delta_dim)
        self.predict_tracking = bool(predict_tracking)
        self.predict_authority_dim = int(predict_authority_dim)
        self.encoder = nnx.Linear(
            in_features=state_action_dim,
            out_features=32,
            rngs=rngs,
        )
        self.conv1 = nnx.Conv(
            in_features=n_steps,
            out_features=32,
            kernel_size=(8,),
            strides=(4,),
            rngs=rngs,
        )
        self.conv2 = nnx.Conv(
            in_features=32,
            out_features=32,
            kernel_size=(5,),
            strides=(1,),
            rngs=rngs,
        )
        self.conv3 = nnx.Conv(
            in_features=32,
            out_features=32,
            kernel_size=(5,),
            strides=(1,),
            rngs=rngs,
        )
        self.query_proj = (
            nnx.Linear(
                in_features=self.query_dim,
                out_features=256,
                rngs=rngs,
            )
            if self.query_dim > 0
            else None
        )
        self.linear_output = nnx.Linear(
            in_features=256,
            out_features=max(1, ext_dim),
            rngs=rngs,
        )
        self.delta_head = (
            nnx.Linear(
                in_features=256,
                out_features=self.predict_delta_dim,
                rngs=rngs,
            )
            if self.predict_delta_dim > 0
            else None
        )
        self.tracking_head = (
            nnx.Linear(
                in_features=256,
                out_features=1,
                rngs=rngs,
            )
            if self.predict_tracking
            else None
        )
        self.authority_head = (
            nnx.Linear(
                in_features=256,
                out_features=self.predict_authority_dim,
                rngs=rngs,
            )
            if self.predict_authority_dim > 0
            else None
        )

    def __call__(
        self,
        x,
        query: jnp.ndarray | None = None,
        *,
        return_predictions: bool = False,
    ):
        x = self.encoder(x)
        x = nnx.relu(x)
        x = jnp.swapaxes(x, 0, 1)
        x = self.conv1(x)
        x = nnx.leaky_relu(x)
        x = self.conv2(x)
        x = nnx.leaky_relu(x)
        x = self.conv3(x)
        x = nnx.leaky_relu(x)
        x = x.reshape(-1)
        if self.query_proj is not None:
            if query is None:
                query = jnp.zeros((self.query_dim,), dtype=x.dtype)
            if query.shape[-1] != self.query_dim:
                raise ValueError(
                    f"expected query dim {self.query_dim}, got {query.shape[-1]}"
                )
            x = x + self.query_proj(query)
        context = self.linear_output(x)
        if not return_predictions:
            return context
        if self.delta_head is not None:
            delta = self.delta_head(x)
        else:
            delta = jnp.zeros((0,), dtype=context.dtype)
        if self.tracking_head is not None:
            tracking = self.tracking_head(x)
        else:
            tracking = jnp.zeros((1,), dtype=context.dtype)
        if self.authority_head is not None:
            authority = self.authority_head(x)
        else:
            authority = jnp.zeros((0,), dtype=context.dtype)
        return context, delta, tracking, authority
