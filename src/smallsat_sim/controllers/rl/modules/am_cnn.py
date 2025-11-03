from flax import nnx
import jax.numpy as jnp


class CNNAdaptationModule(nnx.Module):
    """
    Returns the extrinsics given the history of states and actions. Uses a 1-D CNN based to capture temporal correlations.
    """

    def __init__(self, n_steps: int, state_action_dim: int, ext_dim: int):
        super().__init__()
        self.encoder = nnx.Linear(
            in_features=state_action_dim,
            out_features=32,
            rngs=nnx.Rngs(params=0),
        )
        self.conv1 = nnx.Conv(
            in_features=n_steps,
            out_features=32,
            kernel_size=(8,),
            strides=(4,),
            rngs=nnx.Rngs(params=0),
        )
        self.conv2 = nnx.Conv(
            in_features=32,
            out_features=32,
            kernel_size=(5,),
            strides=(1,),
            rngs=nnx.Rngs(params=0),
        )
        self.conv3 = nnx.Conv(
            in_features=32,
            out_features=32,
            kernel_size=(5,),
            strides=(1,),
            rngs=nnx.Rngs(params=0),
        )
        self.linear_output = nnx.Linear(
            in_features=256,
            out_features=max(1, ext_dim),
            rngs=nnx.Rngs(params=0),
        )

    def __call__(self, x):
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
        x = self.linear_output(x)
        return x
