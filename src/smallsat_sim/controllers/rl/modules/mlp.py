import jax
import jax.numpy as jnp
from flax import nnx


def identity(input: jnp.ndarray) -> jnp.ndarray:
    """
    Identity layer (placeholder).
    """
    return input


def mlp(
    sizes,
    activation,
    output_activation=None,
    std=jnp.sqrt(2).item(),
    last_layer_std=jnp.sqrt(2).item(),
):
    """
    Basic multilayer perceptron architecture.
    """
    modules = []
    current_std = std

    for i in range(len(sizes) - 1):
        if i >= len(sizes) - 2:
            if output_activation is None:
                activation_function = identity
            else:
                activation_function = output_activation
            current_std = last_layer_std
        else:
            activation_function = activation
        modules += [
            nnx.Linear(
                sizes[i],
                sizes[i + 1],
                # Avoid QR/SVD-based initializers here: on some GPU/CUDA stacks
                # cuSolver handle creation fails during startup. Variance scaling
                # keeps the intended gain without invoking cuSolver.
                kernel_init=jax.nn.initializers.variance_scaling(
                    scale=float(current_std) ** 2,
                    mode="fan_avg",
                    distribution="uniform",
                ),
                bias_init=nnx.initializers.constant(0.0),
                rngs=nnx.Rngs(params=0),
            ),
            activation_function,
        ]

    return nnx.Sequential(*modules)
