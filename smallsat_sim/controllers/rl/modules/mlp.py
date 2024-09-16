import jax
import jax.numpy as jnp
from flax import nnx


def identity(input: jnp.ndarray) -> jnp.ndarray:
    """
    Identity layer (placeholder).
    """
    return input


def mlp(sizes, activation, output_activation=identity):
    """
    Basic multilayer perceptron architecture.
    """
    modules = []

    for i in range(len(sizes) - 1):
        if i >= len(sizes) - 2:
            activation_function = output_activation # nnx.relu for pretraining
        else:
            activation_function = activation
        modules += [
            nnx.Linear(sizes[i], sizes[i + 1], rngs=nnx.Rngs(params=0)),
            activation_function,
        ]

    return nnx.Sequential(*modules)
