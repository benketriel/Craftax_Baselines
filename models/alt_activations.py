import jax.numpy as jnp
import flax.linen as nn


def relog(t):
    s = jnp.sign(t)
    x = s * jnp.log1p(jnp.abs(t))
    x = jnp.maximum(x, 0.01 * x)
    return x


def get_activation(name: str):
    if name == "relu":
        return nn.relu
    elif name == "tanh":
        return nn.tanh
    elif name == "relog":
        return relog
    else:
        raise ValueError(f"Unknown activation {name}")


