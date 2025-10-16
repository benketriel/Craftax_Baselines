import jax
import jax.numpy as jnp
from typing import NamedTuple


class EwmaState(NamedTuple):
    mean: jnp.ndarray
    mean_sq: jnp.ndarray
    debias: jnp.ndarray

def ewma_init():
    return EwmaState(jnp.array(0., jnp.float32),
                     jnp.array(0., jnp.float32),
                     jnp.array(0., jnp.float32))

def ewma_update(state: EwmaState, x: jnp.ndarray, weight: float = 0.9) -> EwmaState:
    # x: arbitrary shape; we use mean over all elements
    m = jnp.mean(x)
    m2 = jnp.mean(jnp.square(x))
    mean    = state.mean    * weight + m  * (1.0 - weight)
    mean_sq = state.mean_sq * weight + m2 * (1.0 - weight)
    debias  = state.debias  * weight + 1.0 * (1.0 - weight)
    return EwmaState(mean, mean_sq, debias)

def _ewma_mean_var(state: EwmaState, eps: float = 1e-8):
    deb = jnp.maximum(eps, state.debias)
    mean = state.mean / deb
    mean_sq = state.mean_sq / deb
    var = jnp.maximum(eps, mean_sq - mean**2)
    return mean, var

def ewma_normalize(state: EwmaState, x: jnp.ndarray, also_mean: bool = True, eps: float = 1e-8):
    # handle cold start (no debias yet)
    def _do(x):
        mean, var = _ewma_mean_var(state, eps)
        if also_mean:
            x = x - mean
        return x / jnp.sqrt(var)
    return jax.lax.cond(state.debias <= 0., lambda _: x, lambda _: _do(x), operand=None)

def ewma_denormalize(state: EwmaState, x: jnp.ndarray, eps: float = 1e-8):
    def _do(x):
        mean, var = _ewma_mean_var(state, eps)
        return x * jnp.sqrt(var) + mean
    return jax.lax.cond(state.debias <= 0., lambda _: x, lambda _: _do(x), operand=None)
