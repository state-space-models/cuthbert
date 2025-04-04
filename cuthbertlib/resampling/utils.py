import jax
import numpy as np
from jax import Array, numpy as jnp
from jax.typing import ArrayLike
from jax.scipy.special import logsumexp

import numba as nb
from jax.lax import platform_dependent


@jax.jit
def inverse_cdf(sorted_uniforms: ArrayLike, logits: ArrayLike) -> Array:
    """
    Inverse CDF sampling for resampling algorithms.

    Args:
        sorted_uniforms: Sorted uniforms.
        logits: Log-weights, possibly un-normalized.

    Returns:
        Indices of the particles to be resampled.
    """
    weights = jnp.exp(logits - logsumexp(logits))
    return platform_dependent(
        sorted_uniforms, weights, cpu=inverse_cdf_cpu, default=inverse_cdf_default
    )


@jax.jit
def inverse_cdf_default(sorted_uniforms: ArrayLike, weights: ArrayLike) -> Array:
    weights = jnp.asarray(weights)
    M = weights.shape[0]
    cs = jnp.cumsum(weights)
    idx = jnp.searchsorted(cs, sorted_uniforms, method="sort")
    return jnp.clip(idx, 0, M - 1)


@jax.jit
def inverse_cdf_cpu(sorted_uniforms: ArrayLike, weights: ArrayLike) -> Array:
    sorted_uniforms = jnp.asarray(sorted_uniforms)
    weights = jnp.asarray(weights)
    M = weights.shape[0]
    N = sorted_uniforms.shape[0]
    idx = jnp.zeros(N, dtype=int)

    def callback(args):
        su, w, idx_ = args
        idx_ = np.array(idx_)
        su = np.asarray(su)
        w = np.asarray(w)
        inverse_cdf_numba(su, w, idx_)
        return idx_

    idx = jax.pure_callback(
        callback, idx, (sorted_uniforms, weights, idx), vmap_method="sequential"
    )
    return jnp.clip(idx, 0, M - 1)


@nb.njit
def inverse_cdf_numba(su, ws, idx):
    j = 0
    s = ws[0]
    M = su.shape[0]

    for n in range(M):
        while su[n] > s:
            j += 1
            s += ws[j]
        idx[n] = j
