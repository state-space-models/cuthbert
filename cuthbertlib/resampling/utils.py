import jax
import jax.numpy as jnp
import numba as nb
import numpy as np
from jax.lax import platform_dependent
from jax.scipy.special import logsumexp

from cuthbertlib.types import Array, ArrayLike


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
    return jnp.clip(idx, 0, M - 1).astype(
        int
    )  # Ensure indices are integers from the same dtype as basic jax ints.


@jax.jit
def inverse_cdf_cpu(sorted_uniforms: ArrayLike, weights: ArrayLike) -> Array:
    sorted_uniforms = jnp.asarray(sorted_uniforms)
    weights = jnp.asarray(weights)
    M = weights.shape[0]
    N = sorted_uniforms.shape[0]
    idx = jnp.zeros(N, dtype=int)

    def callback(args):
        su, w, idx_ = args
        idx_ = np.array(idx_, dtype=idx.dtype)
        su = np.asarray(su)
        w = np.asarray(w)
        inverse_cdf_numba(su, w, idx_)
        return idx_

    idx = jax.pure_callback(
        callback, idx, (sorted_uniforms, weights, idx), vmap_method="sequential"
    )
    return jnp.clip(idx, 0, M - 1)


@nb.njit(cache=True)
def inverse_cdf_numba(su, ws, idx):
    j = 0
    s = ws[0]
    M = su.shape[0]

    for n in range(M):
        while su[n] > s:
            j += 1
            s += ws[j]
        idx[n] = j
