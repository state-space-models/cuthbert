"""Utility functions (inverse CDF sampling) for resampling algorithms."""

import jax
import jax.numpy as jnp
import numba as nb
import numpy as np
from jax.lax import platform_dependent
from jax.scipy.special import logsumexp

from cuthbertlib.types import Array, ArrayLike


@jax.jit
def inverse_cdf(sorted_uniforms: ArrayLike, logits: ArrayLike) -> Array:
    """Inverse CDF sampling for resampling algorithms.

    The implementation branches depending on the platform being CPU or GPU (and other parallel envs)
    1. The CPU implementation is a numba-compiled specialized searchsorted(arr, vals) for *sorted* uniforms
        which is guaranteed to run in O(N + M) where N is the size of logits and M that of sorted_uniforms.
        This could be replaced mutatis mutandis by np.searchsorted (not jnp.searchsorted!) but the latter
        does not guarantee this execution time.
    2. On GPU, we use searchsorted with a sorting strategy: this proceeds by merge (arg) sorting,
        which works in log(N+M) and is efficient for large arrays of sorted uniforms, typically our setting.

    Args:
        sorted_uniforms: Sorted uniforms.
        logits: Log-weights, possibly un-normalized.

    Returns:
        Indices of the particles to be resampled.
    """
    weights = jnp.exp(logits - logsumexp(logits))
    return platform_dependent(
        sorted_uniforms, weights, cpu=_inverse_cdf_cpu, default=_inverse_cdf_default
    )


@jax.jit
def _inverse_cdf_default(sorted_uniforms: ArrayLike, weights: ArrayLike) -> Array:
    weights = jnp.asarray(weights)
    M = weights.shape[0]
    cs = jnp.cumsum(weights)
    idx = jnp.searchsorted(cs, sorted_uniforms, method="sort")
    return jnp.clip(idx, 0, M - 1).astype(
        int
    )  # Ensure indices are integers from the same dtype as basic jax ints.


@jax.jit
def _inverse_cdf_cpu(sorted_uniforms: ArrayLike, weights: ArrayLike) -> Array:
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
        _inverse_cdf_numba(su, w, idx_)
        return idx_

    idx = jax.pure_callback(
        callback, idx, (sorted_uniforms, weights, idx), vmap_method="sequential"
    )
    return jnp.clip(idx, 0, M - 1)


@nb.njit
def _inverse_cdf_numba(su, ws, idx):
    j = 0
    s = ws[0]
    M = su.shape[0]
    N = ws.shape[0]

    for n in range(M):
        while su[n] > s and j < N - 1:
            j += 1
            s += ws[j]
        idx[n] = j
