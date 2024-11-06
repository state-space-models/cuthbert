from jax import Array, numpy as jnp
from jax.typing import ArrayLike
from jax.scipy.special import logsumexp


def inverse_cdf(sorted_uniforms: ArrayLike, logits: ArrayLike) -> Array:
    """
    Inverse CDF sampling for resampling algorithms.

    Args:
        sorted_uniforms: Sorted uniforms.
        logits: Log-weights, possibly unnormalized.

    Returns:
        Indices of the particles to be resampled.
    """
    weights = jnp.exp(logits - logsumexp(logits))
    M = weights.shape[0]
    cs = jnp.cumsum(weights)
    idx = jnp.searchsorted(cs, sorted_uniforms, method="sort")
    return jnp.clip(idx, 0, M - 1)
