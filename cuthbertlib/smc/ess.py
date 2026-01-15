"""Importance sampling effective sample size (ESS) computation."""

import jax
import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike


def log_ess(log_weights: ArrayLike) -> Array:
    """Compute the logarithm of the effective sample size (ESS).

    Args:
        log_weights: Array of log weights for the particles.
    """
    return 2 * jax.nn.logsumexp(log_weights) - jax.nn.logsumexp(2 * log_weights)


def ess(log_weights: ArrayLike) -> Array:
    """Compute the effective sample size (ESS).

    Args:
        log_weights: Array of log weights for the particles.
    """
    return jnp.exp(log_ess(log_weights))
