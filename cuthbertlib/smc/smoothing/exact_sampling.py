"""Implements exact backward sampling for smoothing in SMC."""

import jax
from jax import numpy as jnp
from jax import random, vmap
from jax.scipy.special import logsumexp

from cuthbertlib.types import (
    Array,
    ArrayLike,
    ArrayTree,
    ArrayTreeLike,
    KeyArray,
    LogConditionalDensity,
)


def log_weights_single(
    x0: ArrayTreeLike,
    x1: ArrayTreeLike,
    log_weight_x0: ArrayLike,
    log_density: LogConditionalDensity,
) -> Array:
    """Compute smoothing weight for a single sample x0 given a single sample x1.

    Args:
        x0: The previous state.
        x1: The current state.
        log_weight_x0: The log weights of the previous state.
        log_density: The log density function of x1 given x0.

    Returns:
        The smoothing weight for sample x0 given a single sample x1.
    """
    return jnp.asarray(log_weight_x0) + log_density(x0, x1)


def log_weights(x0_all, x1, log_weight_x0_all, log_density) -> Array:
    """Compute log smoothing weights over a collection of x0 given a single x1.

    Args:
        x0_all: Collection of previous states.
        x1: The current state.
        log_weight_x0_all: Collection of log weights of the previous state.
        log_density: The log density function of x1 given x0.

    Returns:
        Log normalized smoothing weights for each sample x0 given single sample x1.
    """
    backward_log_weights_all = vmap(
        lambda x0, log_weight_x0: log_weights_single(x0, x1, log_weight_x0, log_density)
    )(x0_all, log_weight_x0_all)

    # Log normalize
    backward_log_weights_all = backward_log_weights_all - logsumexp(
        backward_log_weights_all, axis=0
    )
    return backward_log_weights_all


def simulate_single(
    key, x0_all, x1, log_weight_x0_all, log_density
) -> tuple[ArrayTree, Array]:
    """Sample x0 from a collection given a single x1.

    Args:
        key: A JAX random key.
        x0_all: Collection of previous states.
        x1: The current state.
        log_weight_x0_all: Collection of log weights of the previous state.
        log_density: The log density function of x1 given x0.

    Returns:
        A single sample x0 from the smoothing trajectory along with its index.
    """
    backward_log_weights_all = log_weights(x0_all, x1, log_weight_x0_all, log_density)
    sampled_index = random.categorical(key, backward_log_weights_all)
    return jax.tree.map(lambda z: z[sampled_index], x0_all), sampled_index


def simulate(
    key: KeyArray,
    x0_all: ArrayTreeLike,
    x1_all: ArrayTreeLike,
    log_weight_x0_all: ArrayLike,
    log_density: LogConditionalDensity,
    x1_ancestor_indices: ArrayLike,
) -> tuple[ArrayTree, Array]:
    """Implements the exact backward sampling algorithm for smoothing in SMC.

    Some arguments are only included for protocol compatibility and not used in this
    implementation.

    Args:
        key: JAX PRNG key.
        x0_all: A collection of previous states $x_0$.
        x1_all: A collection of current states $x_1$.
        log_weight_x0_all: The log weights of $x_0$.
        log_density: The log density function of $x_1$ given $x_0$.
        x1_ancestor_indices: The ancestor indices of $x_1$. Not used.

    Returns:
        A collection of samples $x_0$ and their sampled indices.
    """
    log_weight_x0_all = jnp.asarray(log_weight_x0_all)
    n_smoother_particles = jax.tree.leaves(x1_all)[0].shape[0]
    keys = random.split(key, n_smoother_particles)

    return vmap(
        lambda k, x1: simulate_single(k, x0_all, x1, log_weight_x0_all, log_density)
    )(keys, x1_all)
