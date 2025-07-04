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
    """
    Compute backward smoothing weights given a single sample from x0 with accompanying
    log weight, a single sample x1 and a log conditional density p(x1 | x0).

    Args:
        x0: The previous state.
        x1: The current state.
        log_weight_x0: The log weights of the previous state.
        log_density: The log density of x1 given x0.

    Returns:
        The backward weight for sample x0 given a single sample x1.
    """
    return jnp.asarray(log_weight_x0) + log_density(x0, x1)


def log_weights(x0_all, x1, log_weight_x0_all, log_density) -> Array:
    """
    Compute backward smoothing weights given a collection of samples from x0 with
    accompanying log weights, a single sample x1 and a log conditional density
    p(x1 | x0).

    Args:
        x0: Collection of previous states.
        x1: The current state.
        log_weight_x0_all: Collection of log weights of the previous state.
        log_density: The log density of x1 given x0.

    Returns:
        Log normalized backward weights for each sample x0 given single sample x1.
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
    """
    Sample a backward x0 given a collection of samples from x0 with accompanying
    log weights, a single sample x1 and a log conditional density p(x1 | x0).

    Args:
        key: A JAX random key.
        x0_all: Collection of previous states.
        x1: The current state.
        log_weight_x0_all: Collection of log weights of the previous state.
        log_density: The log density of x1 given x0.

    Returns:
        A single sample x0 from the backward trajectory along with its index.
    """
    backward_log_weights_all = log_weights(x0_all, x1, log_weight_x0_all, log_density)
    sampled_index = random.categorical(key, backward_log_weights_all)
    return x0_all[sampled_index], sampled_index


def simulate(
    key: KeyArray,
    x0_all: ArrayTreeLike,
    x1_all: ArrayTreeLike,
    log_weight_x0_all: ArrayLike,
    log_density: LogConditionalDensity,
) -> tuple[ArrayTreeLike, Array]:
    """
    Sample a collection of x0 that combine with the provided x1 to give a collection of
    pairs (x0, x1) from the smoothing distribution.

    Args:
        key: A JAX random key.
        x0_all: Collection of previous states.
        x1_all: Collection of current states.
        log_weight_x0_all: Collection of log weights of the previous state.
        log_density: The log density of x1 given x0.

    Returns:
        A collection of x0 and their sampled indices.
    """
    keys = random.split(key, x1_all.shape[0])

    return vmap(
        lambda k, x1: simulate_single(k, x0_all, x1, log_weight_x0_all, log_density)
    )(keys, x1_all)
