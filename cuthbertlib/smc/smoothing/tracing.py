import jax
from jax import numpy as jnp

from cuthbertlib.types import (
    Array,
    ArrayLike,
    ArrayTree,
    ArrayTreeLike,
    KeyArray,
    LogConditionalDensity,
)


def simulate(
    key: KeyArray,
    x0_all: ArrayTreeLike,
    x1_all: ArrayTreeLike,
    log_weight_x0_all: ArrayLike,
    log_density: LogConditionalDensity,
    x1_ancestor_indices: ArrayLike,
) -> tuple[ArrayTree, Array]:
    """
    An implementation of the ancestor/genealogy tracing algorithm for smoothing in SMC.

    Some arguments are included for protocol compatibility but not used in this implementation.

    Args:
        key: A JAX random key. Not used.
        x0_all: Collection of previous states.
        x1_all: Collection of current states. Not used.
        log_weight_x0_all: Collection of log weights of the previous state. Not used.
        log_density: The log density function of x1 given x0. Not used.
        x1_ancestor_indices: The ancestor indices of x1.

    Returns:
        A collection of x0 and their sampled indices.

    References:
        https://arxiv.org/abs/2207.00976
    """
    x1_ancestor_indices = jnp.asarray(x1_ancestor_indices)
    x0 = jax.tree.map(lambda z: z[x1_ancestor_indices], x0_all)
    return x0, x1_ancestor_indices
