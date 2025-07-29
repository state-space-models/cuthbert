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
    log_potential: LogConditionalDensity,
    x1_ancestors: ArrayLike,
) -> tuple[ArrayTree, Array]:
    """
    An implementation of the ancestor tracing algorithm for smoothing in SMC.

    Some arguments are included for protocol compatibility but not used in this implementation.

    Args:
        key: A JAX random key. Not used.
        x0_all: Collection of previous states.
        x1_all: Collection of current states. Not used.
        log_weight_x0_all: Collection of log weights of the previous state. Not used.
        log_potential: The log density function of x1 given x0. Not used.
        x1_ancestors: The ancestors of x1.

    Returns:
        A collection of x0 and their sampled indices.

    References:
        https://arxiv.org/abs/2207.00976
    """
    x1_ancestors = jnp.asarray(x1_ancestors)
    x0 = jax.tree.map(lambda z: z[x1_ancestors], x0_all)
    return x0, x1_ancestors
