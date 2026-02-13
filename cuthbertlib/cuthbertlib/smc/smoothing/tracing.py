"""Implements the ancestor/genealogy tracing algorithm for smoothing in SMC."""

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
    """Implements the ancestor/genealogy tracing algorithm for smoothing in SMC.

    Some arguments are only included for protocol compatibility and not used in this
    implementation.

    Args:
        key: JAX PRNG key. Not used
        x0_all: A collection of previous states $x_0$.
        x1_all: A collection of current states $x_1$. Not used.
        log_weight_x0_all: The log weights of $x_0$. Not used.
        log_density: The log density function of $x_1$ given $x_0$. Not used.
        x1_ancestor_indices: The ancestor indices of $x_1$.

    Returns:
        A collection of samples $x_0$ and their sampled indices.

    References:
        https://arxiv.org/abs/2207.00976
    """
    x1_ancestor_indices = jnp.asarray(x1_ancestor_indices)
    x0 = jax.tree.map(lambda z: z[x1_ancestor_indices], x0_all)
    return x0, x1_ancestor_indices
