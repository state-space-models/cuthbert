"""Utility functions to convert between serial and factorial trees."""

from jax import numpy as jnp
from jax import tree
from jax.lax import scan

from cuthbertlib.types import ArrayLike, ArrayTree, ArrayTreeLike


def serial_to_factorial(
    serial_tree: ArrayTreeLike, factorial_inds: ArrayLike
) -> list[ArrayTree]:
    """Convert a serial tree into a list of trees, one for each factor.

    Args:
        serial_tree: The serial tree to convert.
            Each leaf of the tree should have shape (T, F, ...) where T is the number of
            time steps and F is the number of factors.
        factorial_inds: The indices of the factors used in each element of the serial
            tree. Shape (T, F).

    Returns:
        A list of trees, one for each factor.
            Length max(factorial_inds) + 1.
            Each element has shape (T_i, ...) where T_i is the number of occurrences of
            index i in factorial_inds (which may be zero).
    """
    # TODO: This function is not very JAX-like or efficient, we may want to improve it in time.

    factorial_inds = jnp.asarray(factorial_inds)
    num_factors = jnp.max(factorial_inds) + 1
    T = tree.leaves(serial_tree)[0].shape[0]

    factorial_trees = [
        tree.map(lambda x: jnp.zeros((0,) + x.shape[2:]), serial_tree)
        for _ in range(num_factors)
    ]

    for t in range(T):
        for j, ind in enumerate(factorial_inds[t]):
            serial_factor = tree.map(lambda x: x[t, j], serial_tree)
            factorial_trees[ind] = tree.map(
                lambda x, y: jnp.concatenate([x, y[None]]),
                factorial_trees[ind],
                serial_factor,
            )

    return factorial_trees


def serial_to_single_factor(
    serial_tree: ArrayTreeLike, factorial_inds: ArrayLike, factorial_index: int
) -> ArrayTree:
    """Convert a serial tree into a single factor tree.

    Args:
        serial_tree: The serial tree to convert.
        factorial_inds: The indices of the factors used in each element of the serial
        factorial_index: The index of the factor to convert.
    """
    # TODO: As above, we can improve this and make it more JAX-like + efficient.
    factorial_inds = jnp.asarray(factorial_inds)
    T = tree.leaves(serial_tree)[0].shape[0]

    factorial_tree = tree.map(lambda x: jnp.zeros((0,) + x.shape[2:]), serial_tree)

    for t in range(T):
        for j, ind in enumerate(factorial_inds[t]):
            if ind == factorial_index:
                serial_factor = tree.map(lambda x: x[t, j], serial_tree)
                factorial_tree = tree.map(
                    lambda x, y: jnp.concatenate([x, y[None]]),
                    factorial_tree,
                    serial_factor,
                )

    return factorial_tree
