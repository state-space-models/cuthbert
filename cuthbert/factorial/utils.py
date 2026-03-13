"""Utility functions to convert between serial and factorial trees."""

from jax import numpy as jnp
from jax import tree
from jax.lax import scan

from cuthbert.factorial.types import Extract
from cuthbertlib.types import ArrayLike, ArrayTree, ArrayTreeLike

### TODO: Add support for an init factorial state


def serial_to_factorial(
    extract: Extract, serial_tree: ArrayTreeLike, factorial_inds: ArrayLike
) -> list[ArrayTree]:
    """Convert a serial tree into a list of trees, one for each factor.

    Args:
        extract: Function to extract the relevant factors from the serial tree.
        serial_tree: The serial tree to convert.
            Each leaf of the tree should have shape (T, F, ...) where T is the number of
            time steps and F is the number of factors.
            Although some leaves may not have the factorial dimension F, as controlled
            by the `extract` function.
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

    # Initialize factorial trees with empty tree of correct shape (for later concat)
    # This can probably be improved
    init_state = tree.map(lambda x: x[0], serial_tree)
    init_single_factor_state = extract(init_state, jnp.array([0]))
    factorial_trees = [
        tree.map(lambda x: jnp.zeros((0,) + x.shape[1:]), init_single_factor_state)
        for _ in range(num_factors)
    ]

    for t in range(T):
        joint_factor_t = tree.map(lambda x: x[t], serial_tree)
        local_factors_t = extract(joint_factor_t, jnp.arange(len(factorial_inds[t])))

        for j, ind in enumerate(factorial_inds[t]):
            factorial_trees[ind] = tree.map(
                lambda x, y: jnp.concatenate([x, y[j][None]]),
                factorial_trees[ind],
                local_factors_t,
            )

    return factorial_trees


def serial_to_single_factor(
    extract: Extract,
    serial_tree: ArrayTreeLike,
    factorial_inds: ArrayLike,
    factorial_index: int,
) -> ArrayTree:
    """Convert a serial tree into a single factor tree.

    Args:
        extract: Function to extract the relevant factors from the serial tree.
        serial_tree: The serial tree to convert.
            Each leaf of the tree should have shape (T, F, ...) where T is the number of
            time steps and F is the number of factors.
        factorial_inds: The indices of the factors used in each element of the serial
            tree. Shape (T, F).
        factorial_index: Single integer index of the factor to extract.

    Returns:
        A single ArrayTree with shape (T_i, ...) where T_i is the number of occurrences of
        the factorial index in factorial_inds.
    """
    # TODO: As above, we can improve this and make it more JAX-like + efficient.
    factorial_inds = jnp.asarray(factorial_inds)
    T = tree.leaves(serial_tree)[0].shape[0]

    # Initialize factorial tree with empty tree of correct shape (for later concat)
    # This can probably be improved
    init_state = tree.map(lambda x: x[0], serial_tree)
    init_single_factor_state = extract(init_state, jnp.array([0]))
    factorial_tree = tree.map(
        lambda x: jnp.zeros((0,) + x.shape[1:]), init_single_factor_state
    )

    for t in range(T):
        joint_factor_t = tree.map(lambda x: x[t], serial_tree)
        local_factors_t = extract(joint_factor_t, jnp.arange(len(factorial_inds[t])))

        for j, ind in enumerate(factorial_inds[t]):
            if ind == factorial_index:
                factorial_tree = tree.map(
                    lambda x, y: jnp.concatenate([x, y[j][None]]),
                    factorial_tree,
                    local_factors_t,
                )

    return factorial_tree
