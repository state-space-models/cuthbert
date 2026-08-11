"""Utility functions to convert between serial and factorial trees."""

from jax import numpy as jnp
from jax import tree, vmap

from cuthbert.factorial.types import Extract
from cuthbertlib.types import ArrayLike, ArrayTree, ArrayTreeLike


def serial_to_factorial(
    extract: Extract,
    serial_tree: ArrayTreeLike,
    serial_factorial_inds: ArrayLike,
    select_factorial_inds: int | ArrayLike | None = None,
    init_factorial_tree: ArrayTree = None,
) -> ArrayTree | list[ArrayTree]:
    """Convert a serial tree into one or more single-factor trees.

    Args:
        extract: Function to extract the relevant factors from the serial tree.
        serial_tree: The serial tree to convert.
            Each leaf of the tree should have shape (T, F, ...) where T is the number of
            time steps and F is the number of factors.
            Although some leaves may not have the factorial dimension F, as controlled
            by the `extract` function.
        serial_factorial_inds: The indices of the factors used in each element of the
            serial tree. Shape (T, F).
        select_factorial_inds: Single integer index or array of indices of the factors
            to extract in the output. If None, extract all factors from zero through the
            maximum index, preserving the behaviour of earlier versions.
        init_factorial_tree: Optional initial factorial tree to use, as the first
            element of each returned tree.
            Leaves with shape (F, ...)

    Returns:
        A single tree when a single index is selected; otherwise, a list of trees in the
        order of ``select_factorial_inds``. Each tree has shape ``(T_i, ...)``, where
        ``T_i`` is the number of occurrences of that factor (plus one when an initial
        tree is supplied).
    """
    serial_factorial_inds = jnp.asarray(serial_factorial_inds)

    if serial_factorial_inds.ndim != 2:
        raise ValueError(
            "serial_factorial_inds must have shape (T, F), "
            f"got shape {serial_factorial_inds.shape}."
        )

    if select_factorial_inds is None:
        select_factorial_inds = jnp.arange(jnp.max(serial_factorial_inds) + 1)
        return_single_tree = False
    else:
        select_factorial_inds = jnp.asarray(select_factorial_inds)
        return_single_tree = select_factorial_inds.ndim == 0
        select_factorial_inds = jnp.atleast_1d(select_factorial_inds)

    if select_factorial_inds.ndim != 1:
        raise ValueError(
            "select_factorial_inds must be a scalar or one-dimensional array, "
            f"got shape {select_factorial_inds.shape}."
        )

    def extract_single_factor(factorial_index: ArrayLike) -> ArrayTree:
        time_indices, local_indices = jnp.nonzero(
            serial_factorial_inds == factorial_index
        )

        states_at_occurrences = tree.map(lambda leaf: leaf[time_indices], serial_tree)
        factor_states = vmap(extract)(states_at_occurrences, local_indices)

        if init_factorial_tree is None:
            return factor_states

        initial_state = extract(init_factorial_tree, factorial_index)
        return tree.map(
            lambda initial_leaf, history_leaf: jnp.concatenate(
                [initial_leaf[None], history_leaf]
            ),
            initial_state,
            factor_states,
        )

    factorial_trees = [
        extract_single_factor(factorial_index)
        for factorial_index in select_factorial_inds
    ]

    if return_single_tree:
        return factorial_trees[0]
    return factorial_trees
