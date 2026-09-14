"""Utility functions (filling dummy arrays and trees) for cuthbert."""

import jax
import jax.numpy as jnp

from cuthbertlib.types import Array, ArrayLike, ArrayTree, ArrayTreeLike


def _dummy_array(leaf: ArrayLike | jax.ShapeDtypeStruct) -> Array:
    """Returns an array of the same shape and dtype filled with dummy values."""
    if not isinstance(leaf, jax.ShapeDtypeStruct):
        leaf = jnp.asarray(leaf)

    dtype = leaf.dtype
    shape = leaf.shape

    if jnp.issubdtype(dtype, jnp.integer):
        min_val = jnp.iinfo(dtype).min
    elif jnp.issubdtype(dtype, jnp.floating):
        min_val = jnp.finfo(dtype).min
    elif jnp.issubdtype(dtype, jnp.bool_):
        min_val = False
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")

    return jnp.full(shape, min_val, dtype=dtype)


def dummy_tree_like(pytree: ArrayTreeLike) -> ArrayTree:
    """Returns a pytree with the same structure filled with dummy values."""
    return jax.tree.map(_dummy_array, pytree)


def dummy_leading_element(pytree: ArrayTreeLike) -> ArrayTree:
    """Returns dummy values shaped like a single element of the leading axis.

    Only the shape and dtype of each leaf are read, so this is safe for leaves
    whose leading axis has length zero.

    Args:
        pytree: Pytree whose leaves all have a leading axis.

    Returns:
        Pytree with the same structure, each leaf filled with dummy values and
            with the leading axis removed.
    """
    return jax.tree.map(
        lambda x: _dummy_array(
            jax.ShapeDtypeStruct(jnp.shape(x)[1:], jnp.result_type(x))
        ),
        pytree,
    )
