import jax
import jax.numpy as jnp

from cuthbertlib.types import Array, ArrayLike, ArrayTree, ArrayTreeLike


def _dummy_array(leaf: ArrayLike) -> Array:
    """Returns an array of the same shape and dtype filled with dummy values."""
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
