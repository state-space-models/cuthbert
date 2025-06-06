import jax
import jax.numpy as jnp
from cuthbertlib.types import ArrayTreeLike, ArrayTree


def append_tree(
    batched_tree: ArrayTreeLike, tree: ArrayTreeLike, prepend: bool = False
) -> ArrayTree:
    """Append the leaves of a pytree of arrays to the leaves of a batched pytree.

    Args:
        batched_tree: The batched pytree.
        tree: The pytree to append.
        prepend: Whether to prepend `tree` instead of appending it.

    Returns:
        The batched pytree with `tree` appended (or prepended) to it.
    """
    if prepend:
        return jax.tree.map(
            lambda x, y: jnp.concatenate([x[None], y]), tree, batched_tree
        )
    return jax.tree.map(lambda x, y: jnp.concatenate([y, x[None]]), tree, batched_tree)
