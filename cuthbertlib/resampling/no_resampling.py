"""No resampling dummy implementation."""

from functools import partial

from jax import numpy as jnp

from cuthbertlib.resampling.protocols import (
    conditional_resampling_decorator,
    resampling_decorator,
)
from cuthbertlib.resampling.utils import apply_resampling_indices
from cuthbertlib.types import (
    Array,
    ArrayLike,
    ArrayTree,
    ArrayTreeLike,
    ScalarArrayLike,
)

_DESCRIPTION = """
No resampling is performed.
Useful for factorial SMC where resampling is applied during `join` rather than
`filter_combine`."""


@partial(resampling_decorator, name="No Resampling", desc=_DESCRIPTION)
def resampling(
    key: Array, logits: ArrayLike, positions: ArrayTreeLike, n: int
) -> tuple[Array, Array, ArrayTree]:
    logits = jnp.asarray(logits)
    if n != logits.shape[0]:
        raise AssertionError(
            "The number of sampled indices must be equal to the number of "
            "output particles for `No Resampling` resampling."
        )
    return jnp.arange(n), logits, positions


@partial(conditional_resampling_decorator, name="No Resampling", desc=_DESCRIPTION)
def conditional_resampling(
    key: Array,
    logits: ArrayLike,
    positions: ArrayTreeLike,
    n: int,
    pivot_in: ScalarArrayLike,
    pivot_out: ScalarArrayLike,
) -> tuple[Array, Array, ArrayTree]:
    logits = jnp.asarray(logits)
    if n != logits.shape[0]:
        raise AssertionError(
            "The number of sampled indices must be equal to the number of "
            "output particles for `No Resampling` resampling."
        )
    pivot_in = jnp.asarray(pivot_in, dtype=jnp.int32)
    pivot_out = jnp.asarray(pivot_out, dtype=jnp.int32)
    idx = jnp.arange(n).at[pivot_in].set(pivot_out)
    return idx, logits, apply_resampling_indices(positions, idx)
