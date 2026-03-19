"""Shared protocols for resampling algorithms."""

from typing import Protocol, runtime_checkable

import jax

from cuthbertlib.types import (
    Array,
    ArrayLike,
    ArrayTree,
    ArrayTreeLike,
    KeyArray,
    ScalarArrayLike,
)

_RESAMPLING_DOC = """
Args:
    key: JAX PRNG key.
    logits: Logits.
    positions: ArrayTreeLike
    n: Number of indices to sample.

Returns:
    ancestors: Array of resampling indices.
    logits: Array of log-weights after resampling.
    positions: ArrayTreeLike of resampled positions.
"""

_CONDITIONAL_RESAMPLING_DOC = """
Args:
    key: JAX PRNG key.
    logits: Log-weights, possibly unnormalized.
    positions: ArrayTreeLike
    n: Number of indices to sample.
    pivot_in: Index of the particle to keep.
    pivot_out: Value of the output at index `pivot_in`.

Returns:
    ancestors: Array of size n with indices to use for resampling.
    logits: Array of log-weights after resampling.
    positions: ArrayTreeLike of resampled positions.
"""


@runtime_checkable
class Resampling(Protocol):
    """Protocol for resampling operations."""

    def __call__(
        self, key: KeyArray, logits: ArrayLike, positions: ArrayTreeLike, n: int
    ) -> tuple[Array, Array, ArrayTree]:
        f"""Computes resampling indices according to given logits.
        {_RESAMPLING_DOC}
        """
        ...


@runtime_checkable
class ConditionalResampling(Protocol):
    """Protocol for conditional resampling operations."""

    def __call__(
        self,
        key: KeyArray,
        logits: ArrayLike,
        positions: ArrayTreeLike,
        n: int,
        pivot_in: ScalarArrayLike,
        pivot_out: ScalarArrayLike,
    ) -> tuple[Array, Array, ArrayTree]:
        f"""Conditional resampling.
        {_CONDITIONAL_RESAMPLING_DOC}
        """
        ...


def resampling_decorator(func: Resampling, name: str, desc: str = "") -> Resampling:
    """Decorate Resampling function with unified docstring."""
    doc = f"""
    {name} resampling. {desc}
    {_RESAMPLING_DOC}
    """

    func.__doc__ = doc
    return jax.jit(func, static_argnames=("n",))


def conditional_resampling_decorator(
    func: ConditionalResampling, name: str, desc: str = ""
) -> ConditionalResampling:
    """Decorate ConditionalResampling function with unified docstring."""
    doc = f"""
    {name} conditional resampling. {desc}
    {_CONDITIONAL_RESAMPLING_DOC}
    """

    func.__doc__ = doc
    return jax.jit(func, static_argnames=("n",))
