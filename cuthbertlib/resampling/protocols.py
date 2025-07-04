from typing import Protocol, runtime_checkable

import jax

from cuthbertlib.types import Array, ArrayLike


@runtime_checkable
class Resampling(Protocol):
    """Protocol for resampling operations."""

    def __call__(self, key: Array, logits: ArrayLike, n: int) -> Array: ...


@runtime_checkable
class ConditionalResampling(Protocol):
    """Protocol for conditional resampling operations."""

    def __call__(
        self,
        key: Array,
        logits: ArrayLike,
        n: int,
        pivot_in: int,
        pivot_out: int,
    ) -> Array: ...


def resampling_decorator(func: Resampling, name: str, desc: str = "") -> Resampling:
    """Decorate Resampling function with unified docstring."""

    doc = f"""
    {name} resampling. {desc}

    Args:
        key: PRNGKey to use in resampling
        logits: Log-weights, possibly unnormalized.
        n: Number of indices to sample.
    
    Returns:
        Array of size n with indices to use for resampling.
    """

    func.__doc__ = doc
    return jax.jit(func, static_argnames=("n",))


def conditional_resampling_decorator(
    func: ConditionalResampling, name: str, desc: str = ""
) -> ConditionalResampling:
    """Decorate ConditionalResampling function with unified docstring."""

    doc = f"""
    {name} conditional resampling. {desc}

    Args:
        key: PRNGKey to use in resampling
        logits: Log-weights, possibly unnormalized.
        n: Number of indices to sample
        pivot_in: Index of the particle to keep
        pivot_out: Value of the output at index `pivot_in`
    
    Returns:
        Array of size n with indices to use for resampling.
    """

    func.__doc__ = doc
    return jax.jit(func, static_argnames=("n",))
