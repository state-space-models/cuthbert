"""Protocols for MCMC methods."""

from typing import Any, Protocol, runtime_checkable

from cuthbertlib.types import Array, KeyArray


@runtime_checkable
class AncestorMove(Protocol):
    """Protocol for ancestor index selection operations."""

    def __call__(self, key: KeyArray, weights: Array, pivot: int) -> tuple[int, Any]:
        """
        Selects an ancestor index, potentially using an MCMC move around a pivot.

        Args:
            key: JAX PRNG key.
            weights: Normalized weights of the particles.
            pivot: The current index to move from.

        Returns:
            A tuple containing the new index and any auxiliary output from the move.
        """
        ...
