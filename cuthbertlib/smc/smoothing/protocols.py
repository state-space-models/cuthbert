"""Shared protocols for backward smoothing functions in SMC."""

from typing import Protocol, runtime_checkable

from cuthbertlib.types import (
    Array,
    ArrayLike,
    ArrayTree,
    ArrayTreeLike,
    KeyArray,
    LogConditionalDensity,
)


@runtime_checkable
class BackwardSampling(Protocol):
    """Protocol for backward sampling functions."""

    def __call__(
        self,
        key: KeyArray,
        x0_all: ArrayTreeLike,
        x1_all: ArrayTreeLike,
        log_weight_x0_all: ArrayLike,
        log_density: LogConditionalDensity,
        x1_ancestor_indices: ArrayLike,
    ) -> tuple[ArrayTree, Array]:
        """Performs a backward sampling step.

        Samples a collection of $x_0$ that combine with the provided $x_1$ to
        give a collection of pairs $(x_0, x_1)$ from the smoothing distribution.

        Args:
            key: JAX PRNG key.
            x0_all: A collection of previous states $x_0$.
            x1_all: A collection of current states $x_1$.
            log_weight_x0_all: The log weights of $x_0$.
            log_density: The log density function of $x_1$ given $x_0$.
            x1_ancestor_indices: The ancestor indices of $x_1$.

        Returns:
            A collection of samples $x_0$ and their sampled indices.
        """
        ...
