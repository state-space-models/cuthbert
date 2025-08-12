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
    ) -> tuple[ArrayTree, Array]: ...
