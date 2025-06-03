from typing import NamedTuple, Protocol
from cuthbertlib.types import (
    ArrayTree,
    ArrayTreeLike,
    KeyArray,
)


class FilterPrepare(Protocol):
    def __call__(
        self,
        model_inputs: ArrayTreeLike,
        key: KeyArray | None = None,
    ) -> ArrayTree: ...


class FilterCombine(Protocol):
    def __call__(
        self,
        state_1: ArrayTreeLike,
        state_2: ArrayTreeLike,
    ) -> ArrayTree: ...


class SmootherPrepare(Protocol):
    def __call__(
        self,
        filter_state: ArrayTreeLike,
        model_inputs: ArrayTreeLike,
        key: KeyArray | None = None,
    ) -> ArrayTree: ...


class SmootherCombine(Protocol):
    def __call__(
        self,
        state_1: ArrayTreeLike,
        state_2: ArrayTreeLike,
    ) -> ArrayTree: ...


class SSMInference(NamedTuple):
    FilterPrepare: FilterPrepare
    FilterCombine: FilterCombine
    SmootherPrepare: SmootherPrepare
    SmootherCombine: SmootherCombine
    associative_filter: bool = False
    associative_smoother: bool = False
