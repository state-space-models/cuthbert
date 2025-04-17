from typing import NamedTuple, Never, Protocol, Callable
from cuthbert.types import (
    ArrayTree,
    ArrayTreeLike,
    KeyArray,
)


class Init(Protocol):
    def __call__(
        self,
        inputs: ArrayTreeLike,
        key: KeyArray | None = None,
    ) -> ArrayTree: ...


class Predict(Protocol):
    def __call__(
        self,
        state_prev: ArrayTreeLike,
        inputs: ArrayTreeLike,
        key: KeyArray | None = None,
    ) -> ArrayTree: ...


class FilterUpdate(Protocol):
    def __call__(
        self,
        state: ArrayTreeLike,
        observation: ArrayTreeLike,
        inputs: ArrayTreeLike,
        key: KeyArray | None = None,
    ) -> tuple[ArrayTree, ArrayTree]: ...


class SmootherCombine(Protocol):
    def __call__(
        self,
        state_1: ArrayTreeLike,
        state_2: ArrayTreeLike,
        inputs: ArrayTreeLike,
        key: KeyArray | None = None,
    ) -> tuple[ArrayTree, ArrayTree]: ...


class AssociativeFilterInit(Protocol):
    def __call__(
        self,
        observation: ArrayTreeLike,
        inputs: ArrayTreeLike,
        key: KeyArray | None = None,
    ) -> ArrayTree: ...


class AssociativeFilterCombine(Protocol):
    def __call__(
        self,
        state_1: ArrayTreeLike,
        state_2: ArrayTreeLike,
        key: KeyArray | None = None,
    ) -> tuple[ArrayTree, ArrayTree]: ...


class AssociativeSmootherInit(Protocol):
    def __call__(
        self,
        filter_state: ArrayTreeLike,
        inputs: ArrayTreeLike,
        key: KeyArray | None = None,
    ) -> ArrayTree: ...


class AssociativeSmootherCombine(Protocol):
    def __call__(
        self,
        state_1: ArrayTreeLike,
        state_2: ArrayTreeLike,
        key: KeyArray | None = None,
    ) -> tuple[ArrayTree, ArrayTree]: ...


def not_implemented(protocol) -> Callable:
    def f(*args, **kwargs) -> Never:
        raise NotImplementedError(f"{protocol.__name__} not implemented")

    return f


class Inference(NamedTuple):
    init: Init
    predict: Predict
    filter_update: FilterUpdate
    smoother_combine: SmootherCombine

    associative_filter_init: AssociativeFilterInit = not_implemented(
        AssociativeFilterInit
    )
    associative_filter_combine: AssociativeFilterCombine = not_implemented(
        AssociativeFilterCombine
    )
    associative_smoother_init: AssociativeSmootherInit = not_implemented(
        AssociativeSmootherInit
    )
    associative_smoother_combine: AssociativeSmootherCombine = not_implemented(
        AssociativeSmootherCombine
    )
