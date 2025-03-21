from typing import NamedTuple, Protocol
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


class Update(Protocol):
    def __call__(
        self,
        state: ArrayTreeLike,
        observation: ArrayTreeLike,
        inputs: ArrayTreeLike,
        key: KeyArray | None = None,
    ) -> tuple[ArrayTree, ArrayTree]: ...


class Filter(Protocol):
    def __call__(
        self,
        observations: ArrayTreeLike,
        inputs: ArrayTreeLike,
        key: KeyArray | None = None,
    ) -> tuple[ArrayTree, ArrayTree]: ...


class Smoother(Protocol):
    def __call__(
        self,
        filter_states: ArrayTreeLike,
        inputs: ArrayTreeLike,
        key: KeyArray | None = None,
    ) -> tuple[ArrayTree, ArrayTree]: ...


class SSMInference(NamedTuple):
    init: Init
    predict: Predict
    update: Update
    filter: Filter
    smoother: Smoother
