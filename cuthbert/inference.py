from typing import NamedTuple, Protocol

from cuthbert.types import (
    ArrayTree,
    ArrayTreeLike,
    KeyArray,
)
from cuthbert.utils import not_implemented

Info = Any


class Predict(Protocol):
    def __call__(
            self,
            state_prev: ArrayTreeLike,
            inputs: ArrayTreeLike,
            key: KeyArray | None = None,
    ) -> ArrayTree: ...


class FilterInit(Protocol):
    def __call__(
            self,
            inputs: ArrayTreeLike,
            key: KeyArray | None = None,
    ) -> ArrayTree: ...


class Update(Protocol):
    def __call__(
            self,
            state: ArrayTreeLike,
            inputs: ArrayTreeLike,
            key: KeyArray | None = None,
    ) -> tuple[ArrayTree, Info]: ...


class FilterCombine(Protocol):
    def __call__(
            self,
            state: ArrayTreeLike,
            observation: ArrayTreeLike,
            inputs: ArrayTreeLike,
            key: KeyArray | None = None,
    ) -> tuple[ArrayTree, Info]: ...


class SmootherInit(Protocol):
    def __call__(
            self,
            filter_state: ArrayTreeLike,
            inputs: ArrayTreeLike,
            key: KeyArray | None = None,
    ) -> ArrayTree: ...


class SmootherCombine(Protocol):
    def __call__(
            self,
            state_1: ArrayTreeLike,
            state_2: ArrayTreeLike,
            inputs: ArrayTreeLike,
            key: KeyArray | None = None,
    ) -> tuple[ArrayTree, Info]: ...


class Inference(NamedTuple):
    """
    Container for the inference procedures of a given method.

    Attributes:
        predict: Function to predict the next state given the previous state and inputs.
        update: Function to update the state with an observation and inputs.
        filter_init: Function to initialize the filter state.
        smoother_init: Function to initialize the smoother state.

        filter_combine: Function to combine two filter states.
        smoother_combine: Function to combine two smoother states.

        associative (optional): Whether the inference method is associative (default is False).
    """
    predict: Predict
    update: Update

    filter_init: FilterInit
    smoother_init: SmootherInit

    filter_combine: FilterCombine
    smoother_combine: SmootherCombine

    associative: bool = False
