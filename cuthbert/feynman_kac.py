from typing import NamedTuple, Protocol
from cuthbert.types import (
    ArrayTree,
    ArrayTreeLike,
    ScalarArray,
    KeyArray,
)


class InitLogDensity(Protocol):
    def __call__(
        self,
        state: ArrayTreeLike,
        inputs: ArrayTreeLike,
    ) -> ScalarArray: ...


class InitSample(Protocol):
    def __call__(
        self,
        key: KeyArray,
        inputs: ArrayTreeLike,
    ) -> ArrayTree: ...


class DynamicsLogDensity(Protocol):
    def __call__(
        self,
        state: ArrayTreeLike,
        trajectory: ArrayTreeLike,
        inputs: ArrayTreeLike,
    ) -> ScalarArray: ...


class DynamicsSample(Protocol):
    def __call__(
        self,
        key: KeyArray,
        trajectory: ArrayTreeLike,
        inputs: ArrayTreeLike,
    ) -> ArrayTree: ...


class LogLikelihood(Protocol):
    def __call__(
        self,
        trajectory: ArrayTreeLike,
        inputs: ArrayTreeLike,
    ) -> ScalarArray: ...


class FeynmanKac(NamedTuple):
    init_log_density: InitLogDensity
    init_sample: InitSample
    dynamics_log_density: DynamicsLogDensity
    dynamics_sample: DynamicsSample
    log_likelihood: LogLikelihood
