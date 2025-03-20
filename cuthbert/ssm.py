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
        state_prev: ArrayTreeLike,
        inputs: ArrayTreeLike,
    ) -> ScalarArray: ...


class DynamicsSample(Protocol):
    def __call__(
        self,
        key: KeyArray,
        state_prev: ArrayTreeLike,
        inputs: ArrayTreeLike,
    ) -> ArrayTree: ...


class LikelihoodLogDensity(Protocol):
    def __call__(
        self,
        state: ArrayTreeLike,
        observation: ArrayTreeLike,
        inputs: ArrayTreeLike,
    ) -> ScalarArray: ...


class LikelihoodSample(Protocol):
    def __call__(
        self,
        key: KeyArray,
        state: ArrayTreeLike,
        inputs: ArrayTreeLike,
    ) -> ArrayTree: ...


class SSM(NamedTuple):
    init_log_density: InitLogDensity
    init_sample: InitSample
    dynamics_log_density: DynamicsLogDensity
    dynamics_sample: DynamicsSample
    likelihood_log_density: LikelihoodLogDensity
    likelihood_sample: LikelihoodSample
