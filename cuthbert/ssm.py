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
        x: ArrayTreeLike,
        u: ArrayTreeLike,
    ) -> ScalarArray: ...


class InitSample(Protocol):
    def __call__(
        self,
        key: KeyArray,
        u: ArrayTreeLike,
    ) -> ArrayTree: ...


class DynamicsLogDensity(Protocol):
    def __call__(
        self,
        x: ArrayTreeLike,
        x_traj: ArrayTreeLike,
        u: ArrayTreeLike,
    ) -> ScalarArray: ...


class DynamicsSample(Protocol):
    def __call__(
        self,
        key: KeyArray,
        x_traj: ArrayTreeLike,
        u: ArrayTreeLike,
    ) -> ArrayTree: ...


class LogLikelihood(Protocol):
    def __call__(
        self,
        x_traj: ArrayTreeLike,
        u: ArrayTreeLike,
    ) -> ScalarArray: ...


class FeynmanKac(NamedTuple):
    init_log_density: InitLogDensity
    init_sample: InitSample
    dynamics_log_density: DynamicsLogDensity
    dynamics_sample: DynamicsSample
    log_likelihood: LogLikelihood
