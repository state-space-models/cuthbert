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
        u: ArrayTreeLike,
        x: ArrayTreeLike,
    ) -> ScalarArray: ...


class InitSample(Protocol):
    def __call__(
        self,
        u: ArrayTreeLike,
        key: KeyArray,
    ) -> ArrayTree: ...


class DynamicsLogDensity(Protocol):
    def __call__(
        self,
        x_prev: ArrayTreeLike,
        u: ArrayTreeLike,
    ) -> ScalarArray: ...


class DynamicsSample(Protocol):
    def __call__(
        self,
        x_prev: ArrayTreeLike,
        u: ArrayTreeLike,
        key: KeyArray,
    ) -> ArrayTree: ...


class ObservationLogDensity(Protocol):
    def __call__(
        self,
        u: ArrayTreeLike,
        x: ArrayTreeLike,
        y: ArrayTreeLike,
    ) -> ScalarArray: ...


class ObservationSample(Protocol):
    def __call__(
        self,
        u: ArrayTreeLike,
        x: ArrayTreeLike,
        key: KeyArray,
    ) -> ArrayTree: ...


class StateSpaceModel(
    NamedTuple
):  # Not all inference methods will require all components, if not required set to RaiseNotImplementedError function
    init_log_density: InitLogDensity
    init_sample: InitSample
    dynamics_log_density: DynamicsLogDensity
    dynamics_sample: DynamicsSample
    observation_log_density: ObservationLogDensity
    observation_sample: ObservationSample
