from typing import Any, NamedTuple, Protocol
from cuthbert.types import (
    ArrayTree,
    ArrayTreeLike,
    ScalarArray,
    ScalarArrayLike,
    KeyArray,
)


class InitLogDensity(Protocol):
    def __call__(
        self,
        t: ScalarArrayLike,
        u: ArrayTreeLike,
        x: ArrayTreeLike,
        static_params: ArrayTreeLike,
        extra: ArrayTreeLike,
    ) -> ScalarArray: ...


class InitSample(Protocol):
    def __call__(
        self,
        t: ScalarArrayLike,
        u: ArrayTreeLike,
        static_params: ArrayTreeLike,
        extra: Any,
        key: KeyArray,
    ) -> ArrayTree: ...


class DynamicsLogDensity(Protocol):
    def __call__(
        self,
        t_prev: ScalarArrayLike,
        x_prev: ArrayTreeLike,
        t: ScalarArrayLike,
        u: ArrayTreeLike,
        x: ArrayTreeLike,
        static_params: ArrayTreeLike,
        extra: Any,
    ) -> ScalarArray: ...


class DynamicsSample(Protocol):
    def __call__(
        self,
        t_prev: ScalarArrayLike,
        x_prev: ArrayTreeLike,
        t: ScalarArrayLike,
        u: ArrayTreeLike,
        static_params: ArrayTreeLike,
        extra: Any,
        key: KeyArray,
    ) -> ArrayTree: ...


class ObservationLogDensity(Protocol):
    def __call__(
        self,
        t: ScalarArrayLike,
        u: ArrayTreeLike,
        x: ArrayTreeLike,
        y: ArrayTreeLike,
        static_params: ArrayTreeLike,
        extra: Any,
    ) -> ScalarArray: ...


class ObservationSample(Protocol):
    def __call__(
        self,
        t: ScalarArrayLike,
        u: ArrayTreeLike,
        x: ArrayTreeLike,
        static_params: ArrayTreeLike,
        extra: Any,
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
