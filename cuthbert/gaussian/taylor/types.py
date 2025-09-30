from typing import Protocol, TypeAlias

from cuthbert.gaussian.types import (
    LinearizedKalmanFilterState,
)
from cuthbertlib.types import (
    Array,
    ArrayTreeLike,
    LogConditionalDensity,
    LogDensity,
)

LogPotential: TypeAlias = LogDensity


class GetInitLogDensity(Protocol):
    def __call__(self, model_inputs: ArrayTreeLike) -> tuple[LogDensity, Array]:
        """Get the initial log density and initial linearization point.

        Args:
            model_inputs: Model inputs.

        Returns:
            Tuple with initial log density and initial linearization point.
        """
        ...


class GetDynamicsLogDensity(Protocol):
    def __call__(
        self,
        state: LinearizedKalmanFilterState,
        model_inputs: ArrayTreeLike,
    ) -> tuple[LogConditionalDensity, Array, Array]:
        """Get the dynamics log density and linearization points
        (for the previous and current time points)

        `associative_scan` only supported when `state` is ignored.

        Args:
            state: NamedTuple containing `mean` and `mean_prev` attributes.
            model_inputs: Model inputs.

        Returns:
            Tuple with dynamics log density and linearization points.
        """
        ...


class GetObservationFunc(Protocol):
    def __call__(
        self,
        state: LinearizedKalmanFilterState,
        model_inputs: ArrayTreeLike,
    ) -> tuple[LogConditionalDensity, Array, Array] | tuple[LogPotential, Array]:
        """Extract observation function, linearization point and optional observation.
        State is the predicted state after applying the Kalman dynamics propagation.

        `associative_scan` only supported when `state` is ignored.

        Two types of output are supported:
        - Observation log density function log p(y | x) and points x and y
            to linearize around.
        - Log potential function log G(x) and a linearization point x.

        Args:
            state: NamedTuple containing `mean` and `mean_prev` attributes.
                Predicted state after applying the Kalman dynamics propagation.
            model_inputs: Model inputs.

        Returns:
            Either a tuple with observation function to linearize, linearization point
                and observation, or a tuple with log potential function and linearization
                point.
        """
        ...
