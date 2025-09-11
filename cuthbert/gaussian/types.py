from typing import Protocol

from cuthbertlib.types import (
    Array,
    ArrayTreeLike,
    LogConditionalDensity,
    LogDensity,
)
from cuthbertlib.linearize.moments import MeanAndCholCovFunc


### Kalman types
class GetInitParams(Protocol):
    def __call__(self, model_inputs: ArrayTreeLike) -> tuple[Array, Array]:
        """Get initial parameters (m0, chol_P0) from model inputs."""
        ...


class GetDynamicsParams(Protocol):
    def __call__(self, model_inputs: ArrayTreeLike) -> tuple[Array, Array, Array]:
        """Get dynamics parameters (F, c, chol_Q) from model inputs."""
        ...


class GetObservationParams(Protocol):
    def __call__(
        self, model_inputs: ArrayTreeLike
    ) -> tuple[Array, Array, Array, Array]:
        """Get observation parameters (H, d, chol_R, y) from model inputs."""
        ...


### Moments types
class GetDynamicsMoments(Protocol):
    def __call__(
        self,
        state: ArrayTreeLike,
        model_inputs: ArrayTreeLike,
    ) -> tuple[MeanAndCholCovFunc, Array]:
        """
        Get dynamics conditional mean and (generalised) Cholesky covariance
            function and linearization point.

        Args:
            state: Algorithmic NamedTuple containing `mean` and `mean_prev` attributes.
            model_inputs: Model inputs.

        Returns:
            Tuple with dynamics conditional mean and (generalised) Cholesky covariance
                function and linearization point.
        """
        ...


class GetObservationMoments(Protocol):
    def __call__(
        self, state: ArrayTreeLike, model_inputs: ArrayTreeLike
    ) -> tuple[MeanAndCholCovFunc, Array, Array]:
        """
        Get observation conditional mean, (generalised) Cholesky covariance function,
            linearization point and the observation from model inputs.

        Args:
            state: Algorithmic NamedTuple containing `mean` and `mean_prev` attributes.
            model_inputs: Model inputs.

        Returns:
            Tuple with conditional mean, (generalised) Cholesky covariance
                and observation.
        """
        ...


### Taylor types
LogPotential = LogDensity


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
        self, state: ArrayTreeLike, model_inputs: ArrayTreeLike
    ) -> tuple[LogConditionalDensity, Array, Array]:
        """Get the dynamics log density and linearization points
        (for the previous and current time points)

        Args:
            state: Algorithmic NamedTuple containing `mean` and `mean_prev` attributes.
            model_inputs: Model inputs.

        Returns:
            Tuple with dynamics log density and linearization points.
        """
        ...


class GetObservationFunc(Protocol):
    def __call__(
        self, state: ArrayTreeLike, model_inputs: ArrayTreeLike
    ) -> tuple[LogConditionalDensity, Array, Array] | tuple[LogPotential, Array]:
        """Extract observation function, linearization point and optional observation.
        State is the predicted state after applying the Kalman dynamics propagation.

        Two types of output are supported:
        - Observation log density function log p(y | x) and points x and y
            to linearize around.
        - Log potential function log G(x) and a linearization point x.

        Args:
            state: Algorithmic NamedTuple containing `mean` and `mean_prev` attributes.
                Predicted state after applying the Kalman dynamics propagation.
            model_inputs: Model inputs.

        Returns:
            Either a tuple with observation function to linearize, linearization point
                and observation, or a tuple with log potential function and linearization
                point.
        """
        ...
