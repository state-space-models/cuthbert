from typing import NamedTuple, Protocol, TypeAlias

from cuthbertlib.linearize.moments import MeanAndCholCovFunc
from cuthbertlib.kalman import filtering
from cuthbertlib.types import (
    Array,
    ArrayTree,
    ArrayTreeLike,
)


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


### Shared types for linearized Kalman filters
class LinearizedKalmanFilterState(NamedTuple):
    mean: Array
    chol_cov: Array
    log_likelihood: Array
    model_inputs: ArrayTree
    mean_prev: Array


class AssociativeLinearizedKalmanFilterState(NamedTuple):
    elem: filtering.FilterScanElement
    model_inputs: ArrayTree
    mean_prev: Array

    @property
    def mean(self) -> Array:
        return self.elem.b

    @property
    def chol_cov(self) -> Array:
        return self.elem.U

    @property
    def log_likelihood(self) -> Array:
        return self.elem.ell


class GetDynamicsMoments(Protocol):
    def __call__(
        self,
        state: LinearizedKalmanFilterState,
        model_inputs: ArrayTreeLike,
    ) -> tuple[MeanAndCholCovFunc, Array]:
        """
        Get dynamics conditional mean and (generalised) Cholesky covariance
            function and linearization point.

        Args:
            state: NamedTuple containing `mean` and `mean_prev` attributes.
            model_inputs: Model inputs.

        Returns:
            Tuple with dynamics conditional mean and (generalised) Cholesky covariance
                function and linearization point.
        """
        ...


class GetObservationMoments(Protocol):
    def __call__(
        self, state: LinearizedKalmanFilterState, model_inputs: ArrayTreeLike
    ) -> tuple[MeanAndCholCovFunc, Array, Array]:
        """
        Get observation conditional mean, (generalised) Cholesky covariance function,
            linearization point and the observation from model inputs.

        Args:
            state: NamedTuple containing `mean` and `mean_prev` attributes.
            model_inputs: Model inputs.

        Returns:
            Tuple with conditional mean, (generalised) Cholesky covariance
                and observation.
        """
        ...
