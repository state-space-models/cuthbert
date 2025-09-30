from typing import NamedTuple, Protocol

from cuthbert.utils import dummy_tree_like
from cuthbertlib.kalman import filtering
from cuthbertlib.linearize.moments import MeanAndCholCovFunc
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


# class LinearizedKalmanFilterState(NamedTuple):
#     mean: Array
#     chol_cov: Array
#     log_likelihood: Array
#     model_inputs: ArrayTree
#     mean_prev: Array


### Shared state type for linearized Kalman filters
class LinearizedKalmanFilterState(NamedTuple):
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


def linearized_kalman_filter_state_dummy_elem(
    mean: Array,
    chol_cov: Array,
    log_likelihood: Array,
    model_inputs: ArrayTree,
    mean_prev: Array,
) -> LinearizedKalmanFilterState:
    return LinearizedKalmanFilterState(
        elem=filtering.FilterScanElement(
            A=dummy_tree_like(chol_cov),
            b=mean,
            U=chol_cov,
            eta=dummy_tree_like(mean),
            Z=dummy_tree_like(chol_cov),
            ell=log_likelihood,
        ),
        model_inputs=model_inputs,
        mean_prev=mean_prev,
    )


### Moments types
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
