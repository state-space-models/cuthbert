"""Provides types for the moment-based linearization of Gaussian state-space models."""

from typing import Protocol

from cuthbert.gaussian.types import LinearizedKalmanFilterState
from cuthbertlib.linearize.moments import MeanAndCholCovFunc
from cuthbertlib.types import Array, ArrayTreeLike


class GetDynamicsMoments(Protocol):
    """Protocol for extracting the dynamics specifications."""

    def __call__(
        self,
        state: LinearizedKalmanFilterState,
        model_inputs: ArrayTreeLike,
    ) -> tuple[MeanAndCholCovFunc, Array]:
        """Get dynamics conditional mean and chol_cov function and linearization point.

        `associative_scan` only supported when `state` is ignored.

        Args:
            state: NamedTuple containing `mean` and `mean_prev` attributes.
            model_inputs: Model inputs.

        Returns:
            Tuple with dynamics conditional mean and (generalised) Cholesky covariance
                function and linearization point.
        """
        ...


class GetObservationMoments(Protocol):
    """Protocol for extracting the observation specifications."""

    def __call__(
        self, state: LinearizedKalmanFilterState, model_inputs: ArrayTreeLike
    ) -> tuple[MeanAndCholCovFunc, Array, Array]:
        """Get conditional mean and chol_cov function, linearization point and observation.

        `associative_scan` only supported when `state` input is ignored.

        Args:
            state: NamedTuple containing `mean` and `mean_prev` attributes.
            model_inputs: Model inputs.

        Returns:
            Tuple with conditional mean and chol_cov function, linearization point
                and observation.
        """
        ...
