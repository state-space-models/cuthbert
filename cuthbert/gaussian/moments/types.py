from typing import Protocol

from cuthbert.gaussian.types import LinearizedKalmanFilterState
from cuthbertlib.linearize.moments import MeanAndCholCovFunc
from cuthbertlib.types import Array, ArrayTreeLike


class GetDynamicsMoments(Protocol):
    def __call__(
        self,
        state: LinearizedKalmanFilterState,
        model_inputs: ArrayTreeLike,
    ) -> tuple[MeanAndCholCovFunc, Array]:
        """
        Get dynamics conditional mean and (generalised) Cholesky covariance
            function and linearization point.

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
    def __call__(
        self, state: LinearizedKalmanFilterState, model_inputs: ArrayTreeLike
    ) -> tuple[MeanAndCholCovFunc, Array, Array]:
        """
        Get observation conditional mean, (generalised) Cholesky covariance function,
            linearization point and the observation from model inputs.

        `associative_scan` only supported when `state` is ignored.

        Args:
            state: NamedTuple containing `mean` and `mean_prev` attributes.
            model_inputs: Model inputs.

        Returns:
            Tuple with conditional mean, (generalised) Cholesky covariance
                and observation.
        """
        ...
