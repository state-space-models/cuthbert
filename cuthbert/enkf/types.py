"""Provides types for EnKF callback functions."""

from typing import Protocol

from cuthbertlib.enkf.filtering import DynamicsFn, ObservationFn
from cuthbertlib.types import Array, ArrayTreeLike, KeyArray


class InitSample(Protocol):
    """Protocol for sampling from the initial distribution."""

    def __call__(self, key: KeyArray) -> Array:
        """Sample from the initial distribution.

        Args:
            key: JAX PRNG key.

        Returns:
            Sample from the initial distribution, shape (x_dim,).
        """
        ...


class GetEnKFDynamics(Protocol):
    """Protocol for getting dynamics function that describes the general simulator p(x_{t+1} | x_t)."""

    def __call__(self, model_inputs: ArrayTreeLike) -> DynamicsFn:
        """Get dynamics function that describes the general simulator p(x_{t+1} | x_t) from model inputs.

        Args:
            model_inputs: Model inputs.

        Returns:
            Dynamics function that describes the general simulator (x_t, key) -> x_{t+1} ~ p(x_{t+1} | x_t).
        """
        ...


class GetEnKFObservations(Protocol):
    """Protocol for getting observation function, observation noise Cholesky factor chol_R, and observation vector y for an EnKF. i.e., for state space x_t, y_t = h(x_t, model_inputs) + R_t, where R_t ~ N(0, chol_R), return (h, chol_R, y)."""

    def __call__(
        self, model_inputs: ArrayTreeLike
    ) -> tuple[ObservationFn, Array, Array]:
        """Get observation function, observation noise Cholesky factor chol_R, and observation vector y from model inputs.

        Args:
            model_inputs: Model inputs.

        Returns:
            Tuple with observation function, Cholesky factor of the observation noise covariance, and observation vector.
            observation noise covariance and y is the observation vector.
        """
        ...
