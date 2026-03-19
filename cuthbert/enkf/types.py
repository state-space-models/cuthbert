"""Provides types for EnKF callback functions."""

from typing import Protocol

from cuthbertlib.types import Array, ArrayTreeLike


class DynamicsFn(Protocol):
    """Protocol for the dynamics function of an EnKF."""

    def __call__(self, state: Array, model_inputs: ArrayTreeLike) -> Array:
        """Apply dynamics to a single state vector.

        Args:
            state: State vector, shape (x_dim,).
            model_inputs: Model inputs.

        Returns:
            Propagated state vector, shape (x_dim,).
        """
        ...


class ObservationFn(Protocol):
    """Protocol for the observation function of an EnKF."""

    def __call__(self, state: Array, model_inputs: ArrayTreeLike) -> Array:
        """Map a single state vector to observation space.

        Args:
            state: State vector, shape (x_dim,).
            model_inputs: Model inputs.

        Returns:
            Observation vector, shape (y_dim,).
        """
        ...


class GetEnKFInitParams(Protocol):
    """Protocol for getting initial distribution parameters for an EnKF."""

    def __call__(self, model_inputs: ArrayTreeLike) -> tuple[Array, Array]:
        """Get initial parameters (m0, chol_P0) from model inputs.

        Args:
            model_inputs: Model inputs.

        Returns:
            Tuple of (m0, chol_P0) where m0 is the initial mean and
            chol_P0 is the Cholesky factor of the initial covariance.
        """
        ...


class GetEnKFDynamicsParams(Protocol):
    """Protocol for getting dynamics noise parameters for an EnKF."""

    def __call__(self, model_inputs: ArrayTreeLike) -> Array:
        """Get dynamics noise Cholesky factor chol_Q from model inputs.

        Args:
            model_inputs: Model inputs.

        Returns:
            Cholesky factor of the dynamics noise covariance, shape (x_dim, x_dim).
        """
        ...


class GetEnKFObservationParams(Protocol):
    """Protocol for getting observation parameters for an EnKF."""

    def __call__(self, model_inputs: ArrayTreeLike) -> tuple[Array, Array]:
        """Get observation parameters (chol_R, y) from model inputs.

        Args:
            model_inputs: Model inputs.

        Returns:
            Tuple of (chol_R, y) where chol_R is the Cholesky factor of the
            observation noise covariance and y is the observation vector.
        """
        ...
