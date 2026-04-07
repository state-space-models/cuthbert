"""Provides types for EnKF callback functions."""

from typing import Protocol

from cuthbertlib.types import Array, ArrayTreeLike, KeyArray


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


class InitSample(Protocol):
    """Protocol for sampling from the initial distribution."""

    def __call__(self, key: KeyArray, model_inputs: ArrayTreeLike) -> Array:
        """Sample from the initial distribution.

        Args:
            key: JAX PRNG key.
            model_inputs: Model inputs.

        Returns:
            Sample from the initial distribution, shape (x_dim,).
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
