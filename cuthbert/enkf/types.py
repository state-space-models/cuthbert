"""Provides types for EnKF callback functions."""

from typing import Protocol

from cuthbertlib.types import Array, ArrayTreeLike, KeyArray


class DynamicsFn(Protocol):
    """Protocol for the dynamics function of an EnKF."""

    def __call__(self, state: Array, model_inputs: ArrayTreeLike) -> Array:
        """Apply dynamics to a single state vector. i.e., for state space x_{t+1} = f(state, model_inputs) + Q_t, where Q_t ~ N(0, chol_Q), return f(x_t, model_inputs).

        Args:
            state: State vector, shape (x_dim,).
            model_inputs: Model inputs.

        Returns:
            Propagated state vector, shape (x_dim,).
        """
        ...


class ObservationFn(Protocol):
    """Protocol for the observation function of an EnKF. i.e., for state space x_t, y_t = h(x_t, model_inputs) + R_t, where R_t ~ N(0, chol_R), return h(x_t, model_inputs)."""

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


class GetEnKFDynamics(Protocol):
    """Protocol for getting dynamics function and dynamics noise parameters for an EnKF. i.e., for state space x_t+1 = f(x_t, model_inputs) + Q_t, where Q_t ~ N(0, chol_Q), return (f, chol_Q)."""

    def __call__(self, model_inputs: ArrayTreeLike) -> tuple[DynamicsFn, Array]:
        """Get dynamics function and dynamics noise Cholesky factor chol_Q from model inputs.

        Args:
            model_inputs: Model inputs.

        Returns:
            Tuple with dynamics function and Cholesky factor of the dynamics noise covariance, shape (x_dim, x_dim).
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
