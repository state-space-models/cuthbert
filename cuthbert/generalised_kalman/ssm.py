from typing import Protocol, NamedTuple
from jax import Array

from cuthbert.types import ArrayTreeLike


class InitParams(Protocol):
    def __call__(
        self,
        inputs: ArrayTreeLike,
    ) -> tuple[Array, Array]:
        """
        Returns the mean and Cholesky factor of the initial state distribution.

        Args:
            inputs: Inputs to the model.

        Returns:
            A tuple of the mean and Cholesky factor of the initial state distribution.
        """
        ...


class DynamicsParams(Protocol):
    def __call__(
        self,
        mean: Array,
        chol_cov: Array,
        inputs: ArrayTreeLike,
    ) -> tuple[Array, Array, Array]:
        """
        Returns the state transition matrix, shift vector, and Cholesky factor of the
        state transition noise covariance.

        Defines Gaussian dynamics p(x_{t+1} | x_t, inputs) = N(F_t x_t + c_t, Q_t)

        The mean and Cholesky factor of the previous state $x_t$ are provided which
        is required for linearization or sigma point selection.

        Args:
            mean: Mean of the current state.
            chol_cov: Cholesky factor of the current state covariance.
            inputs: Inputs to the model.

        Returns:
            A tuple of the state transition matrix, shift vector, and Cholesky factor of
            the state transition noise covariance.
        """
        ...


class LikelihoodParams(Protocol):
    def __call__(
        self,
        mean: Array,
        chol_cov: Array,
        observation: ArrayTreeLike,
        inputs: ArrayTreeLike,
    ) -> tuple[Array, Array, Array]:
        """
        Returns the observation matrix, shift vector, and Cholesky factor of the
        observation noise covariance.

        Defines Gaussian likelihood p(y_t | x_t, inputs) = N(H_t x_t + d_t, R_t)

        The mean and Cholesky factor of the current state $x_t$ are provided which
        is required for linearization or sigma point selection.

        Args:
            mean: Mean of the current state.
            chol_cov: Cholesky factor of the current state covariance.
            observation: Observation at the current time step.
            inputs: Inputs to the model.

        Returns:
            A tuple of the observation matrix, shift vector, and Cholesky factor of
            the observation noise covariance.
        """
        ...


class LinearGaussianSSM(NamedTuple):
    init_params: InitParams
    dynamics_params: DynamicsParams
    likelihood_params: LikelihoodParams
