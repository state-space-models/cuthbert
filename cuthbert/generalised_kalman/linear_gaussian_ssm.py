from typing import Protocol, NamedTuple
from jax import Array

from cuthbert.types import ArrayTreeLike, KeyArray


class InitParams(Protocol):
    def __call__(
        self,
        inputs: ArrayTreeLike,
        key: KeyArray | None = None,
    ) -> tuple[Array, Array]:
        """
        Returns the mean and Cholesky factor of the initial state distribution.

        Args:
            inputs: Inputs to the model.
            key: Random key, only used for methods with random components.

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
        key: KeyArray | None = None,
    ) -> tuple[Array, Array, Array]:
        """
        Returns the state transition matrix, shift vector, and Cholesky factor of the
        state transition noise covariance.

        Defines Gaussian dynamics p(x_t | x_{t-1}, inputs_t) = N(F_t x_{t-1} + c_t, Q_t)

        The mean and Cholesky factor of the previous state $x_t$ are provided which
        is required for linearization or sigma point selection.

        Args:
            mean: Mean of the previous state.
            chol_cov: Cholesky factor of the previous state covariance.
            inputs: Inputs to the model.
            key: Random key, only used for methods with random components.

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
        key: KeyArray | None = None,
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
            key: Random key, only used for methods with random components.

        Returns:
            A tuple of the observation matrix, shift vector, and Cholesky factor of
            the observation noise covariance.
        """
        ...


class LinearGaussianSSM(NamedTuple):
    """
    Defines a conditionally linear Gaussian state space model.

    Init: p(x_0 | inputs_0) = N(m_0, Q_0)
    Dynamics: p(x_t | x_{t-1}, inputs_t) = N(F_t x_{t-1} + c_t, Q_t)
    Likelihood: p(y_t | x_t, inputs_t) = N(H_t x_t + d_t, R_t)

    Note that this framing can be used more generally than exact linear Gaussian models,
    including sigma point and linearization based approximations of non-linear models.

    Attributes:
        init_params: Initial parameters generator.
        dynamics_params: Dynamics parameters generator.
        likelihood_params: Likelihood parameters generator.
    """

    init_params: InitParams
    dynamics_params: DynamicsParams
    likelihood_params: LikelihoodParams
