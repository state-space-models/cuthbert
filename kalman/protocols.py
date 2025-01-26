from typing import Protocol, Any
from jax import Array

ArrayTree = Any
ArrayTreeLike = Any


class LinearGaussianInit(Protocol):
    def __call__(self, inputs: ArrayTreeLike) -> tuple[Array, Array]:
        """Generates the mean and covariance of the initial Gaussian.

        Defines the initial state distribution N(x | mean, covariance).

        Args:
            inputs: The input PyTree (with ArrayLike leaves) to the initial state.

        Returns:
            Tuple with array mean and covariance defining N(x | mean, covariance).

        """
        ...


class LinearGaussianDynamics(Protocol):
    def __call__(
        self,
        inputs: ArrayTreeLike,
    ) -> tuple[Array, Array, Array]:
        """Generates the shift, matrix and covariance of the dynamics.

        Defines the state transition distribution
        N(x | shift + matrix @ x_prev, covariance).

        Args:
            inputs: The input PyTree (with ArrayLike leaves) to the dynamics.

        Returns:
            Tuple with array shift, matrix and covariance
                defining N(x | shift + matrix @ x_prev, covariance).
        """
        ...


class LinearGaussianObservation(Protocol):
    def __call__(self, inputs: ArrayTreeLike) -> tuple[Array, Array, Array]:
        """Generates the shift, matrix and covariance of the Gaussian observation.

        Defines the observation distribution N(y | shift + matrix @ x, covariance).

        Args:
            inputs: The input PyTree (with ArrayLike leaves) to the observation
                distribution.

        Returns:
            Tuple with array shift, matrix and covariance
                defining N(y | shift + matrix @ x, covariance).
        """
        ...
