"""Provides types for representing discrete HMMs."""

from typing import Protocol

from cuthbertlib.types import Array, ArrayTreeLike


class GetInitDist(Protocol):
    """Protocol for specifying the initial distribution."""

    def __call__(self, model_inputs: ArrayTreeLike) -> Array:
        """Get the initial distribution.

        Args:
            model_inputs: Model inputs.

        Returns:
            An array $m$ of shape (N,) where N is the number of states,
                with $m_i = p(x_0 = i)$.
        """
        ...


class GetTransitionMatrix(Protocol):
    """Protocol for specifying the transition matrix."""

    def __call__(self, model_inputs: ArrayTreeLike) -> Array:
        r"""Get the transition matrix.


        Args:
            model_inputs: Model inputs.

        Returns:
            An array $A$ of shape (N, N) where N is the number of
                states, with $A_{ij} = p(x_t = j \mid x_{t-1} = i)$.
        """
        ...


class GetObsLogLikelihoods(Protocol):
    """Protocol for specifying the observation log likelihoods."""

    def __call__(self, model_inputs: ArrayTreeLike) -> Array:
        r"""Get the observation log likelihoods.

        Args:
            model_inputs: Model inputs.

        Returns:
            An array $b$ of shape (N,) where N is the number of states,
                with $b_i = \log p(y_t \mid x_t = i)$.
        """
        ...
