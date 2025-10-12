from typing import Protocol

from cuthbertlib.types import Array, ArrayTreeLike


class GetInitDist(Protocol):
    def __call__(self, model_inputs: ArrayTreeLike) -> Array:
        """Get the initial distribution.

        Should return an array $m$ of shape (N,) where N is the number of states,
        with $m_i = p(x_0 = i)$.
        """
        ...


class GetTransitionMatrix(Protocol):
    def __call__(self, model_inputs: ArrayTreeLike) -> Array:
        r"""Get the transition matrix.

        Should return an array $A$ of shape (N, N) where N is the number of
        states, with $A_{ij} = p(x_t = j \mid x_{t-1} = i)$.
        """
        ...


class GetObsLogLikelihoods(Protocol):
    def __call__(self, model_inputs: ArrayTreeLike) -> Array:
        r"""Get the observation log likelihoods.

        Should return an array $b$ of shape (N,) where N is the number of states,
        with $b_i = \log p(y_t \mid x_t = i)$.
        """
        ...
