from typing import Protocol

from cuthbertlib.types import ArrayTree, ArrayTreeLike, KeyArray, ScalarArray


class InitSample(Protocol):
    """Get a sample from the initial distribution :math:`M_0(x_0)`."""

    def __call__(self, key: KeyArray, model_inputs: ArrayTreeLike) -> ArrayTree: ...


class PropagateSample(Protocol):
    """Sample from the Markov kernel :math:`M_t(x_t \\mid x_{t-1})`."""

    def __call__(
        self, key: KeyArray, state: ArrayTreeLike, model_inputs: ArrayTreeLike
    ) -> ArrayTree: ...


class LogPotential(Protocol):
    """Compute the log potential function :math:`\\log G_t(x_{t-1}, x_t)`."""

    def __call__(
        self,
        state_prev: ArrayTreeLike,
        state: ArrayTreeLike,
        model_inputs: ArrayTreeLike,
    ) -> ScalarArray: ...
