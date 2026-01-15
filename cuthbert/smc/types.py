r"""Provides types for representing generic Feynman--Kac models.

$$
\mathbb{Q}_{t}(x_{0:t}) \propto \mathbb{M}_0(x_0) \, G_0(x_0) \prod_{s=1}^{t} M_s(x_s \mid x_{s-1}) \, G_s(x_{s-1}, x_s).
$$
"""

from typing import Protocol

from cuthbertlib.types import ArrayTree, ArrayTreeLike, KeyArray, ScalarArray


class InitSample(Protocol):
    """Protocol for  sampling from the initial distribution $M_0(x_0)$."""

    def __call__(self, key: KeyArray, model_inputs: ArrayTreeLike) -> ArrayTree:
        """Samples from the initial distribution $M_0(x_0)$.

        Args:
            key: JAX PRNG key.
            model_inputs: Model inputs.

        Returns:
            A sample $x_0$.
        """
        ...


class PropagateSample(Protocol):
    r"""Protocol for sampling from the Markov kernel $M_t(x_t \mid x_{t-1})$."""

    def __call__(
        self, key: KeyArray, state: ArrayTreeLike, model_inputs: ArrayTreeLike
    ) -> ArrayTree:
        r"""Samples from the Markov kernel $M_t(x_t \mid x_{t-1})$.

        Args:
            key: JAX PRNG key.
            state: State at the previous step $x_{t-1}$.
            model_inputs: Model inputs.

        Returns:
            A sample $x_t$.
        """
        ...


class LogPotential(Protocol):
    r"""Protocol for computing the log potential function $\log G_t(x_{t-1}, x_t)$."""

    def __call__(
        self,
        state_prev: ArrayTreeLike,
        state: ArrayTreeLike,
        model_inputs: ArrayTreeLike,
    ) -> ScalarArray:
        r"""Computes the log potential function $\log G_t(x_{t-1}, x_t)$.

        Args:
            state_prev: State at the previous step $x_{t-1}$.
            state: State at the current step $x_{t}$.
            model_inputs: Model inputs.

        Returns:
            A scalar value $\log G_t(x_{t-1}, x_t)$.
        """
        ...
