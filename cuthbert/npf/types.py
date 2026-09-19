"""Provides types for the nested particle filter."""

from typing import Protocol

from cuthbertlib.types import ArrayTree, ArrayTreeLike, KeyArray, ScalarArray


class SampleParam(Protocol):
    r"""Protocol for sampling from the initial distribution $\mu(\theta_0)$."""

    def __call__(self, key: KeyArray) -> ArrayTree:
        r"""Samples from the initial distribution $\mu(\theta_0)$.

        Args:
            key: JAX PRNG key.

        Returns:
            A sample $\theta_0$.
        """
        ...


class PropagateSample(Protocol):
    r"""Protocol for sampling from the Markov kernel $M_t(x_t \mid x_{t-1}, \theta)$."""

    def __call__(
        self,
        key: KeyArray,
        state: ArrayTreeLike,
        param: ArrayTreeLike,
        model_inputs: ArrayTreeLike,
    ) -> ArrayTree:
        r"""Samples from the Markov kernel $M_t(x_t \mid x_{t-1}, \theta)$.

        Args:
            key: JAX PRNG key.
            state: State at the previous step $x_{t-1}$.
            param: Hidden parameter $\theta$.
            model_inputs: Model inputs.

        Returns:
            A sample $x_t$.
        """
        ...


class LogPotential(Protocol):
    r"""Protocol for computing the log potential function $\log G_t(x_{t-1}, x_t, \theta)$."""

    def __call__(
        self,
        state_prev: ArrayTreeLike,
        state: ArrayTreeLike,
        param: ArrayTreeLike,
        model_inputs: ArrayTreeLike,
    ) -> ScalarArray:
        r"""Computes the log potential function $\log G_t(x_{t-1}, x_t, \theta)$.

        Args:
            state_prev: State at the previous step $x_{t-1}$.
            state: State at the current step $x_{t}$.
            param: Hidden parameter $\theta$.
            model_inputs: Model inputs.

        Returns:
            A scalar value $\log G_t(x_{t-1}, x_t, \theta)$.
        """
        ...


class JitteringKernel(Protocol):
    """Protocol for a jittering kernel used in the nested particle filter."""

    def __call__(self, key: KeyArray, particle: ArrayTree, **kwargs) -> ArrayTree:
        """Applies the jittering kernel to a particle.

        Args:
            key: JAX PRNG key.
            particle: Particle to be jittered.
            kwargs: Additional arguments for the jittering kernel.

        Returns:
            Jittered particle.
        """
        ...
