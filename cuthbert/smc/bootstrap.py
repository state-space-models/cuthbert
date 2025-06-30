from typing import NamedTuple, Protocol

import jax
import jax.numpy as jnp
from jax import Array, random, tree

from cuthbertlib.resampling import Resampling
from cuthbertlib.smc.ess import log_ess
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


class BootstrapFilterState(NamedTuple):
    key: KeyArray
    particles: ArrayTree
    log_weights: Array
    ancestor_indices: Array
    model_inputs: ArrayTreeLike
    log_likelihood: ScalarArray


def init_prepare(
    model_inputs: ArrayTreeLike,
    init_sample: InitSample,
    n_filter_particles: int,
    key: KeyArray | None = None,
) -> BootstrapFilterState:
    """
    Prepare the initial state for the bootstrap particle filter.

    Args:
        model_inputs: Model inputs.
        init_sample: Function to sample from the initial distribution M_0(x_0).
        n_filter_particles: Number of particles to sample.
        key: JAX random key.

    Returns:
        Initial state for the bootstrap filter.

    Raises:
        ValueError: If `key` is None.
    """
    if key is None:
        raise ValueError("A JAX PRNG key must be provided.")
    keys = random.split(key, n_filter_particles)
    particles = jax.vmap(init_sample, (0, None))(keys, model_inputs)
    return BootstrapFilterState(
        key=key,
        particles=particles,
        log_weights=jnp.zeros(n_filter_particles),
        ancestor_indices=jnp.arange(n_filter_particles),
        model_inputs=model_inputs,
        log_likelihood=jnp.array(0.0),
    )


def filter_prepare(
    model_inputs: ArrayTreeLike,
    init_sample: InitSample,
    n_filter_particles: int,
    key: KeyArray | None = None,
) -> BootstrapFilterState:
    """
    Prepare a state for a bootstrap particle filter step.

    Args:
        model_inputs: Model inputs.
        init_sample: Function to sample from the initial distribution M_0(x_0).
            Only used to infer particle shapes.
        n_filter_particles: Number of particles for the filter.
        key: JAX random key.

    Returns:
        Prepared state for the bootstrap filter.

    Raises:
        ValueError: If `key` is None.
    """
    if key is None:
        raise ValueError("A JAX PRNG key must be provided.")
    dummy_particle = jax.eval_shape(init_sample, key, model_inputs)
    particles = tree.map(
        lambda x: jnp.empty((n_filter_particles,) + x.shape), dummy_particle
    )
    return BootstrapFilterState(
        key=key,
        particles=particles,
        log_weights=jnp.zeros(n_filter_particles),
        ancestor_indices=jnp.arange(n_filter_particles),
        model_inputs=model_inputs,
        log_likelihood=jnp.array(0.0),
    )


def filter_combine(
    state_1: BootstrapFilterState,
    state_2: BootstrapFilterState,
    propagate_sample: PropagateSample,
    log_potential: LogPotential,
    resampling_fn: Resampling,
    ess_threshold: float,
) -> BootstrapFilterState:
    """
    Combine the filter state from the previous time step with the state prepared
    for the current step.

    Implements the bootstrap particle filter update: conditional resampling,
    propagation through state dynamics, and reweighting based on the potential function.

    Args:
        state_1: Filter state from the previous time step.
        state_2: Filter state prepared for the current step.
        propagate_sample: Function to sample from the Markov kernel M_t(x_t | x_{t-1}).
        log_potential: Function to compute the log potential log G_t(x_{t-1}, x_t).
        resampling_fn: Resampling algorithm to use (e.g., systematic, multinomial).
        ess_threshold: Fraction of particle count specifying when to resample.
            Resampling is triggered when the effective sample size (ESS) < ess_threshold * N.

    Returns:
        The filtered state at the current time step.
    """
    N = state_1.particles.shape[0]
    keys = random.split(state_1.key, N + 1)
    n_filter_particles = state_1.particles.shape[0]

    # Resample
    ancestor_indices, log_weights = jax.lax.cond(
        log_ess(state_1.log_weights) < jnp.log(ess_threshold * n_filter_particles),
        lambda: (resampling_fn(keys[0], state_1.log_weights, N), jnp.zeros(N)),
        lambda: (jnp.arange(N), state_1.log_weights),
    )
    ancestors = tree.map(lambda x: x[ancestor_indices], state_1.particles)

    # Propagate
    next_particles = jax.vmap(propagate_sample, (0, 0, None))(
        keys[1:], ancestors, state_2.model_inputs
    )

    # Reweight
    log_potentials = jax.vmap(log_potential, (0, 0, None))(
        ancestors, next_particles, state_2.model_inputs
    )
    next_log_weights = log_potentials + log_weights

    # Compute the log likelihood
    logsum_weights = jax.nn.logsumexp(next_log_weights)
    log_likelihood_incr = logsum_weights - jax.nn.logsumexp(log_weights)
    log_likelihood = log_likelihood_incr + state_1.log_likelihood

    return BootstrapFilterState(
        state_2.key,
        next_particles,
        next_log_weights,
        ancestor_indices,
        state_2.model_inputs,
        log_likelihood,
    )
