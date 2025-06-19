from typing import NamedTuple, Protocol

import jax
import jax.numpy as jnp
from jax import Array, random, tree

from cuthbertlib.resampling import Resampling
from cuthbertlib.types import ArrayTree, ArrayTreeLike, KeyArray, ScalarArray


class InitSample(Protocol):
    """Get a sample from the initial distribution :math:`M_0(x_0)`."""

    def __call__(self, key: KeyArray, model_inputs: ArrayTreeLike) -> ArrayTree: ...


class PropagateSample(Protocol):
    """Sample from the Markov kernel :math:`M_t(x_t \\mid x_{t-1})`."""

    def __call__(
        self, key: KeyArray, state: ArrayTreeLike, model_inputs: ArrayTreeLike
    ) -> ArrayTree: ...


class TransitionDensity(Protocol):
    """Compute the transition density :math:`\\log M_t(x_t \\mid x_{t-1})`."""

    def __call__(
        self,
        state_prev: ArrayTreeLike,
        state: ArrayTreeLike,
        model_inputs: ArrayTreeLike,
    ) -> ScalarArray: ...


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


class SmootherState(NamedTuple):
    particles: ArrayTree
    filtered_particles: ArrayTree
    filtered_log_weights: Array


def init_prepare(
    model_inputs: ArrayTreeLike,
    init_sample: InitSample,
    num_samples: int,
    key: KeyArray,
) -> BootstrapFilterState:
    keys = random.split(key, num_samples)
    particles = jax.vmap(init_sample, (0, None))(keys, model_inputs)
    return BootstrapFilterState(
        key=key,
        particles=particles,
        log_weights=jnp.zeros(num_samples),
        ancestor_indices=jnp.arange(num_samples),
        model_inputs=model_inputs,
        log_likelihood=jnp.array(0.0),
    )


def filter_prepare(
    model_inputs: ArrayTreeLike,
    init_sample: InitSample,
    num_samples: int,
    key: KeyArray,
) -> BootstrapFilterState:
    dummy_particle = jax.eval_shape(init_sample, key, model_inputs)
    particles = tree.map(lambda x: jnp.empty((num_samples,) + x.shape), dummy_particle)
    return BootstrapFilterState(
        key=key,
        particles=particles,
        log_weights=jnp.zeros(num_samples),
        ancestor_indices=jnp.arange(num_samples),
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
    N = state_1.particles.shape[0]
    keys = random.split(state_1.key, N + 1)

    # Resample
    ancestor_indices, log_weights = jax.lax.cond(
        log_ess(state_1.log_weights) < jnp.log(ess_threshold),
        lambda: (resampling_fn(keys[0], state_1.log_weights, N), jnp.zeros(N)),
        lambda: (jnp.arange(N), state_1.log_weights),
    )
    ancestors = tree.map(lambda x: x[ancestor_indices], state_1.particles)

    # Propagate
    next_particles = jax.vmap(propagate_sample, (0, 0, None))(
        keys[1:], ancestors, state_1.model_inputs
    )

    # Reweight
    log_potentials = jax.vmap(log_potential, (0, 0, None))(
        ancestors, next_particles, state_1.model_inputs
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


def log_ess(log_weights: Array) -> Array:
    """Logarithm of the effective sample size."""
    return 2 * jax.nn.logsumexp(log_weights) - jax.nn.logsumexp(2 * log_weights)


def convert_filter_to_smoother_state(
    filter_state: BootstrapFilterState,
    n_smoother_particles: int,
    key: KeyArray,
) -> SmootherState:
    weights = jax.nn.softmax(filter_state.log_weights)
    indices = random.choice(
        key, filter_state.particles.shape[0], (n_smoother_particles,), p=weights
    )
    return SmootherState(
        filter_state.particles[indices],
        filter_state.particles,
        filter_state.log_weights,
    )


def smoother_prepare(
    filter_state: BootstrapFilterState,
    model_inputs: ArrayTreeLike,
    n_smoother_particles: int,
    key: KeyArray,
) -> SmootherState:
    dummy_smoothed_particles = tree.map(
        lambda x: jnp.empty((n_smoother_particles,) + x.shape[1:]),
        filter_state.particles,
    )
    return SmootherState(
        dummy_smoothed_particles,
        filter_state.particles,
        filter_state.log_weights,
    )


def smoother_combine(
    state_1: SmootherState,
    state_2: SmootherState,
    transition_density: TransitionDensity,
    key: KeyArray,
) -> SmootherState:
    """Backward sampling.

    Args:
        state_1: Output of `smoother_prepare` at time t.
        state_2: Smoother state at time t + 1.
        transition_density: The state transition density function.
        key: JAX random key.
    """
    ...
