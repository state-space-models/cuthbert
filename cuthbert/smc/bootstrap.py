from functools import partial
from typing import NamedTuple, Protocol

import jax
import jax.numpy as jnp
from jax import Array, random, tree

from cuthbert.inference import Inference
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


class GTSmootherState(NamedTuple):
    particles: ArrayTree
    ancestor_indices: Array


def build(
    init_sample: InitSample,
    propagate_sample: PropagateSample,
    log_potential: LogPotential,
    n_filter_particles: int,
    n_smoother_particles: int,
    resampling_fn: Resampling,
    ess_threshold: float,
) -> Inference:
    """
    Build a bootstrap particle filter inference object for general state-space models.

    Args:
        init_sample: Function to sample from the initial distribution M_0(x_0).
        propagate_sample: Function to sample from the Markov kernel M_t(x_t | x_{t-1}).
        log_potential: Function to compute the log potential function log G_t(x_{t-1}, x_t).
            Typically the log observation likelihood log p(y_t | x_t).
        n_filter_particles: Number of particles for the filter.
        n_smoother_particles: Number of particles for the smoother.
        resampling_fn: Resampling algorithm to use (e.g., systematic, multinomial).
        ess_threshold: Effective sample size threshold for triggering resampling.
            Resampling occurs when ESS < ess_threshold * n_filter_particles.

    Returns:
        Inference object for bootstrap particle filter and genealogy tracking smoother.
            Associative scan is not supported.
    """
    return Inference(
        init_prepare=partial(
            init_prepare, init_sample=init_sample, n_filter_particles=n_filter_particles
        ),
        filter_prepare=partial(
            filter_prepare,
            init_sample=init_sample,
            n_filter_particles=n_filter_particles,
        ),
        filter_combine=partial(
            filter_combine,
            propagate_sample=propagate_sample,
            log_potential=log_potential,
            resampling_fn=resampling_fn,
            ess_threshold=ess_threshold,
        ),
        smoother_prepare=partial(
            smoother_prepare, n_smoother_particles=n_smoother_particles
        ),
        smoother_combine=smoother_combine,
        convert_filter_to_smoother_state=partial(
            convert_filter_to_smoother_state, n_smoother_particles=n_smoother_particles
        ),
        associative_filter=False,
        associative_smoother=False,
    )


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
        ess_threshold: ESS threshold as fraction of particle count for triggering resampling.

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


def log_ess(log_weights: Array) -> Array:
    """Compute the logarithm of the effective sample size (ESS).

    Args:
        log_weights: Array of log weights for the particles.
    """
    return 2 * jax.nn.logsumexp(log_weights) - jax.nn.logsumexp(2 * log_weights)


def convert_filter_to_smoother_state(
    filter_state: BootstrapFilterState,
    n_smoother_particles: int,
) -> GTSmootherState:
    """
    Convert the final filter state to initial smoother state.

    Samples particles from the final filter distribution to initialize
    the backward smoother. The sampling is done according to the
    normalized filter weights.

    Args:
        filter_state: Final filter state containing particles and log weights.
        n_smoother_particles: Number of particles for the smoother.

    Returns:
        Initial smoother state with sampled particles and corresponding
        ancestor indices from the filter state.
    """
    weights = jax.nn.softmax(filter_state.log_weights)
    # Note: The key for the final filter state is used here.
    # It must not have been used before in the `filter_combine` function.
    indices = random.choice(
        filter_state.key,
        filter_state.particles.shape[0],
        (n_smoother_particles,),
        p=weights,
    )
    return GTSmootherState(
        filter_state.particles[indices], filter_state.ancestor_indices[indices]
    )


def smoother_prepare(
    filter_state: BootstrapFilterState,
    model_inputs: ArrayTreeLike,
    n_smoother_particles: int,
    key: KeyArray | None = None,
) -> GTSmootherState:
    """
    Prepare a state for a genealogy tracking smoother step.

    Args:
        filter_state: Filter state used to infer particle shapes.
        model_inputs: Model inputs for the current time step (not used).
        n_smoother_particles: Number of particles for the smoother.
        key: JAX random key (not used).

    Returns:
        Prepared smoother state with empty particle and ancestor index arrays.
    """
    dummy_smoothed_particles = tree.map(
        lambda x: jnp.empty((n_smoother_particles,) + x.shape[1:]),
        filter_state.particles,
    )
    dummy_ancestor_indices = jnp.empty((n_smoother_particles,), dtype=int)
    return GTSmootherState(dummy_smoothed_particles, dummy_ancestor_indices)


def smoother_combine(
    state_1: GTSmootherState,
    state_2: GTSmootherState,
) -> GTSmootherState:
    """
    Combine step for the genealogy tracking smoother.

    Performs the backward pass of the smoother by tracing particle genealogies.
    The smoother iterates backwards in time, using ancestor indices to
    reconstruct the particle trajectories.

    Args:
        state_1: Prepared state at time t.
        state_2: Smoother state at time t + 1.

    Returns:
        Smoother state at time t.
    """
    particles = state_1.particles[state_2.ancestor_indices]
    ancestor_indices = state_1.ancestor_indices[state_2.ancestor_indices]
    return GTSmootherState(particles, ancestor_indices)
