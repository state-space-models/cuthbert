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
    """Build a bootstrap particle filter inference object."""
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
) -> GTSmootherState:
    weights = jax.nn.softmax(filter_state.log_weights)
    # TODO: The key for the final filter state is used here.
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
    """Combine step for the genealogy tracking smoother.

    Args:
        state_1: Output of `smoother_prepare` at time t.
        state_2: Smoother state at time t + 1.

    Returns:
        Smoother state at time t.
    """
    particles = state_1.particles[state_2.ancestor_indices]
    ancestor_indices = state_1.ancestor_indices[state_2.ancestor_indices]
    return GTSmootherState(particles, ancestor_indices)
