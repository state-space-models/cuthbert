from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import Array, random, tree

from cuthbert.smc.types import InitSample, LogPotential, PropagateSample
from cuthbertlib.resampling import Resampling
from cuthbertlib.smc.ess import log_ess
from cuthbertlib.types import ArrayTree, ArrayTreeLike, KeyArray, ScalarArray


class MarginalParticleFilterState(NamedTuple):
    # no ancestors, as it does not make sense for marginal particle filters
    key: KeyArray
    particles: ArrayTree
    log_weights: Array
    model_inputs: ArrayTreeLike
    log_likelihood: ScalarArray


def init_prepare(
    model_inputs: ArrayTreeLike,
    init_sample: InitSample,
    n_filter_particles: int,
    key: KeyArray | None = None,
) -> MarginalParticleFilterState:
    """
    Prepare the initial state for the particle filter.

    Args:
        model_inputs: Model inputs.
        init_sample: Function to sample from the initial distribution M_0(x_0).
        n_filter_particles: Number of particles to sample.
        key: JAX random key.

    Returns:
        Initial state for the filter.

    Raises:
        ValueError: If `key` is None.
    """
    if key is None:
        raise ValueError("A JAX PRNG key must be provided.")
    keys = random.split(key, n_filter_particles)
    particles = jax.vmap(init_sample, (0, None))(keys, model_inputs)
    return MarginalParticleFilterState(
        key=key,
        particles=particles,
        log_weights=jnp.zeros(n_filter_particles),
        model_inputs=model_inputs,
        log_likelihood=jnp.array(0.0),
    )


def filter_prepare(
    model_inputs: ArrayTreeLike,
    init_sample: InitSample,
    n_filter_particles: int,
    key: KeyArray | None = None,
) -> MarginalParticleFilterState:
    """
    Prepare a state for a particle filter step.

    Args:
        model_inputs: Model inputs.
        init_sample: Function to sample from the initial distribution M_0(x_0).
            Only used to infer particle shapes.
        n_filter_particles: Number of particles for the filter.
        key: JAX random key.

    Returns:
        Prepared state for the filter.

    Raises:
        ValueError: If `key` is None.
    """
    if key is None:
        raise ValueError("A JAX PRNG key must be provided.")
    dummy_particle = jax.eval_shape(init_sample, key, model_inputs)
    particles = tree.map(
        lambda x: jnp.empty((n_filter_particles,) + x.shape), dummy_particle
    )
    return MarginalParticleFilterState(
        key=key,
        particles=particles,
        log_weights=jnp.zeros(n_filter_particles),
        model_inputs=model_inputs,
        log_likelihood=jnp.array(0.0),
    )


def filter_combine(
    state_1: MarginalParticleFilterState,
    state_2: MarginalParticleFilterState,
    propagate_sample: PropagateSample,
    log_potential: LogPotential,
    resampling_fn: Resampling,
    ess_threshold: float,
) -> MarginalParticleFilterState:
    """
    Combine the filter state from the previous time step with the state prepared
    for the current step.

    Implements the particle filter update: conditional resampling,
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
    N = state_1.log_weights.shape[0]
    keys = random.split(state_1.key, N + 1)

    # Resample
    prev_log_weights = state_1.log_weights
    ancestor_indices, log_weights = jax.lax.cond(
        log_ess(state_1.log_weights) < jnp.log(ess_threshold * N),
        lambda: (resampling_fn(keys[0], state_1.log_weights, N), jnp.zeros(N)),
        lambda: (jnp.arange(N), state_1.log_weights),
    )
    ancestors = tree.map(lambda x: x[ancestor_indices], state_1.particles)

    # Propagate
    next_particles = jax.vmap(propagate_sample, (0, 0, None))(
        keys[1:], ancestors, state_2.model_inputs
    )

    # N^2 Reweight by comparing all ancestors with all next particles
    log_potential_vmapped = jax.vmap(
        jax.vmap(log_potential, (0, None, None), out_axes=0),
        (None, 0, None),
        out_axes=0,
    )

    log_potentials = log_potential_vmapped(
        state_1.particles, next_particles, state_2.model_inputs
    )
    next_log_weights = log_potentials + prev_log_weights[None, :]
    next_log_weights = jax.nn.logsumexp(next_log_weights, axis=1)

    # Compute the log likelihood
    logsum_weights = jax.nn.logsumexp(next_log_weights)
    log_likelihood_incr = logsum_weights - jax.nn.logsumexp(log_weights)
    log_likelihood = log_likelihood_incr + state_1.log_likelihood

    return MarginalParticleFilterState(
        state_2.key,
        next_particles,
        next_log_weights,
        state_2.model_inputs,
        log_likelihood,
    )
