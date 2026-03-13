from functools import partial

import jax
import jax.numpy as jnp
from jax import random

from cuthbert.inference import Filter
from cuthbert.smc.particle_filter import ParticleFilterState
from cuthbert.smc.types import InitSample, LogPotential, PropagateSample
from cuthbert.utils import dummy_tree_like
from cuthbertlib.resampling.protocols import ConditionalResampling
from cuthbertlib.types import ArrayTreeLike, KeyArray


def build_csmc_filter(
    init_sample: InitSample,
    propagate_sample: PropagateSample,
    log_potential: LogPotential,
    n_particles: int,
    resampling_fn: ConditionalResampling,
) -> Filter:
    """Builds a conditional particle filter object.
    Args:
        init_sample: Function to sample from the initial distribution.
        propagate_sample: Function to sample from the Markov kernel.
        log_potential: Function to compute the log potential.
        n_particles: Number of particles for the filter.
        resampling_fn: Conditional resampling algorithm to use.
    Returns:
        Filter object for the conditional particle filter.
    """
    return Filter(
        init_prepare=partial(
            init_prepare,
            init_sample=init_sample,
            log_potential=log_potential,
            n_particles=n_particles,
        ),
        filter_prepare=partial(
            filter_prepare,
            init_sample=init_sample,
            n_particles=n_particles,
        ),
        filter_combine=partial(
            filter_combine,
            propagate_sample=propagate_sample,
            log_potential=log_potential,
            resampling_fn=resampling_fn,
        ),
        associative=False,
    )


def init_prepare(
    model_inputs: ArrayTreeLike,
    init_sample: InitSample,
    log_potential: LogPotential,
    n_particles: int,
    key: KeyArray | None = None,
) -> ParticleFilterState:
    """Prepare the initial state for the conditional particle filter."""
    if key is None:
        raise ValueError("A JAX PRNG key must be provided.")

    # Sample
    keys = random.split(key, n_particles)
    particles = jax.vmap(init_sample, (0, None))(keys, model_inputs)

    # Pin reference particle
    _, reference_particle, reference_index = model_inputs
    particles = particles.at[reference_index].set(reference_particle)

    # Weight
    log_weights = jax.vmap(log_potential, (None, 0, None))(
        None, particles, model_inputs
    )

    # Compute the log normalizing constant
    log_normalizing_constant = jax.nn.logsumexp(log_weights) - jnp.log(n_particles)

    return ParticleFilterState(
        key=key,
        particles=particles,
        log_weights=log_weights,
        ancestor_indices=jnp.arange(n_particles),
        model_inputs=model_inputs,
        log_normalizing_constant=log_normalizing_constant,
    )


def filter_prepare(
    model_inputs: ArrayTreeLike,
    init_sample: InitSample,
    n_particles: int,
    key: KeyArray | None = None,
) -> ParticleFilterState:
    """Prepare a state for a conditional particle filter step."""
    if key is None:
        raise ValueError("A JAX PRNG key must be provided.")
    dummy_particle = jax.eval_shape(init_sample, key, model_inputs)
    particles = jax.tree.map(
        lambda x: jnp.empty((n_particles,) + x.shape), dummy_particle
    )
    particles = dummy_tree_like(particles)
    return ParticleFilterState(
        key=key,
        particles=particles,
        log_weights=jnp.zeros((n_particles, 1)),
        ancestor_indices=jnp.arange(n_particles),
        model_inputs=model_inputs,
        log_normalizing_constant=jnp.array(0.0),
    )


def filter_combine(
    state_1: ParticleFilterState,
    state_2: ParticleFilterState,
    propagate_sample: PropagateSample,
    log_potential: LogPotential,
    resampling_fn: ConditionalResampling,
    ess_threshold: float,
) -> ParticleFilterState:
    """Combine previous filter state with the state prepared for the current step."""
    n_particles = state_1.log_weights.shape[0]
    keys = random.split(state_1.key, n_particles + 1)

    # Get conditional info from states
    _, _, prev_ref_idx = state_1.model_inputs
    _, current_ref_particle, current_ref_idx = state_2.model_inputs

    # Resample
    # Here we assume that if conditional is True, a ConditionalResampling function is provided.
    ancestor_indices = resampling_fn(
        keys[0], state_1.log_weights, n_particles, prev_ref_idx, current_ref_idx
    )

    ancestors = jax.tree.map(lambda x: x[ancestor_indices], state_1.particles)
    log_weights = jnp.zeros((n_particles, 1))  # Reset weights after resampling

    # Propagate
    next_particles = jax.vmap(propagate_sample, (0, 0, None))(
        keys[1:], ancestors, state_2.model_inputs
    )

    # Pin reference particle
    next_particles = next_particles.at[current_ref_idx].set(current_ref_particle)

    # Reweight
    log_potentials = jax.vmap(log_potential, (0, 0, None))(
        ancestors, next_particles, state_2.model_inputs
    )
    next_log_weights = log_weights + log_potentials

    # Compute the log normalizing constant
    logsum_weights = jax.nn.logsumexp(next_log_weights)
    log_normalizing_constant_incr = logsum_weights - jnp.log(n_particles)
    log_normalizing_constant = (
        log_normalizing_constant_incr + state_1.log_normalizing_constant
    )

    return ParticleFilterState(
        key=state_2.key,
        particles=next_particles,
        log_weights=next_log_weights,
        ancestor_indices=ancestor_indices,
        model_inputs=state_2.model_inputs,
        log_normalizing_constant=log_normalizing_constant,
    )
