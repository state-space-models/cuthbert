"""Implements the marginal particle filter.

See [Klaas et. al. (2005)](https://www.cs.ubc.ca/~arnaud/klass_defreitas_doucet_marginalparticlefilterUAI2005.pdf)
"""

from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import Array, random, tree

from cuthbert.inference import Filter
from cuthbert.smc.types import InitSample, LogPotential, PropagateSample
from cuthbert.utils import dummy_tree_like
from cuthbertlib.resampling import Resampling
from cuthbertlib.smc.ess import log_ess
from cuthbertlib.types import ArrayTree, ArrayTreeLike, KeyArray, ScalarArray


class MarginalParticleFilterState(NamedTuple):
    """Marginal particle filter state."""

    # no ancestors, as it does not make sense for marginal particle filters
    key: KeyArray
    particles: ArrayTree
    log_weights: Array
    model_inputs: ArrayTree
    log_normalizing_constant: ScalarArray


def build_filter(
    init_sample: InitSample,
    propagate_sample: PropagateSample,
    log_potential: LogPotential,
    n_filter_particles: int,
    resampling_fn: Resampling,
    ess_threshold: float,
) -> Filter:
    r"""Builds a marginal particle filter object.

    Args:
        init_sample: Function to sample from the initial distribution $M_0(x_0)$.
        propagate_sample: Function to sample from the Markov kernel $M_t(x_t \mid x_{t-1})$.
        log_potential: Function to compute the log potential $\log G_t(x_{t-1}, x_t)$.
        n_filter_particles: Number of particles for the filter.
        resampling_fn: Resampling algorithm to use (e.g., systematic, multinomial).
        ess_threshold: Fraction of particle count specifying when to resample.
            Resampling is triggered when the
            effective sample size (ESS) < ess_threshold * n_filter_particles.

    Returns:
        Filter object for the particle filter.
    """
    return Filter(
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
        associative=False,
    )


def init_prepare(
    model_inputs: ArrayTreeLike,
    init_sample: InitSample,
    n_filter_particles: int,
    key: KeyArray | None = None,
) -> MarginalParticleFilterState:
    """Prepare the initial state for the marginal particle filter.

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
    model_inputs = tree.map(lambda x: jnp.asarray(x), model_inputs)
    if key is None:
        raise ValueError("A JAX PRNG key must be provided.")

    # Sample
    keys = random.split(key, n_filter_particles)
    particles = jax.vmap(init_sample, (0, None))(keys, model_inputs)

    # Weight
    log_weights = jnp.zeros(n_filter_particles)

    # Compute the log normalizing constant
    log_normalizing_constant = jax.nn.logsumexp(log_weights) - jnp.log(
        n_filter_particles
    )

    return MarginalParticleFilterState(
        key=key,
        particles=particles,
        log_weights=log_weights,
        model_inputs=model_inputs,
        log_normalizing_constant=log_normalizing_constant,
    )


def filter_prepare(
    model_inputs: ArrayTreeLike,
    init_sample: InitSample,
    n_filter_particles: int,
    key: KeyArray | None = None,
) -> MarginalParticleFilterState:
    """Prepare a state for a marginal particle filter step.

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
    model_inputs = tree.map(lambda x: jnp.asarray(x), model_inputs)
    if key is None:
        raise ValueError("A JAX PRNG key must be provided.")
    dummy_particle = jax.eval_shape(init_sample, key, model_inputs)
    particles = tree.map(
        lambda x: jnp.empty((n_filter_particles,) + x.shape), dummy_particle
    )
    particles = dummy_tree_like(particles)
    return MarginalParticleFilterState(
        key=key,
        particles=particles,
        log_weights=jnp.zeros(n_filter_particles),
        model_inputs=model_inputs,
        log_normalizing_constant=jnp.array(0.0),
    )


def filter_combine(
    state_1: MarginalParticleFilterState,
    state_2: MarginalParticleFilterState,
    propagate_sample: PropagateSample,
    log_potential: LogPotential,
    resampling_fn: Resampling,
    ess_threshold: float,
) -> MarginalParticleFilterState:
    """Combine previous filter state with the state prepared for the current step.

    Implements the marginal particle filter update: conditional resampling,
    propagation through state dynamics, and N^2 reweighting based on the
    potential function.

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
    prev_log_weights = state_1.log_weights - jax.nn.logsumexp(
        state_1.log_weights
    )  # Ensure normalized
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

    # Compute the log normalizing constant
    logsum_weights = jax.nn.logsumexp(next_log_weights)
    log_normalizing_constant_incr = logsum_weights - jax.nn.logsumexp(log_weights)
    log_normalizing_constant = (
        log_normalizing_constant_incr + state_1.log_normalizing_constant
    )

    return MarginalParticleFilterState(
        key=state_2.key,
        particles=next_particles,
        log_weights=next_log_weights,
        model_inputs=state_2.model_inputs,
        log_normalizing_constant=log_normalizing_constant,
    )
