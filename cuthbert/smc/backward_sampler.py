from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import Array, random

from cuthbert.inference import Smoother
from cuthbert.smc.particle_filter import LogPotential, ParticleFilterState
from cuthbertlib.resampling import Resampling
from cuthbertlib.smc.smoothing.protocols import BackwardSampling
from cuthbertlib.types import ArrayTree, ArrayTreeLike, KeyArray


class ParticleSmootherState(NamedTuple):
    key: KeyArray | None
    particles: ArrayTree
    ancestor_indices: Array
    model_inputs: ArrayTreeLike
    log_weights: Array

    @property
    def n_particles(self) -> int:
        """Number of particles in the smoother state."""
        return self.ancestor_indices.shape[-1]


def build_smoother(
    log_potential: LogPotential,
    backward_sampling_fn: BackwardSampling,
    resampling_fn: Resampling,
) -> Smoother:
    """
    Build a particle filter object.

    Args:
        log_potential: Function to compute the JOINT log potential log G_t(x_{t-1}, x_t) + log M_t(x_t | x_{t-1}).
        backward_sampling_fn: Backward sampling algorithm to use (e.g., exact backward sampling, IMH)
        resampling_fn: Resampling algorithm to use (e.g., multinomial, systematic).

    Returns:
        Smoother object for the particle smoother.
    """
    return Smoother(
        convert_filter_to_smoother_state=partial(
            convert_filter_to_smoother_state, resampling=resampling_fn
        ),
        smoother_prepare=smoother_prepare,
        smoother_combine=partial(
            smoother_combine,
            backward_sampling_fn=backward_sampling_fn,
            log_potential=log_potential,
        ),
        associative=False,
    )


def convert_filter_to_smoother_state(
    filter_state: ParticleFilterState,
    resampling: Resampling,
    key: KeyArray | None = None,
) -> ParticleSmootherState:
    """
    Convert a particle filter state to a particle smoother state.

    Args:
        filter_state: Particle filter state.
        resampling: Resampling algorithm to use (e.g., multinomial, systematic).
        key: JAX random key.

    Returns:
        Particle smoother state.

    Raises:
        ValueError: If key is None.
    """
    if key is None:
        raise ValueError("A JAX PRNG key must be provided.")
    key, resampling_key = random.split(key)
    indices = resampling(
        resampling_key, filter_state.log_weights, filter_state.n_particles
    )
    n_samples = indices.shape[0]

    return ParticleSmootherState(
        key=key,
        particles=jax.tree.map(lambda z: z[indices], filter_state.particles),
        ancestor_indices=filter_state.ancestor_indices[indices],
        model_inputs=filter_state.model_inputs,
        log_weights=-jnp.log(n_samples) * jnp.ones_like(filter_state.log_weights),
    )


def smoother_prepare(
    filter_state: ParticleFilterState,
    model_inputs: ArrayTreeLike | None,
    key: KeyArray | None = None,
) -> ParticleSmootherState:
    """
    Prepare the initial state for the particle filter.

    Args:
        filter_state: Particle filter state from the previous time step.
        model_inputs: Model inputs for the current time step if None, uses model inputs from filter state.
        key: JAX random key.

    Returns:
        Prepared state for the particle smoother.
    """
    if key is None:
        raise ValueError("A JAX PRNG key must be provided.")

    return ParticleSmootherState(
        key,
        filter_state.particles,
        filter_state.ancestor_indices,
        model_inputs if model_inputs is not None else filter_state.model_inputs,
        filter_state.log_weights,
    )


def smoother_combine(
    state_1: ParticleSmootherState,
    state_2: ParticleSmootherState,
    backward_sampling_fn: BackwardSampling,
    log_potential: LogPotential,
) -> ParticleSmootherState:
    """
    Combine smoother state from next time point with state prepared
    with latest model inputs.

    Remember smoothing iterates backwards in time.

    Applies backward sampling smoother update.

    Args:
        state_1: State prepared with model inputs at time t.
        state_2: Smoother state at time t + 1.
        backward_sampling_fn: Function to perform backward sampling from the joint distribution.
        log_potential: Function to compute log potential.

    Returns:
        Combined Particle smoother state.
        Contains particles, the original ancestor indices of the particles, and model inputs.
    """
    new_particles_1, ancestors_1 = backward_sampling_fn(
        state_1.key,
        x0_all=state_1.particles,
        x1_all=state_2.particles,
        log_weight_x0_all=state_1.log_weights,
        log_potential=lambda s1, s2: log_potential(s1, s2, state_2.model_inputs),
    )

    n_samples = state_1.n_particles
    log_weights = -jnp.log(n_samples) * jnp.ones_like(state_2.log_weights)
    new_state = ParticleSmootherState(
        key=state_1.key,
        particles=new_particles_1,
        ancestor_indices=state_1.ancestor_indices[ancestors_1],
        model_inputs=state_1.model_inputs,
        log_weights=log_weights,
    )
    return new_state
