from functools import partial
from typing import NamedTuple, cast

import jax
import jax.numpy as jnp
from jax import Array, random

from cuthbert.inference import Smoother
from cuthbert.smc.particle_filter import LogPotential, ParticleFilterState
from cuthbert.utils import dummy_tree_like
from cuthbertlib.resampling import Resampling
from cuthbertlib.smc.smoothing.protocols import BackwardSampling
from cuthbertlib.types import ArrayTree, ArrayTreeLike, KeyArray


class ParticleSmootherState(NamedTuple):
    key: KeyArray
    particles: ArrayTree
    ancestor_indices: Array
    model_inputs: ArrayTree
    log_weights: Array

    @property
    def n_particles(self) -> int:
        """Number of particles in the smoother state."""
        return self.ancestor_indices.shape[-1]


def build_smoother(
    log_potential: LogPotential,
    backward_sampling_fn: BackwardSampling,
    resampling_fn: Resampling,
    n_smoother_particles: int,
) -> Smoother:
    """
    Build a particle smoother object.

    Args:
        log_potential: Function to compute the JOINT log potential log G_t(x_{t-1}, x_t) + log M_t(x_t | x_{t-1}).
        backward_sampling_fn: Backward sampling algorithm to use (e.g., genealogy tracing, exact backward sampling).
            This choice specifies how to sample x_{t-1} ~ p(x_{t-1} | x_t, y_{0:t-1}) given
            samples x_{t} ~ p(x_t | y_{0:T}). See `cuthbertlib/smc/smoothing/` for possible choices.
        resampling_fn: Resampling algorithm to use (e.g., multinomial, systematic).
        n_smoother_particles: Number of samples to draw from the backward sampling algorithm.

    Returns:
        Particle smoother object.
    """
    return Smoother(
        convert_filter_to_smoother_state=partial(
            convert_filter_to_smoother_state,
            resampling=resampling_fn,
            n_smoother_particles=n_smoother_particles,
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
    n_smoother_particles: int,
    model_inputs: ArrayTreeLike | None = None,
    key: KeyArray | None = None,
) -> ParticleSmootherState:
    """
    Convert a particle filter state to a particle smoother state.

    Args:
        filter_state: Particle filter state.
        resampling: Resampling algorithm to use (e.g., multinomial, systematic).
        n_smoother_particles: Number of smoother samples to draw.
        model_inputs: Only used to create an empty model_inputs tree
            (the values are ignored).
            Useful so that the final smoother state has the same structure as the rest.
            By default, filter_state.model_inputs is used. So this
            is only needed if the smoother model_inputs have a different tree
            structure to filter_state.model_inputs.
        key: JAX random key.

    Returns:
        Particle smoother state. Note that the model_inputs are set to dummy values.

    Raises:
        ValueError: If key is None.
    """
    if key is None:
        raise ValueError("A JAX PRNG key must be provided.")

    if model_inputs is None:
        model_inputs = filter_state.model_inputs

    dummy_model_inputs = dummy_tree_like(model_inputs)

    key, resampling_key = random.split(key)
    indices = resampling(resampling_key, filter_state.log_weights, n_smoother_particles)

    return ParticleSmootherState(
        key=cast(KeyArray, key),
        particles=jax.tree.map(lambda z: z[indices], filter_state.particles),
        ancestor_indices=filter_state.ancestor_indices[indices],
        model_inputs=dummy_model_inputs,
        log_weights=-jnp.log(n_smoother_particles) * jnp.ones(n_smoother_particles),
    )


def smoother_prepare(
    filter_state: ParticleFilterState,
    model_inputs: ArrayTreeLike,
    key: KeyArray | None = None,
) -> ParticleSmootherState:
    """
    Prepare a state for a particle smoother step.

    Note that the model_inputs here are different to filter_state.model_inputs.
    The model_inputs required here are for the transition from t to t+1.
    filter_state.model_inputs represents the transition from t-1 to t.

    Args:
        filter_state: Particle filter state from time t.
        model_inputs: Model inputs for the transition from t to t+1.
        key: JAX random key.

    Returns:
        Prepared state for the particle smoother.
    """
    if key is None:
        raise ValueError("A JAX PRNG key must be provided.")

    model_inputs = jax.tree.map(lambda x: jnp.asarray(x), model_inputs)

    return ParticleSmootherState(
        key,
        filter_state.particles,
        filter_state.ancestor_indices,
        model_inputs,
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

    Args:
        state_1: State prepared with model inputs at time t.
        state_2: Smoother state at time t + 1.
        backward_sampling_fn: Function to perform backward sampling from the joint distribution.
        log_potential: Function to compute log potential.

    Returns:
        Combined particle smoother state.
        Contains particles, the original ancestor indices of the particles, and model inputs.
    """
    new_particles_1, ancestors_1 = backward_sampling_fn(
        state_1.key,
        x0_all=state_1.particles,
        x1_all=state_2.particles,
        log_weight_x0_all=state_1.log_weights,
        log_density=lambda s1, s2: log_potential(s1, s2, state_2.model_inputs),
        x1_ancestor_indices=state_2.ancestor_indices,
    )

    n_particles = len(ancestors_1)
    log_weights = -jnp.log(n_particles) * jnp.ones(n_particles)
    new_state = ParticleSmootherState(
        key=state_1.key,
        particles=new_particles_1,
        ancestor_indices=state_1.ancestor_indices[ancestors_1],
        model_inputs=state_1.model_inputs,
        log_weights=log_weights,
    )
    return new_state
