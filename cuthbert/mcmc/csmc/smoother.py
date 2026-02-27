"""Implements the backward pass for the Conditional Sequential Monte Carlo."""

from functools import partial

import jax
from jax import numpy as jnp

from cuthbert.smc.particle_filter import ParticleFilterState
from cuthbert.smc.types import LogPotential
from cuthbertlib.mcmc.protocols import AncestorMove
from cuthbertlib.resampling.utils import normalize
from cuthbertlib.types import KeyArray



def _backward_sampling_step(
    log_potential_fn: LogPotential,
    particle_t: jnp.ndarray,
    inp: tuple[ParticleFilterState, KeyArray],
):
    """A single step of the backward sampling pass."""
    state_t_minus_1, key_t = inp
    log_weights = log_potential_fn(
        state_t_minus_1.particles, particle_t, state_t_minus_1.model_inputs
    )
    log_weights -= jnp.max(log_weights)
    log_weights += state_t_minus_1.log_weights

    weights = normalize(log_weights)
    ancestor_idx = jax.random.choice(key_t, weights.shape[0], p=weights, shape=())
    particle_t_minus_1 = state_t_minus_1.particles[ancestor_idx]

    return particle_t_minus_1, (particle_t_minus_1, ancestor_idx)


def _backward_trace_step(ancestor_idx_t: int, state_t_minus_1: ParticleFilterState):
    """A single step of the backward tracing pass."""
    ancestor_idx_t_minus_1 = state_t_minus_1.ancestor_indices[ancestor_idx_t]
    particle_t_minus_1 = state_t_minus_1.particles[ancestor_idx_t_minus_1]
    return ancestor_idx_t_minus_1, (particle_t_minus_1, ancestor_idx_t_minus_1)


def build_csmc_smoother(
    log_potential_fn: LogPotential,
    ancestor_move_fn: AncestorMove,
    conditional: bool = True,
):
    """Builds a CSMC smoother function.

    Args:
        log_potential_fn: The log potential function.
        ancestor_move_fn: The function to move the final ancestor.
        conditional: Whether the pass is conditional.

    Returns:
        A smoother function that can be applied to the output of a forward pass.
    """

    def _smoother(
        forward_states: ParticleFilterState, key: KeyArray, do_sampling: bool
    ):
        """The smoother function to be returned.

        Args:
            forward_states: The output of the forward pass.
            key: JAX random number generator key.
            do_sampling: Whether to perform backward sampling or tracing.
        """
        T = forward_states.particles.shape[0]
        keys = jax.random.split(key, T)

        # Select last ancestor
        final_state = jax.tree_map(lambda x: x[-1], forward_states)
        final_log_weights = final_state.log_weights
        if not conditional:
            weights = normalize(final_log_weights)
            final_ancestor_idx = jax.random.choice(
                keys[-1], weights.shape[0], p=weights
            )
        else:
            final_ancestor_idx, _ = ancestor_move_fn(
                keys[-1],
                normalize(final_log_weights),
                final_state.particles.shape[0] - 1,
            )
        final_particle = final_state.particles[final_ancestor_idx]

        def sampling_pass():
            """Performs a backward sampling pass."""
            backward_step_fn = partial(_backward_sampling_step, log_potential_fn)
            init_carry = final_particle
            inputs = (
                jax.tree_map(lambda x: x[:-1], forward_states)[::-1],
                keys[:-1],
            )
            _, (particles, indices) = jax.lax.scan(
                backward_step_fn, init_carry, inputs
            )
            return particles, indices

        def tracing_pass():
            """Performs a backward tracing pass."""
            backward_step_fn = _backward_trace_step
            init_carry = final_ancestor_idx
            inputs = jax.tree_map(lambda x: x[:-1], forward_states)[::-1]
            _, (particles, indices) = jax.lax.scan(
                backward_step_fn, init_carry, inputs
            )
            return particles, indices

        particles, indices = jax.lax.cond(
            do_sampling, sampling_pass, tracing_pass
        )

        particles = jnp.insert(particles, 0, final_particle, axis=0)
        indices = jnp.insert(indices, 0, final_ancestor_idx, axis=0)

        return particles[::-1], indices[::-1]

    return _smoother
