"""Factorial utilities for SMC particle-filter states."""

from typing import TypeVar

import jax
from jax import numpy as jnp
from jax import random, tree

from cuthbert.factorial.types import Factorializer, GetFactorialIndices
from cuthbert.smc.marginal_particle_filter import MarginalParticleFilterState
from cuthbert.smc.particle_filter import ParticleFilterState
from cuthbertlib.resampling import Resampling, ess_decorator
from cuthbertlib.types import Array, ArrayLike

GeneralParticleFilterState = TypeVar(
    "GeneralParticleFilterState", ParticleFilterState, MarginalParticleFilterState
)


def build_factorializer(
    get_factorial_indices: GetFactorialIndices,
    resampling_fn: Resampling,
) -> Factorializer:
    """Build a factorializer for particle-filter states.

    In `cuthbert.smc`, resampling happens before propagation/reweighting.
    Factorial `join` needs unweighted particles, so `resampling_fn` is required
    and applied in `join` whenever local factor weights are not constant.

    Any weights passed to marginalize will be duplicated across factors.

    Args:
        get_factorial_indices: Function to extract factorial indices from model inputs.
        resampling_fn: Resampling function used in `join` when local factor
            weights are not constant.
            Consider setting the main SMC filter's `resampling_fn` to a no-op
            policy to avoid redundant resampling, since `join` handles this
            pre-join resampling step when needed.
            Any adaptive ESS threshold will be overwritten, and resampling will be
            applied with a threshold of 1.
            # TODO: explicit reference to resampling no-op

    Returns:
        Factorializer for SMC states with extract, join, marginalize, and insert.
    """
    return Factorializer(
        get_factorial_indices=get_factorial_indices,
        extract=extract,
        join=lambda local_factorial_state: join(local_factorial_state, resampling_fn),
        marginalize=marginalize,
        insert=insert,
    )


def extract(
    factorial_state: GeneralParticleFilterState,
    factorial_inds: ArrayLike,
) -> GeneralParticleFilterState:
    """Extract selected factors from a factorial particle-filter state.

    Args:
        factorial_state: Factorial particle-filter state with factorized fields
            on leading axis F (`particles`, `log_weights`).
        factorial_inds: Indices of factors to extract.

    Returns:
        Local factorial particle-filter state with selected factors on the
        leading axis of factorized fields.
    """
    factorial_inds = jnp.asarray(factorial_inds)
    particles = tree.map(lambda x: x[factorial_inds], factorial_state.particles)
    log_weights = factorial_state.log_weights[factorial_inds]

    new_state = factorial_state._replace(
        particles=particles,
        log_weights=log_weights,
    )

    if isinstance(factorial_state, ParticleFilterState):
        new_state = new_state._replace(
            ancestor_indices=factorial_state.ancestor_indices[factorial_inds]
        )

    return new_state


def join(
    local_factorial_state: GeneralParticleFilterState,
    resampling_fn: Resampling,
) -> GeneralParticleFilterState:
    """Join local factorial state into a single joint local particle-filter state.

    Resampling is applied first, independently over factors, when local
    factor weights are not constant (detected via effective sample size).
    Then factorized particles are stacked into a local joint particle state.
    Joined bookkeeping uses always-resampled conventions: zero log weights.

    Ancestor indices are valid for the resampling but ignored for the join
    i.e. retain the factorial axis (F, n_particles) and assumed not used in the
    particle filter.

    Args:
        local_factorial_state: Local factorial particle-filter state.
        resampling_fn: Resampling function for factor-wise pre-join resampling.

    Returns:
        Joint local particle-filter state with no factorial axis on particle values.
    """
    n_factors, n_particles = local_factorial_state.log_weights.shape

    # Resample independently over factors
    # Applied if logits are not constant (i.e. ess_threshold = 1)
    resampling_fn = ess_decorator(resampling_fn, threshold=1 - 1e-6)
    keys = random.split(local_factorial_state.key, n_factors + 1)
    key = keys[0]
    resampling_keys = keys[1:]
    ancestor_indices, _, particles = jax.vmap(resampling_fn, in_axes=(0, 0, 0, None))(
        resampling_keys,
        local_factorial_state.log_weights,
        local_factorial_state.particles,
        n_particles,
    )  # log_weights ignored, all zeros

    # Combine factorial particles into joint
    # resampled_local_factorial_state.particles is shape e.g (F, n_particles, d)
    # resampled_local_factorial_state.particles is shape (F, n_particles)
    joint_state = local_factorial_state._replace(
        key=key,
        particles=tree.map(_join_particles_arr, particles),
        log_weights=jnp.zeros(
            (n_particles,), dtype=local_factorial_state.log_weights.dtype
        ),  # all zeros
    )

    if isinstance(joint_state, ParticleFilterState):
        joint_state = joint_state._replace(
            ancestor_indices=ancestor_indices
        )  # ancestor_indices retain factorial axis (F, n_particles) - assumed not used in particle filter

    return joint_state


def _join_particles_arr(arr: Array) -> Array:
    """Join one factorized particle leaf `(F, N, ...)` into `(N, F * ...)`."""
    # (F, N, ...) -> (N, F * ...)
    arr = jnp.moveaxis(arr, 0, 1)
    return arr.reshape(arr.shape[0], -1)


def marginalize(
    local_state: GeneralParticleFilterState,
    num_factors: int,
) -> GeneralParticleFilterState:
    """Marginalize a joint local particle state back to factorial form.

    Weights are duplicated across factors.
    Ancestor indices are ignored for marginalization.

    Args:
        local_state: Joint local particle-filter state with particle leaves
            shaped `(N, D_joint)`.
        num_factors: Number of local factors in the factorial representation.

    Returns:
        Local factorial particle-filter state where particle leaves are shaped
        `(F, N, D_local)` and bookkeeping follows always-resampled semantics
        with missing ancestor indices (`-1`).
    """
    particles = tree.map(
        lambda x: _marginalize_particles_arr(x, num_factors), local_state.particles
    )
    log_weights = local_state.log_weights.repeat(num_factors, axis=0)
    return local_state._replace(
        particles=particles,
        log_weights=log_weights,
    )


def _marginalize_particles_arr(arr: Array, num_factors: int) -> Array:
    """Split one joined particle leaf `(N, D_joint)` into `(F, N, D_local)`."""
    n_particles, joint_dim = arr.shape[0], arr.shape[1]
    local_dim = joint_dim // num_factors
    return arr.reshape(n_particles, num_factors, local_dim).transpose(1, 0, 2)


def insert(
    local_factorial_state: GeneralParticleFilterState,
    factorial_state: GeneralParticleFilterState,
    factorial_inds: ArrayLike,
) -> GeneralParticleFilterState:
    """Insert local factorial update into the global factorial state.

    Args:
        local_factorial_state: Updated local factorial particle-filter state.
        factorial_state: Previous global factorial particle-filter state.
        factorial_inds: Factor indices where local updates are inserted.

    Returns:
        Updated global factorial particle-filter state.
    """
    factorial_inds = jnp.asarray(factorial_inds)
    factorial_inds = jnp.atleast_1d(factorial_inds)

    particles = tree.map(
        lambda loc, glob: glob.at[factorial_inds].set(loc),
        local_factorial_state.particles,
        factorial_state.particles,
    )
    log_weights = factorial_state.log_weights.at[factorial_inds].set(
        local_factorial_state.log_weights
    )

    new_factorial_state = factorial_state._replace(
        key=local_factorial_state.key,
        particles=particles,
        log_weights=log_weights,
    )

    if isinstance(factorial_state, ParticleFilterState) and isinstance(
        local_factorial_state, ParticleFilterState
    ):
        ancestor_indices = factorial_state.ancestor_indices.at[factorial_inds].set(
            local_factorial_state.ancestor_indices
        )
        new_factorial_state = new_factorial_state._replace(
            ancestor_indices=ancestor_indices
        )

    return new_factorial_state
