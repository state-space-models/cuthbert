"""Factorial utilities for SMC particle-filter states."""

from jax import numpy as jnp
from jax import tree

from cuthbert.factorial.types import Factorializer, GetFactorialIndices
from cuthbert.smc.particle_filter import ParticleFilterState
from cuthbertlib.types import Array, ArrayLike


def build_factorializer(
    get_factorial_indices: GetFactorialIndices,
) -> Factorializer:
    """Build a factorializer for particle-filter states.

    Args:
        get_factorial_indices: Function to extract factorial indices from model inputs.

    Returns:
        Factorializer for SMC states with extract, join, marginalize, and insert.
    """
    return Factorializer(
        get_factorial_indices=get_factorial_indices,
        extract=extract,
        join=join,
        marginalize=marginalize,
        insert=insert,
    )


def extract(
    factorial_state: ParticleFilterState,
    factorial_inds: ArrayLike,
) -> ParticleFilterState:
    """Extract selected factors from a factorial particle-filter state.

    Args:
        factorial_state: Factorial particle-filter state with factorized fields
            on leading axis F (`particles`, `log_weights`, `ancestor_indices`).
        factorial_inds: Indices of factors to extract.

    Returns:
        Local factorial particle-filter state with selected factors on the
        leading axis of factorized fields.
    """
    factorial_inds = jnp.asarray(factorial_inds)
    particles = tree.map(lambda x: x[factorial_inds], factorial_state.particles)
    log_weights = factorial_state.log_weights[factorial_inds]
    ancestor_indices = factorial_state.ancestor_indices[factorial_inds]
    return factorial_state._replace(
        particles=particles,
        log_weights=log_weights,
        ancestor_indices=ancestor_indices,
    )


def join(local_factorial_state: ParticleFilterState) -> ParticleFilterState:
    """Join local factorial state into a single joint local particle-filter state.

    Factorial SMC doesn't work nicely with log_weights or ancestor_indices.
    So we assume the log_weights are constant (i.e. always-resampled),
    and reset the ancestor_indices to identity indices.

    Args:
        local_factorial_state: Local factorial particle-filter state.

    Returns:
        Joint local particle-filter state with no factorial axis on particle values.
    """
    particles = tree.map(_join_particles_arr, local_factorial_state.particles)
    n_particles = local_factorial_state.log_weights.shape[-1]

    # Set log_weights to zeros if they are constant, otherwise set to NaN.
    weights_constant = jnp.allclose(
        local_factorial_state.log_weights, local_factorial_state.log_weights[:1]
    )
    log_weights = jnp.where(
        weights_constant,
        jnp.zeros((n_particles,), dtype=local_factorial_state.log_weights.dtype),
        jnp.full(
            (n_particles,), jnp.nan, dtype=local_factorial_state.log_weights.dtype
        ),
    )

    # Set ancestor_indices to identity indices.
    ancestor_indices = jnp.arange(
        n_particles, dtype=local_factorial_state.ancestor_indices.dtype
    )
    return local_factorial_state._replace(
        particles=particles,
        log_weights=log_weights,
        ancestor_indices=ancestor_indices,
    )


def _join_particles_arr(arr: Array) -> Array:
    """Join one factorized particle leaf `(F, N, ...)` into `(N, F * ...)`."""
    # (F, N, ...) -> (N, F * ...)
    arr = jnp.moveaxis(arr, 0, 1)
    return arr.reshape(arr.shape[0], -1)


def marginalize(
    local_state: ParticleFilterState,
    num_factors: int,
) -> ParticleFilterState:
    """Marginalize a joint local particle state back to factorial form.

    Args:
        local_state: Joint local particle-filter state with particle leaves
            shaped `(N, D_joint)`.
        num_factors: Number of local factors in the factorial representation.

    Returns:
        Local factorial particle-filter state where particle leaves are shaped
        `(F, N, D_local)` and bookkeeping follows always-resampled semantics.
    """
    particles = tree.map(
        lambda x: _marginalize_particles_arr(x, num_factors), local_state.particles
    )
    n_particles = local_state.log_weights.shape[-1]
    log_weights = jnp.zeros(
        (num_factors, n_particles), dtype=local_state.log_weights.dtype
    )
    ancestor_indices = _identity_ancestor_indices(
        num_factors, n_particles, local_state.ancestor_indices.dtype
    )
    return local_state._replace(
        particles=particles,
        log_weights=log_weights,
        ancestor_indices=ancestor_indices,
    )


def _marginalize_particles_arr(arr: Array, num_factors: int) -> Array:
    """Split one joined particle leaf `(N, D_joint)` into `(F, N, D_local)`."""
    n_particles, joint_dim = arr.shape[0], arr.shape[1]
    if joint_dim % num_factors != 0:
        raise ValueError(
            f"Cannot split joint particle dimension {joint_dim} into "
            f"{num_factors} factors."
        )
    local_dim = joint_dim // num_factors
    return arr.reshape(n_particles, num_factors, local_dim).transpose(1, 0, 2)


def insert(
    local_factorial_state: ParticleFilterState,
    factorial_state: ParticleFilterState,
    factorial_inds: ArrayLike,
) -> ParticleFilterState:
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
    # Keep bookkeeping consistent with always-resampled convention.
    log_weights = jnp.zeros_like(factorial_state.log_weights)
    ancestor_indices = _identity_ancestor_indices(
        factorial_state.ancestor_indices.shape[0],
        factorial_state.ancestor_indices.shape[-1],
        factorial_state.ancestor_indices.dtype,
    )
    return factorial_state._replace(
        key=local_factorial_state.key,
        particles=particles,
        log_weights=log_weights,
        ancestor_indices=ancestor_indices,
        model_inputs=local_factorial_state.model_inputs,
        log_normalizing_constant=local_factorial_state.log_normalizing_constant,
    )


def _identity_ancestor_indices(num_factors: int, n_particles: int, dtype) -> Array:
    """Create identity ancestor-index rows with shape `(F, N)`."""
    return jnp.tile(jnp.arange(n_particles, dtype=dtype)[None], (num_factors, 1))
