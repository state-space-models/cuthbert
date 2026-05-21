import itertools
from typing import NamedTuple

import chex
import jax
import jax.numpy as jnp
import pytest
from jax import Array

from cuthbert import factorial
from cuthbert.inference import Filter
from cuthbert.smc.particle_filter import ParticleFilterState


@pytest.fixture(scope="module", autouse=True)
def config():
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", False)


class SMCModelInputs(NamedTuple):
    t: Array
    drift: Array
    obs_scale: Array
    factorial_inds: Array


def make_model(seed, num_factors, local_num_factors, num_particles, num_time_steps):
    key = jax.random.key(seed)
    k1, k2, k3, k4 = jax.random.split(key, 4)

    init_particles = jax.random.normal(k1, (num_factors, num_particles, 1))
    drifts = jax.random.normal(k2, (num_time_steps, local_num_factors, 1))
    obs_scales = jax.random.normal(k3, (num_time_steps, local_num_factors, 1))
    factorial_inds = jax.vmap(
        lambda k: jax.random.choice(
            k, jnp.arange(num_factors), shape=(local_num_factors,), replace=False
        )
    )(jax.random.split(k4, num_time_steps))

    model_inputs = SMCModelInputs(
        t=jnp.arange(num_time_steps + 1),
        drift=jnp.concatenate([jnp.zeros((1, local_num_factors, 1)), drifts], axis=0),
        obs_scale=jnp.concatenate(
            [jnp.zeros((1, local_num_factors, 1)), obs_scales], axis=0
        ),
        factorial_inds=jnp.concatenate(
            [jnp.zeros((1, local_num_factors), dtype=jnp.int32), factorial_inds], axis=0
        ),
    )
    return init_particles, model_inputs


def build_deterministic_filter(init_particles, num_particles):
    num_factors = init_particles.shape[0]

    def init_prepare(model_inputs, key=None):
        return ParticleFilterState(
            key=jax.random.key(0),
            particles=init_particles,
            log_weights=jnp.zeros((num_factors, num_particles)),
            ancestor_indices=jnp.tile(
                jnp.arange(num_particles)[None], (num_factors, 1)
            ),
            model_inputs=model_inputs,
            log_normalizing_constant=jnp.array(0.0),
        )

    def filter_prepare(model_inputs, key=None):
        return ParticleFilterState(
            key=jax.random.key(0),
            particles=jnp.empty((num_particles, 0)),
            log_weights=jnp.zeros(num_particles),
            ancestor_indices=jnp.arange(num_particles),
            model_inputs=model_inputs,
            log_normalizing_constant=jnp.array(0.0),
        )

    def filter_combine(state_1, state_2):
        # Deterministic "SMC-like" update with always-resampled bookkeeping.
        particles = state_1.particles + state_2.model_inputs.drift.reshape(1, -1)
        obs_score = jnp.sum(
            particles * state_2.model_inputs.obs_scale.reshape(1, -1), axis=1
        )
        ell_inc = jax.nn.logsumexp(obs_score) - jnp.log(state_1.n_particles)
        return ParticleFilterState(
            key=state_2.key,
            particles=particles,
            log_weights=jnp.zeros(state_1.n_particles),
            ancestor_indices=jnp.arange(state_1.n_particles),
            model_inputs=state_2.model_inputs,
            log_normalizing_constant=state_1.log_normalizing_constant + ell_inc,
        )

    return Filter(
        init_prepare=init_prepare,
        filter_prepare=filter_prepare,
        filter_combine=filter_combine,
        associative=False,
    )


def reference_factorial_smc(init_particles, model_inputs):
    num_time_steps = model_inputs.t.shape[0] - 1
    num_factors = init_particles.shape[0]
    num_particles = init_particles.shape[1]
    particles = init_particles
    log_weights = jnp.zeros((num_factors, num_particles))
    ancestor_indices = jnp.tile(jnp.arange(num_particles)[None], (num_factors, 1))
    ell = jnp.array(0.0)

    local_particles_all = []
    local_log_weights_all = []
    local_ancestor_indices_all = []
    local_ell_all = []
    factorial_particles_all = [particles]
    factorial_log_weights_all = [log_weights]
    factorial_ancestor_indices_all = [ancestor_indices]
    factorial_ell_all = [ell]

    for t in range(1, num_time_steps + 1):
        inds = model_inputs.factorial_inds[t]
        drift = model_inputs.drift[t]
        obs_scale = model_inputs.obs_scale[t]

        local = particles[inds]  # (L, N, 1)
        local_joint = jnp.moveaxis(local, 0, 1).reshape(local.shape[1], -1)  # (N, L)
        local_joint = local_joint + drift.reshape(1, -1)
        obs_score = jnp.sum(local_joint * obs_scale.reshape(1, -1), axis=1)
        ell_inc = jax.nn.logsumexp(obs_score) - jnp.log(num_particles)
        local_ell = ell + ell_inc
        local_log_weights = jnp.zeros((len(inds), num_particles))
        local_ancestor_indices = jnp.tile(
            jnp.arange(num_particles)[None], (len(inds), 1)
        )

        local_next = local_joint.reshape(local.shape[1], local.shape[0], 1).transpose(
            1, 0, 2
        )
        particles = particles.at[inds].set(local_next)
        log_weights = jnp.zeros_like(log_weights)
        ancestor_indices = ancestor_indices.at[inds].set(local_ancestor_indices)
        ell = local_ell

        local_particles_all.append(local_next)
        local_log_weights_all.append(local_log_weights)
        local_ancestor_indices_all.append(local_ancestor_indices)
        local_ell_all.append(local_ell)
        factorial_particles_all.append(particles)
        factorial_log_weights_all.append(log_weights)
        factorial_ancestor_indices_all.append(ancestor_indices)
        factorial_ell_all.append(ell)

    return (
        jnp.stack(local_particles_all),
        jnp.stack(local_log_weights_all),
        jnp.stack(local_ancestor_indices_all),
        jnp.stack(local_ell_all),
        jnp.stack(factorial_particles_all),
        jnp.stack(factorial_log_weights_all),
        jnp.stack(factorial_ancestor_indices_all),
        jnp.stack(factorial_ell_all),
    )


params = list(itertools.product([0, 1], [8], [2], [25], [1, 10]))


@pytest.mark.parametrize(
    "seed,num_factors,local_num_factors,num_particles,num_time_steps",
    params,
)
def test_factorial_smc_filter(
    seed, num_factors, local_num_factors, num_particles, num_time_steps
):
    init_particles, model_inputs = make_model(
        seed, num_factors, local_num_factors, num_particles, num_time_steps
    )
    filter_obj = build_deterministic_filter(init_particles, num_particles)
    factorializer = factorial.smc.build_factorializer(lambda mi: mi.factorial_inds)
    (
        local_particles_ref,
        local_log_weights_ref,
        local_ancestor_indices_ref,
        local_ells_ref,
        factorial_particles_ref,
        factorial_log_weights_ref,
        factorial_ancestor_indices_ref,
        factorial_ells_ref,
    ) = reference_factorial_smc(init_particles, model_inputs)

    init_state, local_states = factorial.filter(
        filter_obj, factorializer, model_inputs, output_factorial=False
    )
    chex.assert_trees_all_close(
        init_state.particles, init_particles, rtol=1e-10, atol=0.0
    )
    chex.assert_trees_all_close(
        local_states.particles, local_particles_ref, rtol=1e-10, atol=0.0
    )
    chex.assert_trees_all_close(
        local_states.log_weights, local_log_weights_ref, rtol=1e-10, atol=0.0
    )
    chex.assert_trees_all_close(
        local_states.ancestor_indices, local_ancestor_indices_ref, rtol=1e-10, atol=0.0
    )
    chex.assert_trees_all_close(
        local_states.log_normalizing_constant, local_ells_ref, rtol=1e-10, atol=0.0
    )

    factorial_states = factorial.filter(
        filter_obj, factorializer, model_inputs, output_factorial=True
    )
    chex.assert_trees_all_close(
        factorial_states.particles, factorial_particles_ref, rtol=1e-10, atol=0.0
    )
    chex.assert_trees_all_close(
        factorial_states.log_weights, factorial_log_weights_ref, rtol=1e-10, atol=0.0
    )
    chex.assert_trees_all_close(
        factorial_states.ancestor_indices,
        factorial_ancestor_indices_ref,
        rtol=1e-10,
        atol=0.0,
    )
    chex.assert_trees_all_close(
        factorial_states.log_normalizing_constant,
        factorial_ells_ref,
        rtol=1e-10,
        atol=0.0,
    )
