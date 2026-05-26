from typing import NamedTuple, cast

import chex
import jax
import jax.numpy as jnp
import pytest
from jax import Array, random

from cuthbert import factorial
from cuthbert.inference import Filter
from cuthbert.smc.particle_filter import ParticleFilterState
from cuthbertlib.resampling import no_resampling, systematic
from cuthbertlib.stats.multivariate_normal import logpdf
from cuthbertlib.types import ArrayTree
from tests.cuthbert.factorial.gaussian_utils import generate_factorial_kalman_model
from tests.cuthbert.factorial.test_kalman import build_pairwise_factorial_filter


@pytest.fixture(scope="module", autouse=True)
def config():
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", False)


class SMCModelInputs(NamedTuple):
    t: Array
    factorial_inds: Array


def build_factorial_smc_filter(
    model_params: tuple[Array, ...], n_particles: int
) -> tuple[Filter, factorial.Factorializer, SMCModelInputs]:
    m0, chol_P0, Fs, cs, chol_Qs, Hs, ds, chol_Rs, ys, factorial_indices = model_params
    num_factors, x_dim = m0.shape

    def init_prepare(model_inputs, key=None):
        if key is None:
            raise ValueError("A JAX PRNG key must be provided.")
        eps = random.normal(key, (num_factors, n_particles, x_dim))
        particles = m0[:, None, :] + jnp.einsum("fij,fnj->fni", chol_P0, eps)
        return ParticleFilterState(
            key=key,
            particles=particles,
            log_weights=jnp.zeros((num_factors, n_particles)),
            ancestor_indices=jnp.tile(jnp.arange(n_particles)[None], (num_factors, 1)),
            model_inputs=model_inputs,
            log_normalizing_constant=jnp.array(0.0),
        )

    def filter_prepare(model_inputs, key=None):
        if key is None:
            raise ValueError("A JAX PRNG key must be provided.")
        local_dim = model_inputs.factorial_inds.shape[0] * x_dim
        return ParticleFilterState(
            key=key,
            particles=jnp.empty((n_particles, local_dim)),
            log_weights=jnp.zeros((n_particles,)),
            ancestor_indices=jnp.arange(n_particles),
            model_inputs=model_inputs,
            log_normalizing_constant=jnp.array(0.0),
        )

    def filter_combine(state_1, state_2):
        n = state_1.n_particles
        key_resample, key_propagate = random.split(state_1.key)
        ancestor_indices, log_weights, ancestors = no_resampling.resampling(
            key_resample, state_1.log_weights, state_1.particles, n
        )

        t = state_2.model_inputs.t - 1
        F, c, chol_Q = Fs[t], cs[t], chol_Qs[t]
        H, d, chol_R, y = Hs[t], ds[t], chol_Rs[t], ys[t]
        keys = random.split(key_propagate, n + 1)
        mean_particles = ancestors @ F.T + c
        noise = jax.vmap(
            lambda k: chol_Q @ random.normal(k, (mean_particles.shape[-1],))
        )(keys[1:])
        next_particles = mean_particles + noise
        log_potentials = jax.vmap(
            lambda x: logpdf(H @ x + d, y, chol_R, nan_support=False)
        )(next_particles)
        next_log_weights = log_weights + log_potentials
        log_normalizing_constant = state_1.log_normalizing_constant + (
            jax.nn.logsumexp(next_log_weights) - jax.nn.logsumexp(log_weights)
        )

        return ParticleFilterState(
            key=keys[0],
            particles=next_particles,
            log_weights=next_log_weights,
            ancestor_indices=ancestor_indices,
            model_inputs=state_2.model_inputs,
            log_normalizing_constant=log_normalizing_constant,
        )

    filter_obj = Filter(
        init_prepare=init_prepare,
        filter_prepare=filter_prepare,
        filter_combine=filter_combine,
        associative=False,
    )
    smc_model_inputs = SMCModelInputs(
        t=jnp.arange(factorial_indices.shape[0] + 1),
        factorial_inds=jnp.concatenate(
            [
                jnp.zeros((1, factorial_indices.shape[1]), dtype=jnp.int32),
                factorial_indices,
            ],
            axis=0,
        ),
    )
    factorializer = factorial.smc.build_factorializer(
        lambda model_inputs: model_inputs.factorial_inds,
        resampling_fn=systematic.resampling,
    )
    return filter_obj, factorializer, smc_model_inputs


params = [(0, 8, 2, 8, 3000), (1, 8, 2, 8, 3000)]


@pytest.mark.parametrize(
    "seed,num_factors,local_num_factors,num_time_steps,num_particles",
    params,
)
def test_factorial_smc_filter(
    seed, num_factors, local_num_factors, num_time_steps, num_particles
):
    model_params = generate_factorial_kalman_model(
        seed=seed,
        x_dim=1,
        y_dim=1,
        num_factors=num_factors,
        num_factors_local=local_num_factors,
        num_time_steps=num_time_steps,
    )
    kalman_filter, kalman_factorializer, kalman_model_inputs = (
        build_pairwise_factorial_filter(model_params)
    )
    kalman_states = cast(
        ArrayTree,
        factorial.filter(
            kalman_filter,
            kalman_factorializer,
            kalman_model_inputs,
            output_factorial=True,
        ),
    )
    _, kalman_local_states = factorial.filter(
        kalman_filter, kalman_factorializer, kalman_model_inputs, output_factorial=False
    )
    kalman_covs = kalman_states.chol_cov @ kalman_states.chol_cov.transpose(0, 1, 3, 2)

    smc_filter, smc_factorializer, smc_model_inputs = build_factorial_smc_filter(
        model_params, n_particles=num_particles
    )
    smc_states = cast(
        ParticleFilterState,
        factorial.filter(
            smc_filter,
            smc_factorializer,
            smc_model_inputs,
            output_factorial=True,
            key=random.key(seed + 123),
        ),
    )
    _, smc_local_states = factorial.filter(
        smc_filter,
        smc_factorializer,
        smc_model_inputs,
        output_factorial=False,
        key=random.key(seed + 123),
    )
    smc_weights = jax.nn.softmax(smc_states.log_weights, axis=-1)
    smc_means = jnp.sum(smc_states.particles[..., 0] * smc_weights, axis=-1)
    smc_vars = jnp.sum(
        (smc_states.particles[..., 0] - smc_means[..., None]) ** 2 * smc_weights,
        axis=-1,
    )

    kalman_means = kalman_states.mean[..., 0]
    kalman_vars = kalman_covs[..., 0, 0]
    kalman_ells = kalman_local_states.log_normalizing_constant
    smc_ells = smc_local_states.log_normalizing_constant
    smc_ells_cumulative = jnp.cumsum(smc_ells)
    chex.assert_trees_all_close(
        smc_means,
        kalman_means,
        rtol=1e-1,
        atol=1e-1,
    )
    chex.assert_trees_all_close(
        smc_vars,
        kalman_vars,
        rtol=1e-1,
        atol=1e-1,
    )
    chex.assert_trees_all_close(
        smc_ells_cumulative,
        kalman_ells,
        rtol=1e-1,
        atol=1e-1,
    )
