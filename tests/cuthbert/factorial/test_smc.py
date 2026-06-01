from typing import NamedTuple, cast

import chex
import jax
import jax.numpy as jnp
import pytest
from jax import Array, random

from cuthbert import factorial
from cuthbert.inference import Filter
from cuthbert.smc.particle_filter import ParticleFilterState, build_filter
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

    def init_sample(key, model_inputs):
        # Generates a single particle for all factors
        eps = random.normal(key, (num_factors, x_dim))
        return m0 + jax.vmap(lambda chol_P, e: chol_P @ e)(chol_P0, eps)

    def propagate_sample(key, state, model_inputs):
        t = model_inputs.t - 1
        mean_particle = Fs[t] @ state + cs[t]
        return mean_particle + chol_Qs[t] @ random.normal(key, mean_particle.shape)

    def log_potential(state_prev, state, model_inputs):
        t = model_inputs.t - 1
        return logpdf(Hs[t] @ state + ds[t], ys[t], chol_Rs[t], nan_support=False)

    # `init_sample` returns a per-factor sample `(F, x_dim)`, so `init_particle_axis=1`
    # makes the factorial machinery's factor axis lead the particle axis in the
    # initial state, i.e. particles `(F, N, x_dim)` and weights `(F, N)`.
    filter_obj = build_filter(
        init_sample,
        propagate_sample,
        log_potential,
        n_filter_particles=n_particles,
        resampling_fn=no_resampling.resampling,
        init_particle_axis=1,
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


def weighted_mean_and_var(states: ParticleFilterState) -> tuple[Array, Array]:
    """Weighted particle mean and variance of the (1D) state, over the particle axis."""
    weights = jax.nn.softmax(states.log_weights, axis=-1)
    particles = states.particles[..., 0]
    mean = jnp.sum(particles * weights, axis=-1)
    var = jnp.sum((particles - mean[..., None]) ** 2 * weights, axis=-1)
    return mean, var


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
    kalman_local_covs = (
        kalman_local_states.chol_cov
        @ kalman_local_states.chol_cov.transpose(0, 1, 3, 2)
    )

    # Factorial states hold all factors at every time step; local states hold the
    # factors updated at each time step. Check means, variances and (cumulative)
    # log normalizing constants agree with the Kalman reference for both.
    smc_means, smc_vars = weighted_mean_and_var(smc_states)
    smc_local_means, smc_local_vars = weighted_mean_and_var(smc_local_states)
    chex.assert_trees_all_close(
        (
            smc_means,
            smc_vars,
            smc_states.log_normalizing_constant,
            smc_local_means,
            smc_local_vars,
            smc_local_states.log_normalizing_constant,
        ),
        (
            kalman_states.mean[..., 0],
            kalman_covs[..., 0, 0],
            kalman_states.log_normalizing_constant,
            kalman_local_states.mean[..., 0],
            kalman_local_covs[..., 0, 0],
            kalman_local_states.log_normalizing_constant,
        ),
        rtol=1e-1,
        atol=1e-1,
    )
