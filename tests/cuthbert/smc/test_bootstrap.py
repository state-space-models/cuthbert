import itertools

import chex
import jax
import jax.numpy as jnp
import pytest
from jax import random

from cuthbert import filter
from cuthbert.smc import bootstrap
from cuthbertlib.resampling import systematic
from cuthbertlib.stats.multivariate_normal import logpdf
from tests.cuthbert.gaussian.test_kalman import std_kalman_filter
from tests.cuthbertlib.kalman.utils import generate_lgssm


@pytest.fixture(scope="module", autouse=True)
def config():
    jax.config.update("jax_enable_x64", True)
    jax.config.update("jax_disable_jit", True)
    yield
    jax.config.update("jax_enable_x64", False)


def load_bootstrap_inference(m0, chol_P0, Fs, cs, chol_Qs, Hs, ds, chol_Rs, ys):
    def init_sample(key, model_inputs):
        return m0 + chol_P0 @ random.normal(key, m0.shape)

    def propagate_sample(key, state, model_inputs: int):
        idx = model_inputs - 1
        mean_sample = Fs[idx] @ state + cs[idx]
        return mean_sample + chol_Qs[idx] @ random.normal(key, mean_sample.shape)

    def log_potential(state_prev, state, model_inputs: int):
        idx = model_inputs - 1
        return logpdf(
            Hs[idx] @ state + ds[idx], ys[idx], chol_Rs[idx], nan_support=False
        )

    bootstrap_inference = bootstrap.build(
        init_sample=init_sample,
        propagate_sample=propagate_sample,
        log_potential=log_potential,
        n_filter_particles=1000_000,
        n_smoother_particles=10,
        resampling_fn=systematic.resampling,
        ess_threshold=0.7,
    )
    model_inputs = jnp.arange(len(ys) + 1)
    return bootstrap_inference, model_inputs


seeds = [1, 42, 99, 123, 456]
x_dims = [3]
y_dims = [2]
num_time_steps = [20]

common_params = list(itertools.product(seeds, x_dims, y_dims, num_time_steps))


@pytest.mark.parametrize("seed,x_dim,y_dim,num_time_steps", common_params)
def test_filter(seed, x_dim, y_dim, num_time_steps):
    m0, chol_P0, Fs, cs, chol_Qs, Hs, ds, chol_Rs, ys = generate_lgssm(
        seed, x_dim, y_dim, num_time_steps
    )

    # Run the bootstrap particle filter.
    bootstrap_inference, model_inputs = load_bootstrap_inference(
        m0, chol_P0, Fs, cs, chol_Qs, Hs, ds, chol_Rs, ys
    )
    key = random.key(seed + 1)
    bootstrap_states = filter(
        bootstrap_inference, model_inputs, parallel=False, key=key
    )
    weights = jax.nn.softmax(bootstrap_states.log_weights)
    bt_means = jnp.sum(bootstrap_states.particles * weights[..., None], axis=1)
    bt_ells = bootstrap_states.log_likelihood

    # Run the standard Kalman filter.
    P0 = chol_P0 @ chol_P0.T
    Qs = chol_Qs @ chol_Qs.transpose(0, 2, 1)
    Rs = chol_Rs @ chol_Rs.transpose(0, 2, 1)
    des_means, des_covs, des_ells = std_kalman_filter(
        m0, P0, Fs, cs, Qs, Hs, ds, Rs, ys
    )

    chex.assert_trees_all_close(
        (bt_ells[1:], bt_means), (des_ells, des_means), atol=1e-1, rtol=1e-1
    )
