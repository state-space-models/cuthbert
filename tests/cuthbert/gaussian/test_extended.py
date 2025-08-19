import itertools

import chex
import jax
import jax.numpy as jnp
import pytest
from jax import Array

from cuthbert import filter, smoother
from cuthbert.gaussian import extended
from cuthbert.inference import Filter, Smoother
from tests.cuthbert.gaussian.test_kalman import std_kalman_filter
from tests.cuthbertlib.kalman.test_smoothing import std_kalman_smoother
from tests.cuthbertlib.kalman.utils import generate_lgssm


@pytest.fixture(scope="module", autouse=True)
def config():
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", False)


def load_extended_inference(
    m0: Array,
    chol_P0: Array,
    Fs: Array,
    cs: Array,
    chol_Qs: Array,
    Hs: Array,
    ds: Array,
    chol_Rs: Array,
    ys: Array,
) -> tuple[Filter, Smoother, Array]:
    """Builds extended Kalman filter and smoother objects and model_inputs for a linear-Gaussian SSM."""

    def get_init_params(model_inputs: int) -> tuple[Array, Array]:
        return m0, chol_P0

    def dynamics_mean_and_chol_cov(x, model_inputs):
        return Fs[model_inputs - 1] @ x + cs[model_inputs - 1], chol_Qs[
            model_inputs - 1
        ]

    def observation_mean_and_chol_cov_and_y(x, model_inputs):
        return (
            Hs[model_inputs] @ x + ds[model_inputs],
            chol_Rs[model_inputs],
            ys[model_inputs],
        )

    filter = extended.build_filter(
        get_init_params, dynamics_mean_and_chol_cov, observation_mean_and_chol_cov_and_y
    )
    smoother = extended.build_smoother(dynamics_mean_and_chol_cov)
    model_inputs = jnp.arange(len(ys))
    return filter, smoother, model_inputs


seeds = [0, 42, 99, 123, 456]
x_dims = [3]
y_dims = [1, 2]
num_time_steps = [1, 25]

common_params = list(itertools.product(seeds, x_dims, y_dims, num_time_steps))


@pytest.mark.parametrize("seed,x_dim,y_dim,num_time_steps", common_params)
def test_offline_filter(seed, x_dim, y_dim, num_time_steps):
    # Generate a linear-Gaussian state-space model.
    m0, chol_P0, Fs, cs, chol_Qs, Hs, ds, chol_Rs, ys = generate_lgssm(
        seed, x_dim, y_dim, num_time_steps
    )

    if num_time_steps > 1:
        # Set an observation to nan
        ys = ys.at[1, 0].set(jnp.nan)

    extended_filter, _, model_inputs = load_extended_inference(
        m0, chol_P0, Fs, cs, chol_Qs, Hs, ds, chol_Rs, ys
    )

    # Run sequential sqrt filter
    seq_states = filter(extended_filter, model_inputs, parallel=False)
    seq_means, seq_chol_covs, seq_ells = (
        seq_states.mean,
        seq_states.chol_cov,
        seq_states.log_likelihood,
    )

    # Run the standard Kalman filter.
    P0 = chol_P0 @ chol_P0.T
    Qs = chol_Qs @ chol_Qs.transpose(0, 2, 1)
    Rs = chol_Rs @ chol_Rs.transpose(0, 2, 1)
    des_means, des_covs, des_ells = std_kalman_filter(
        m0, P0, Fs, cs, Qs, Hs, ds, Rs, ys
    )

    seq_covs = seq_chol_covs @ seq_chol_covs.transpose(0, 2, 1)
    chex.assert_trees_all_close(
        (seq_means, seq_covs, seq_ells),
        (des_means, des_covs, des_ells),
        rtol=1e-10,
    )


@pytest.mark.parametrize("seed,x_dim,y_dim,num_time_steps", common_params)
def test_smoother(seed, x_dim, y_dim, num_time_steps):
    # Generate a linear-Gaussian state-space model.
    m0, chol_P0, Fs, cs, chol_Qs, Hs, ds, chol_Rs, ys = generate_lgssm(
        seed, x_dim, y_dim, num_time_steps
    )

    extended_filter, extended_smoother, model_inputs = load_extended_inference(
        m0, chol_P0, Fs, cs, chol_Qs, Hs, ds, chol_Rs, ys
    )

    # Run the Kalman filter and the standard Kalman smoother.
    filt_states = filter(extended_filter, model_inputs)
    filt_means, filt_chol_covs = filt_states.mean, filt_states.chol_cov
    filt_covs = filt_chol_covs @ filt_chol_covs.transpose(0, 2, 1)
    Qs = chol_Qs @ chol_Qs.transpose(0, 2, 1)
    (des_means, des_covs), des_cross_covs = std_kalman_smoother(
        filt_means, filt_covs, Fs, cs, Qs
    )

    # Run the sequential and parallel versions of the square root smoother.
    seq_smoother_states = smoother(
        extended_smoother, filt_states, model_inputs, parallel=False
    )
    seq_means, seq_chol_covs, seq_gains = (
        seq_smoother_states.mean,
        seq_smoother_states.chol_cov,
        seq_smoother_states.gain,
    )

    par_smoother_states = smoother(
        extended_smoother, filt_states, model_inputs, parallel=True
    )
    par_means, par_chol_covs, par_gains = (
        par_smoother_states.mean,
        par_smoother_states.chol_cov,
        par_smoother_states.gain,
    )

    seq_covs = seq_chol_covs @ seq_chol_covs.transpose(0, 2, 1)
    par_covs = par_chol_covs @ par_chol_covs.transpose(0, 2, 1)
    seq_cross_covs = seq_gains[:-1] @ seq_covs[1:]
    par_cross_covs = par_gains[:-1] @ par_covs[1:]
    chex.assert_trees_all_close(
        (seq_means, seq_covs, seq_cross_covs),
        (par_means, par_covs, par_cross_covs),
        (des_means, des_covs, des_cross_covs),
        rtol=1e-10,
    )

    seq_default_mi = smoother(extended_smoother, filt_states, parallel=False)
    chex.assert_trees_all_close(seq_default_mi, seq_smoother_states, rtol=1e-10)

    par_default_mi = smoother(extended_smoother, filt_states, parallel=True)
    chex.assert_trees_all_close(par_default_mi, par_smoother_states, rtol=1e-10)
