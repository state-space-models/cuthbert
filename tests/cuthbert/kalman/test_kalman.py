import itertools

import chex
import jax
import jax.numpy as jnp
import pytest

from cuthbert.generalised_kalman.kalman import filter, sampler, smoother
from tests.cuthbertlib.kalman.test_filtering import std_predict, std_update
from tests.cuthbertlib.kalman.test_smoothing import std_kalman_smoother
from tests.cuthbertlib.kalman.utils import generate_lgssm


@pytest.fixture(scope="module", autouse=True)
def config():
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", False)


def std_kalman_filter(m0, P0, Fs, cs, Qs, Hs, ds, Rs, ys):
    """The standard Kalman filter."""

    def body(carry, inp):
        m, P, ell = carry
        F, c, Q, H, d, R, y = inp
        pred_m, pred_P = std_predict(m, P, F, c, Q)
        m, P, ell_incr = std_update(pred_m, pred_P, H, d, R, y)
        ell_cumulative = ell + ell_incr
        return (m, P, ell_cumulative), (m, P, ell_cumulative)

    (_, _, _), (m, P, ell_cumulative) = jax.lax.scan(
        body, (m0, P0, 0.0), (Fs, cs, Qs, Hs, ds, Rs, ys)
    )
    m = jnp.vstack([m0[None, ...], m])
    P = jnp.vstack([P0[None, ...], P])
    return m, P, ell_cumulative


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

    # Run both sequential and parallel versions of the square root filter.
    (seq_means, seq_chol_covs), (seq_ells,) = filter(
        m0, chol_P0, Fs, cs, chol_Qs, Hs, ds, chol_Rs, ys, parallel=False
    )
    (par_means, par_chol_covs), (par_ells,) = filter(
        m0, chol_P0, Fs, cs, chol_Qs, Hs, ds, chol_Rs, ys, parallel=True
    )

    # Run the standard Kalman filter.
    P0 = chol_P0 @ chol_P0.T
    Qs = chol_Qs @ chol_Qs.transpose(0, 2, 1)
    Rs = chol_Rs @ chol_Rs.transpose(0, 2, 1)
    des_means, des_covs, des_ells = std_kalman_filter(
        m0, P0, Fs, cs, Qs, Hs, ds, Rs, ys
    )

    seq_covs = seq_chol_covs @ seq_chol_covs.transpose(0, 2, 1)
    par_covs = par_chol_covs @ par_chol_covs.transpose(0, 2, 1)
    chex.assert_trees_all_close(
        (seq_means, seq_covs, seq_ells),
        (par_means, par_covs, par_ells),
        (des_means, des_covs, des_ells),
        rtol=1e-10,
    )


@pytest.mark.parametrize("seed,x_dim,y_dim,num_time_steps", common_params)
def test_smoother(seed, x_dim, y_dim, num_time_steps):
    # Generate a linear-Gaussian state-space model.
    m0, chol_P0, Fs, cs, chol_Qs, Hs, ds, chol_Rs, ys = generate_lgssm(
        seed, x_dim, y_dim, num_time_steps
    )

    # Run the Kalman filter and the standard Kalman smoother.
    (filt_means, filt_chol_covs), _ = filter(
        m0, chol_P0, Fs, cs, chol_Qs, Hs, ds, chol_Rs, ys, parallel=False
    )
    filt_covs = filt_chol_covs @ filt_chol_covs.transpose(0, 2, 1)
    Qs = chol_Qs @ chol_Qs.transpose(0, 2, 1)
    (des_means, des_covs), des_cross_covs = std_kalman_smoother(
        filt_means, filt_covs, Fs, cs, Qs
    )

    # Run the sequential and parallel versions of the square root smoother.
    (seq_means, seq_chol_covs), _ = smoother(
        filt_means, filt_chol_covs, Fs, cs, chol_Qs, parallel=False
    )
    (par_means, par_chol_covs), (smoother_gains,) = smoother(
        filt_means, filt_chol_covs, Fs, cs, chol_Qs, parallel=True
    )

    seq_covs = seq_chol_covs @ seq_chol_covs.transpose(0, 2, 1)
    par_covs = par_chol_covs @ par_chol_covs.transpose(0, 2, 1)
    cross_covs = smoother_gains @ par_covs[1:]
    chex.assert_trees_all_close(
        (seq_means, seq_covs), (par_means, par_covs), (des_means, des_covs), rtol=1e-10
    )
    chex.assert_trees_all_close(cross_covs, des_cross_covs, rtol=1e-10)


@pytest.mark.parametrize("seed,x_dim,y_dim,num_time_steps", common_params)
@pytest.mark.parametrize("parallel", [False, True])
def test_sampler(seed, x_dim, y_dim, num_time_steps, parallel):
    m0, chol_P0, Fs, cs, chol_Qs, Hs, ds, chol_Rs, ys = generate_lgssm(
        seed, x_dim, y_dim, num_time_steps
    )

    (filt_means, filt_chol_covs), _ = filter(
        m0, chol_P0, Fs, cs, chol_Qs, Hs, ds, chol_Rs, ys, parallel=False
    )
    (des_means, des_chol_covs), _ = smoother(
        filt_means, filt_chol_covs, Fs, cs, chol_Qs, parallel=False
    )
    des_covs = des_chol_covs @ des_chol_covs.transpose(0, 2, 1)

    key = jax.random.key(seed)

    # Check default
    sample = sampler(
        key, filt_means, filt_chol_covs, Fs, cs, chol_Qs, parallel=parallel
    )
    assert sample.shape == (num_time_steps + 1, x_dim)

    # Check large number of samples
    shape = (50, 1000)
    samples = sampler(
        key, filt_means, filt_chol_covs, Fs, cs, chol_Qs, shape, parallel=parallel
    )
    assert samples.shape == (*shape, num_time_steps + 1, x_dim)
    samples_flat = samples.reshape(
        (-1, num_time_steps + 1, x_dim)
    )  # Flatten axis 0 and 1
    sample_means = jnp.mean(samples_flat, 0)
    sample_covs = jax.vmap(lambda x: jnp.cov(x, rowvar=False), in_axes=1)(samples_flat)
    chex.assert_trees_all_close(
        (sample_means, sample_covs), (des_means, des_covs), atol=1e-2, rtol=1e-2
    )
