import chex
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from kalman.filter import offline_filter
from kalman.utils import mvn_logpdf


@pytest.fixture(scope="module", autouse=True)
def config():
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", False)


def generate_cholesky_factor(rng, dim):
    chol_A = rng.random((dim, dim))
    chol_A[np.triu_indices(dim, 1)] = 0.0
    return chol_A


def generate_trans_model(rng, x_dim):
    F = rng.random((x_dim, x_dim))
    b = rng.random(x_dim)
    chol_Q = generate_cholesky_factor(rng, x_dim)
    return F, b, chol_Q


def generate_obs_model(rng, x_dim, y_dim):
    H = rng.random((y_dim, x_dim))
    c = rng.random(y_dim)
    chol_R = generate_cholesky_factor(rng, y_dim)
    y = rng.random(y_dim)
    return H, c, chol_R, y


def std_kalman_filter(m0, P0, Fs, cs, Qs, Hs, ds, Rs, ys):
    """The standard Kalman filter."""

    def std_predict(m, P, F, c, Q):
        m = F @ m + c
        P = F @ P @ F.T + Q
        return m, P

    def std_update(m, P, H, d, R, y):
        residual = y - H @ m - d
        S = H @ P @ H.T + R
        K = jax.scipy.linalg.solve(S, H @ P, assume_a="pos").T
        m = m + K @ residual
        P = P - K @ S @ K.T
        ell = mvn_logpdf(residual, jnp.linalg.cholesky(S))
        return m, P, ell

    def body(carry, inp):
        m, P, ell = carry
        F, c, Q, H, d, R, y = inp
        pred_m, pred_P = std_predict(m, P, F, c, Q)
        m, P, ell_incr = std_update(pred_m, pred_P, H, d, R, y)
        return (m, P, ell + ell_incr), (m, P)

    (_, _, ell), (m, P) = jax.lax.scan(
        body, (m0, P0, 0.0), (Fs, cs, Qs, Hs, ds, Rs, ys)
    )
    m = jnp.vstack([m0[None, ...], m])
    P = jnp.vstack([P0[None, ...], P])
    return m, P, ell


def batch_arrays(t, *args):
    out = []
    for arg in args:
        out.append(jnp.repeat(arg[None, ...], t, axis=0))
    return out


@pytest.mark.parametrize("seed", [0, 42, 99, 123, 456])
@pytest.mark.parametrize("x_dim", [3])
@pytest.mark.parametrize("y_dim", [1, 2])
@pytest.mark.parametrize("num_time_steps", [1, 25])
def test_offline_filter(seed, x_dim, y_dim, num_time_steps):
    # Generate a random state-space model.
    rng = np.random.default_rng(seed)
    m0 = rng.normal(size=x_dim)
    chol_P0 = generate_cholesky_factor(rng, x_dim)
    F, c, chol_Q = generate_trans_model(rng, x_dim)
    H, d, chol_R, y = generate_obs_model(rng, x_dim, y_dim)

    # Make copies for T time steps.
    Fs, cs, chol_Qs, Hs, ds, chol_Rs, ys = batch_arrays(
        num_time_steps, F, c, chol_Q, H, d, chol_R, y
    )

    # Run both sequential and parallel versions of the square root filter.
    (seq_means, seq_chol_covs), seq_ell = offline_filter(
        m0, chol_P0, Fs, cs, chol_Qs, Hs, ds, chol_Rs, ys, parallel=False
    )
    (par_means, par_chol_covs), par_ell = offline_filter(
        m0, chol_P0, Fs, cs, chol_Qs, Hs, ds, chol_Rs, ys, parallel=True
    )

    # Run the standard Kalman filter.
    P0 = chol_P0 @ chol_P0.T
    Qs = chol_Qs @ chol_Qs.transpose(0, 2, 1)
    Rs = chol_Rs @ chol_Rs.transpose(0, 2, 1)
    des_means, des_covs, des_ell = std_kalman_filter(m0, P0, Fs, cs, Qs, Hs, ds, Rs, ys)

    seq_covs = seq_chol_covs @ seq_chol_covs.transpose(0, 2, 1)
    par_covs = par_chol_covs @ par_chol_covs.transpose(0, 2, 1)
    chex.assert_trees_all_close(
        (seq_means, seq_covs, seq_ell),
        (par_means, par_covs, par_ell),
        (des_means, des_covs, des_ell),
        rtol=1e-10,
    )
