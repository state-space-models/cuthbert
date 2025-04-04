import chex
import jax
import jax.numpy as jnp
import pytest

from cuthbertlib.kalman.filtering import predict, update, filter
from cuthbertlib.kalman.utils import mvn_logpdf
from tests.cuthbertlib.kalman.utils import generate_lgssm


@pytest.fixture(scope="module", autouse=True)
def config():
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", False)


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


@pytest.mark.parametrize("seed", [0, 42, 99, 123, 456])
@pytest.mark.parametrize("x_dim", [3])
def test_predict(seed, x_dim):
    m0, chol_P0, Fs, bs, chol_Qs = generate_lgssm(seed, x_dim, 0, 1)[:5]
    F = Fs[0]
    c = bs[0]
    chol_Q = chol_Qs[0]
    P0 = chol_P0 @ chol_P0.T
    Q = chol_Q @ chol_Q.T

    pred_m, pred_chol_cov = predict(m0, chol_P0, F, c, chol_Q)
    pred_cov = pred_chol_cov @ pred_chol_cov.T

    des_m, des_cov = std_predict(m0, P0, F, c, Q)

    chex.assert_trees_all_close((pred_m, pred_cov), (des_m, des_cov), rtol=1e-10)


@pytest.mark.parametrize("seed", [0, 42, 99, 123, 456])
@pytest.mark.parametrize("x_dim", [3])
@pytest.mark.parametrize("y_dim", [1, 2])
def test_update(seed, x_dim, y_dim):
    lgssm = generate_lgssm(seed, x_dim, y_dim, 1)
    m0, chol_P0 = lgssm[:2]
    Hs, ds, chol_Rs, ys = lgssm[-4:]

    H = Hs[0]
    d = ds[0]
    chol_R = chol_Rs[0]
    y = ys[0]
    P0 = chol_P0 @ chol_P0.T
    R = chol_R @ chol_R.T

    (m, chol_P), (ell,) = update(m0, chol_P0, H, d, chol_R, y)
    P = chol_P @ chol_P.T

    des_m, des_P, des_ell = std_update(m0, P0, H, d, R, y)

    chex.assert_trees_all_close((m, P), (des_m, des_P), rtol=1e-10)
    chex.assert_trees_all_close(ell, des_ell, rtol=1e-10)


@pytest.mark.parametrize("seed", [0, 42, 99, 123, 456])
@pytest.mark.parametrize("x_dim", [3])
@pytest.mark.parametrize("y_dim", [1, 2])
@pytest.mark.parametrize("num_time_steps", [1, 25])
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
