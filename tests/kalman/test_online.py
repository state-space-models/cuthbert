import chex
import jax
import pytest

from kalman.online import predict, update
from tests.kalman.utils import generate_lgssm
from tests.kalman.test_filter import std_predict, std_update


@pytest.fixture(scope="module", autouse=True)
def config():
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", False)


@pytest.mark.parametrize("seed", [0, 42, 99, 123, 456])
@pytest.mark.parametrize("x_dim", [3])
def test_predict(seed, x_dim):
    m0, chol_P0, Fs, bs, chol_Qs = generate_lgssm(seed, x_dim, 1, 1)[:5]
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
