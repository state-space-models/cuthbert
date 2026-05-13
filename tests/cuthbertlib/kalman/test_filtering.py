import chex
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.linalg import solve_discrete_are

from cuthbertlib.kalman.filtering import (
    FilterScanElement,
    SteadyStateFilterParams,
    compute_steady_state_filter_params,
    predict,
    update,
)
from cuthbertlib.kalman.generate import generate_lgssm


@pytest.fixture(scope="module", autouse=True)
def config():
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", False)


def std_predict(m, P, F, c, Q):
    m = F @ m + c
    P = F @ P @ F.T + Q
    return m, P


def _std_update(m, P, H, d, R, y):
    residual = y - H @ m - d
    S = H @ P @ H.T + R
    K = jax.scipy.linalg.solve(S, H @ P, assume_a="pos").T

    m = m + K @ residual
    P = P - K @ S @ K.T

    ell = jax.scipy.stats.multivariate_normal.logpdf(
        residual, jnp.zeros(residual.shape[0]), S
    )
    return m, P, ell


def std_update(m, P, H, d, R, y):
    flag = jnp.isnan(y)
    m_update, P_update, ell = _std_update(
        m, P, H[~flag], d[~flag], R[~flag][:, ~flag], y[~flag]
    )
    return m_update, P_update, ell


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

    (m, chol_P), ell = update(m0, chol_P0, H, d, chol_R, y)
    P = chol_P @ chol_P.T

    chex.assert_numerical_grads(
        lambda *args: update(*args)[-1], (m0, chol_P0, H, d, chol_R, y), order=1
    )

    des_m, des_P, des_ell = std_update(m0, P0, H, d, R, y)

    chex.assert_trees_all_close((m, P), (des_m, des_P), rtol=1e-10)
    chex.assert_trees_all_close(ell, des_ell, rtol=1e-10)


# ---------------------------------------------------------------------------
# Helpers shared by the steady-state tests
# ---------------------------------------------------------------------------


def _make_lgssm(seed, nx, ny):
    """Return (F, c, chol_Q, H, d, chol_R, Q, R) for a stable random LG-SSM."""
    rng = np.random.default_rng(seed)
    # Stable transition: random orthogonal matrix scaled below 1
    _A = rng.standard_normal((nx, nx))
    U, _, Vt = np.linalg.svd(_A)
    F = 0.8 * (U @ Vt)
    c = rng.standard_normal(nx) * 0.1
    _L = rng.standard_normal((nx, nx))
    Q = _L @ _L.T / nx + np.eye(nx)
    chol_Q = np.linalg.cholesky(Q)
    H = rng.standard_normal((ny, nx)) / np.sqrt(nx)
    _M = rng.standard_normal((ny, ny))
    R = _M @ _M.T / ny + np.eye(ny)
    chol_R = np.linalg.cholesky(R)
    d = rng.standard_normal(ny) * 0.1
    return F, c, chol_Q, H, d, chol_R, Q, R


# ---------------------------------------------------------------------------
# Test 1 – compute_steady_state_filter_params against closed-form expressions
#
# The parallel-scan gain is derived from a *single* block-QR where the prior
# covariance is Q (process noise), NOT from a Riccati iteration.  It satisfies
# the closed-form:
#
#   K  = Q H^T (H Q H^T + R)^{-1}          (Kalman gain, prior = Q)
#   U  = chol(Q − K H Q)                    (posterior sqrt covariance)
#   A  = F − K H F                          (corrected transition)
#   Z_filter = (H F)^T S^{−T}              (pre-padded information gain)
#   Psi11_inv = chol_S^{−1}                 (precomputed inverse)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", [0, 42, 99])
@pytest.mark.parametrize("nx, ny", [(3, 2), (4, 4), (5, 3)])
def test_compute_steady_state_filter_params_closed_form(seed, nx, ny):
    F, c, chol_Q, H, d, chol_R, Q, R = _make_lgssm(seed, nx, ny)

    ss = compute_steady_state_filter_params(
        jnp.array(F),
        jnp.array(chol_Q),
        jnp.array(H),
        jnp.array(chol_R),
    )

    # closed-form quantities
    S = H @ Q @ H.T + R  # innovation covariance
    K_ref = Q @ H.T @ np.linalg.inv(S)
    P_post_ref = Q - K_ref @ H @ Q  # posterior covariance
    HF = H @ F

    # gain
    chex.assert_trees_all_close(ss.K, jnp.array(K_ref), atol=1e-10)

    # innovation Cholesky: chol_S @ chol_S.T == S
    S_reconstructed = ss.chol_S @ ss.chol_S.T
    chex.assert_trees_all_close(S_reconstructed, jnp.array(S), atol=1e-10)

    # Psi11_inv is the inverse of chol_S
    I_check = ss.chol_S @ ss.Psi11_inv
    chex.assert_trees_all_close(I_check, jnp.eye(ny), atol=1e-10)

    # Z_filter = (H F)^T chol_S^{-T}
    Z_filter_ref = HF.T @ np.linalg.inv(np.array(ss.chol_S)).T
    chex.assert_trees_all_close(ss.Z_filter, jnp.array(Z_filter_ref), atol=1e-10)

    # posterior covariance: U @ U.T == P_post
    P_post_reconstructed = ss.elem.U @ ss.elem.U.T
    chex.assert_trees_all_close(P_post_reconstructed, jnp.array(P_post_ref), atol=1e-10)

    # corrected transition: A = F - K H F
    A_ref = F - K_ref @ HF
    chex.assert_trees_all_close(ss.elem.A, jnp.array(A_ref), atol=1e-10)


# ---------------------------------------------------------------------------
# Test 2 – Riccati-based SteadyStateFilterParams for the sequential update
#
# The TRUE steady-state gain for the sequential filter is the Riccati K,
# obtained by solving the discrete algebraic Riccati equation (DARE):
#
#   solve_discrete_are(F, H.T, Q, R)  →  P_pred_ss
#   K_riccati = P_pred_ss H^T (H P_pred_ss H^T + R)^{-1}
#
# We construct a SteadyStateFilterParams from Riccati quantities and verify
# that update(..., steady_state_params=ss_riccati) produces the same posterior
# mean as the standard update once the running filter has converged.
# ---------------------------------------------------------------------------


def _riccati_steady_state_params(F, Q, H, R):
    """Build SteadyStateFilterParams from the DARE solution (Riccati K).

    ``scipy.linalg.solve_discrete_are(a, b, q, r)`` solves
    ``a.T X a - X - (a.T X b) inv(r + b.T X b) (b.T X a) + q = 0``,
    so to obtain the Kalman predictive-covariance DARE
    ``P = F P_post F.T + Q - F P_post H.T inv(S) H P_post F.T``
    the first argument must be ``F.T``, not ``F``.
    """
    # DARE: solve_discrete_are(F.T, H.T, Q, R) returns P_pred_ss
    P_pred = solve_discrete_are(F.T, H.T, Q, R)
    S = H @ P_pred @ H.T + R
    chol_S = np.linalg.cholesky(S)
    Psi11_inv = np.linalg.inv(chol_S)
    K = P_pred @ H.T @ Psi11_inv.T @ Psi11_inv  # K = P_pred H^T S^{-1}
    P_post = P_pred - K @ H @ P_pred
    chol_P_post = np.linalg.cholesky(P_post)
    # Z_filter and A are needed only by associative_params_single, not update;
    # populate them correctly for completeness.
    HF = H @ F
    Z_filter = HF.T @ Psi11_inv.T
    nx = F.shape[0]
    ny = H.shape[0]
    if nx > ny:
        Z = np.concatenate([Z_filter, np.zeros((nx, nx - ny))], axis=1)
    else:
        from scipy.linalg import qr

        _, Z = qr(Z_filter.T, mode="economic")
        Z = Z.T
    A = F - K @ HF
    dummy = np.zeros(nx)
    elem = FilterScanElement(
        A=jnp.array(A),
        b=jnp.array(dummy),
        U=jnp.array(chol_P_post),
        eta=jnp.array(dummy),
        Z=jnp.array(Z),
        ell=jnp.array(0.0),
    )
    return SteadyStateFilterParams(
        elem=elem,
        K=jnp.array(K),
        chol_S=jnp.array(chol_S),
        Psi11_inv=jnp.array(Psi11_inv),
        Z_filter=jnp.array(Z_filter),
    )


@pytest.mark.parametrize("seed", [0, 42, 99])
@pytest.mark.parametrize("nx, ny", [(3, 2), (4, 4), (5, 3)])
def test_update_steady_state_riccati(seed, nx, ny):
    """Riccati steady-state update matches the standard update initialized at the Riccati covariance.

    When the standard filter is started at the Riccati posterior covariance, its
    running K is exactly the Riccati K at every subsequent step.  The two filters
    must therefore produce identical posterior means.
    """
    F, c, chol_Q, H, d, chol_R, Q, R = _make_lgssm(seed, nx, ny)
    ss_riccati = _riccati_steady_state_params(F, Q, H, R)

    rng = np.random.default_rng(seed + 1)

    # Both filters start from the same arbitrary mean and the Riccati posterior covariance.
    # With chol_P = chol_P_riccati, one predict+update cycle in the standard filter
    # uses exactly K_riccati, so both means must agree at every step.
    m0 = jnp.array(rng.standard_normal(nx))
    chol_P_riccati = ss_riccati.elem.U

    m_std = m0
    chol_P_std = chol_P_riccati
    m_ss = m0

    for _ in range(20):
        # standard: full predict + update
        m_std, chol_P_std = predict(
            m_std,
            chol_P_std,
            jnp.array(F),
            jnp.array(c),
            jnp.array(chol_Q),
        )
        y = jnp.array(rng.standard_normal(ny))
        (m_std, chol_P_std), _ = update(
            m_std,
            chol_P_std,
            jnp.array(H),
            jnp.array(d),
            jnp.array(chol_R),
            y,
        )

        # steady-state: same predict step, then update with fixed Riccati params
        m_pred_ss = jnp.array(F @ np.array(m_ss) + c)
        (m_ss, _), _ = update(
            m_pred_ss,
            chol_P_riccati,
            jnp.array(H),
            jnp.array(d),
            jnp.array(chol_R),
            y,
            steady_state_params=ss_riccati,
        )

        chex.assert_trees_all_close(m_ss, m_std, atol=1e-10)
