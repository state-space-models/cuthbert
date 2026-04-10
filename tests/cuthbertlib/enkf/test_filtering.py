import chex
import jax
import jax.numpy as jnp
import pytest
from jax import random

from cuthbertlib.enkf.filtering import predict, update
from cuthbertlib.kalman.filtering import update as kalman_update
from cuthbertlib.kalman.generate import generate_lgssm


@pytest.fixture(scope="module", autouse=True)
def config():
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", False)


@pytest.mark.parametrize("seed", [0, 42, 99, 123, 456])
@pytest.mark.parametrize("x_dim", [3])
def test_predict_identity(seed, x_dim):
    """Identity dynamics with zero noise should preserve the ensemble."""
    key = random.key(seed)
    N = 100
    ensemble = random.normal(key, (N, x_dim))
    predicted = predict(key, ensemble, lambda x, key: x, inflation=0.0)
    chex.assert_trees_all_close(predicted, ensemble, atol=1e-12)


@pytest.mark.parametrize("seed", [0, 42, 99, 123, 456])
@pytest.mark.parametrize("x_dim", [3])
def test_predict_linear(seed, x_dim):
    """Linear dynamics should shift the ensemble mean correctly."""
    key = random.key(seed)
    N = 1_000_000

    lgssm = generate_lgssm(seed, x_dim, 1, 1)
    F, c, chol_Q = lgssm[2][0], lgssm[3][0], lgssm[4][0]

    # Generate ensemble from known distribution
    m0, chol_P0 = lgssm[0], lgssm[1]
    keys = random.split(key, N)
    ensemble = jax.vmap(lambda k: m0 + chol_P0 @ random.normal(k, (x_dim,)))(keys)

    predicted = predict(
        random.key(seed + 100),
        ensemble,
        lambda x, key: F @ x + c + chol_Q @ random.normal(key, (x_dim,)),
        inflation=0.0,
    )

    # Expected mean: F @ m0 + c (noise is zero-mean)
    expected_mean = F @ m0 + c
    pred_mean = jnp.mean(predicted, axis=0)
    chex.assert_trees_all_close(pred_mean, expected_mean, atol=1e-2)


@pytest.mark.parametrize("seed", [0, 42, 99, 123, 456])
@pytest.mark.parametrize("x_dim", [3])
def test_predict_inflation(seed, x_dim):
    """Inflation should scale deviations from the mean."""
    key = random.key(seed)
    N = 100
    ensemble = random.normal(key, (N, x_dim))
    delta = 0.05

    predicted = predict(key, ensemble, lambda x, key: x, inflation=delta)

    mean = jnp.mean(ensemble, axis=0)
    expected = mean + (1 + delta) * (ensemble - mean)
    chex.assert_trees_all_close(predicted, expected, atol=1e-12)


@pytest.mark.parametrize("seed", [0, 42, 99, 123, 456])
@pytest.mark.parametrize("x_dim", [3])
@pytest.mark.parametrize("y_dim", [1, 2])
def test_update_linear_gaussian(seed, x_dim, y_dim):
    """EnKF update should converge to Kalman update for large ensemble."""
    key = random.key(seed)
    N = 100_000

    lgssm = generate_lgssm(seed, x_dim, y_dim, 1)
    m0, chol_P0 = lgssm[0], lgssm[1]
    H, d, chol_R, y = lgssm[5][0], lgssm[6][0], lgssm[7][0], lgssm[8][0]

    # Generate large ensemble
    keys = random.split(key, N)
    ensemble = jax.vmap(lambda k: m0 + chol_P0 @ random.normal(k, (x_dim,)))(keys)

    # EnKF update
    updated, ll = update(
        random.key(seed + 200),
        ensemble,
        lambda x: H @ x + d,
        chol_R,
        y,
        perturbed_obs=True,
    )

    enkf_mean = jnp.mean(updated, axis=0)
    enkf_dev = updated - enkf_mean
    enkf_cov = enkf_dev.T @ enkf_dev / (N - 1)

    # Kalman update
    (kalman_mean, kalman_chol_cov), kalman_ll = kalman_update(
        m0, chol_P0, H, d, chol_R, y
    )
    kalman_cov = kalman_chol_cov @ kalman_chol_cov.T

    chex.assert_trees_all_close(enkf_mean, kalman_mean, atol=1e-2)
    chex.assert_trees_all_close(enkf_cov, kalman_cov, atol=1e-2)


@pytest.mark.parametrize("seed", [0, 42, 99, 123, 456])
@pytest.mark.parametrize("x_dim", [3])
@pytest.mark.parametrize("y_dim", [1, 2])
def test_update_perturbed_vs_unperturbed(seed, x_dim, y_dim):
    """Both perturbed and unperturbed modes should produce correct shapes."""
    key = random.key(seed)
    N = 100

    lgssm = generate_lgssm(seed, x_dim, y_dim, 1)
    m0, chol_P0 = lgssm[0], lgssm[1]
    H, d, chol_R, y = lgssm[5][0], lgssm[6][0], lgssm[7][0], lgssm[8][0]

    keys = random.split(key, N)
    ensemble = jax.vmap(lambda k: m0 + chol_P0 @ random.normal(k, (x_dim,)))(keys)

    for perturbed in [True, False]:
        updated, ll = update(
            random.key(seed + 300),
            ensemble,
            lambda x: H @ x + d,
            chol_R,
            y,
            perturbed_obs=perturbed,
        )
        chex.assert_shape(updated, (N, x_dim))
        chex.assert_shape(ll, ())
        assert jnp.isfinite(ll)


@pytest.mark.parametrize("seed", [0, 42, 99, 123, 456])
@pytest.mark.parametrize("x_dim", [3])
@pytest.mark.parametrize("y_dim", [1, 2])
def test_update_log_likelihood(seed, x_dim, y_dim):
    """Log-likelihood should match MVN logpdf evaluated at the ensemble mean prediction."""
    key = random.key(seed)
    N = 100_000

    lgssm = generate_lgssm(seed, x_dim, y_dim, 1)
    m0, chol_P0 = lgssm[0], lgssm[1]
    H, d, chol_R, y = lgssm[5][0], lgssm[6][0], lgssm[7][0], lgssm[8][0]

    keys = random.split(key, N)
    ensemble = jax.vmap(lambda k: m0 + chol_P0 @ random.normal(k, (x_dim,)))(keys)

    _, ll = update(
        random.key(seed + 400),
        ensemble,
        lambda x: H @ x + d,
        chol_R,
        y,
        perturbed_obs=True,
    )

    # Reference: Kalman filter log-likelihood
    _, kalman_ll = kalman_update(m0, chol_P0, H, d, chol_R, y)

    chex.assert_trees_all_close(ll, kalman_ll, atol=2e-2)


@pytest.mark.parametrize("seed", [0, 42, 99, 123, 456])
@pytest.mark.parametrize("x_dim", [3])
@pytest.mark.parametrize("y_dim", [1, 2])
def test_update_nan_observation(seed, x_dim, y_dim):
    """NaN observation should return ensemble unchanged with zero log-likelihood."""
    key = random.key(seed)
    N = 100

    lgssm = generate_lgssm(seed, x_dim, y_dim, 1)
    m0, chol_P0 = lgssm[0], lgssm[1]
    H, d, chol_R = lgssm[5][0], lgssm[6][0], lgssm[7][0]
    y_nan = jnp.full(y_dim, jnp.nan)

    keys = random.split(key, N)
    ensemble = jax.vmap(lambda k: m0 + chol_P0 @ random.normal(k, (x_dim,)))(keys)

    updated, ll = update(
        random.key(seed + 500),
        ensemble,
        lambda x: H @ x + d,
        chol_R,
        y_nan,
        perturbed_obs=True,
    )

    chex.assert_trees_all_close(updated, ensemble, atol=1e-12)
    chex.assert_trees_all_close(ll, jnp.array(0.0), atol=1e-12)


@pytest.mark.parametrize("seed", [0, 42, 99, 123, 456])
@pytest.mark.parametrize("x_dim", [3])
@pytest.mark.parametrize("y_dim", [2, 3])
def test_update_partial_nan_observation(seed, x_dim, y_dim):
    """Partially-NaN observations should match Kalman missing-dimension behavior."""
    key = random.key(seed)
    N = 100_000

    lgssm = generate_lgssm(seed, x_dim, y_dim, 1)
    m0, chol_P0 = lgssm[0], lgssm[1]
    H, d, chol_R, y = lgssm[5][0], lgssm[6][0], lgssm[7][0], lgssm[8][0]
    y = y.at[0].set(jnp.nan)

    keys = random.split(key, N)
    ensemble = jax.vmap(lambda k: m0 + chol_P0 @ random.normal(k, (x_dim,)))(keys)

    updated, ll = update(
        random.key(seed + 600),
        ensemble,
        lambda x: H @ x + d,
        chol_R,
        y,
        perturbed_obs=True,
    )
    assert jnp.isfinite(ll)
    assert jnp.all(jnp.isfinite(updated))

    enkf_mean = jnp.mean(updated, axis=0)
    enkf_dev = updated - enkf_mean
    enkf_cov = enkf_dev.T @ enkf_dev / (N - 1)

    (kalman_mean, kalman_chol_cov), kalman_ll = kalman_update(
        m0, chol_P0, H, d, chol_R, y
    )
    kalman_cov = kalman_chol_cov @ kalman_chol_cov.T

    chex.assert_trees_all_close(enkf_mean, kalman_mean, atol=2e-2)
    chex.assert_trees_all_close(enkf_cov, kalman_cov, atol=3e-2)
    chex.assert_trees_all_close(ll, kalman_ll, atol=3e-2)
