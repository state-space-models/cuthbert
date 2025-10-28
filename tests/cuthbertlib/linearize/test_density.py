import chex
import jax
import pytest
from jax import numpy as jnp
from jax.scipy.stats import multivariate_normal

from cuthbertlib.kalman.generate import generate_init_model, generate_obs_model
from cuthbertlib.linearize import (
    linearize_log_density,
    linearize_log_density_given_chol_cov,
)


@pytest.fixture(scope="module", autouse=True)
def config():
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", False)


@pytest.mark.parametrize("seed", [0, 42, 99, 123, 456])
@pytest.mark.parametrize("x_dim", [3])
def test_linearize_log_density_init(seed, x_dim) -> None:
    key = jax.random.key(seed)
    init_key, lin_key = jax.random.split(key)
    mean, chol_P0 = generate_init_model(init_key, x_dim)

    linearization_point = jax.random.normal(lin_key, (x_dim,))

    def init_log_density(x):
        return jnp.asarray(multivariate_normal.logpdf(x, mean, chol_P0 @ chol_P0.T))

    mat_given_chol_cov, shift_given_chol_cov = linearize_log_density_given_chol_cov(
        lambda _, x: init_log_density(x),
        linearization_point,
        linearization_point,
        chol_P0,
    )

    chex.assert_trees_all_close(
        (mat_given_chol_cov, shift_given_chol_cov),
        (jnp.zeros((x_dim, x_dim)), mean),
        rtol=1e-8,
    )

    mat, shift, chol_cov = linearize_log_density(
        lambda _, x: init_log_density(x), linearization_point, linearization_point
    )

    chex.assert_trees_all_close(
        (mat, shift, chol_cov @ chol_cov.T),
        (jnp.zeros((x_dim, x_dim)), mean, chol_P0 @ chol_P0.T),
        rtol=1e-7,
    )

    # Test with auxiliary value and chol_cov
    def init_log_density_aux(_, x):
        return init_log_density(x), {"aux": x}

    mat, shift, aux = linearize_log_density_given_chol_cov(
        init_log_density_aux,
        linearization_point,
        linearization_point,
        chol_P0,
        has_aux=True,
    )

    chex.assert_trees_all_close(
        (mat, shift, aux),
        (
            jnp.zeros((x_dim, x_dim)),
            mean,
            {"aux": linearization_point},
        ),
        rtol=1e-7,
    )

    # Test with auxiliary value
    mat, shift, chol_cov, aux = linearize_log_density(
        init_log_density_aux,
        linearization_point,
        linearization_point,
        has_aux=True,
    )

    chex.assert_trees_all_close(
        (mat, shift, chol_cov @ chol_cov.T, aux),
        (
            jnp.zeros((x_dim, x_dim)),
            mean,
            chol_P0 @ chol_P0.T,
            {"aux": linearization_point},
        ),
        rtol=1e-7,
    )


@pytest.mark.parametrize("seed", [0, 42, 99, 123, 456])
@pytest.mark.parametrize("x_dim", [3])
@pytest.mark.parametrize("y_dim", [1, 2])
def test_linearize_log_density(seed, x_dim, y_dim) -> None:
    key = jax.random.key(seed)
    param_key, obs_key, lin_key = jax.random.split(key, 3)
    H, d, chol_R = generate_obs_model(param_key, x_dim, y_dim)
    y = jax.random.normal(obs_key, (y_dim,))

    linearization_point = jax.random.normal(lin_key, (x_dim,))

    def log_density(x, y):
        return jnp.asarray(multivariate_normal.logpdf(y, H @ x + d, chol_R @ chol_R.T))

    mat_given_chol_cov, shift_given_chol_cov = linearize_log_density_given_chol_cov(
        log_density, linearization_point, y, chol_R
    )

    chex.assert_trees_all_close(
        (mat_given_chol_cov, shift_given_chol_cov),
        (H, d),
        rtol=1e-8,
    )

    mat, shift, chol_cov = linearize_log_density(log_density, linearization_point, y)

    chex.assert_trees_all_close(
        (mat, shift, chol_cov @ chol_cov.T),
        (H, d, chol_R @ chol_R.T),
        rtol=1e-7,
    )

    # Test with auxiliary value and chol_cov
    def log_density_aux(x, y):
        return log_density(x, y), {"aux": x}

    mat, shift, aux = linearize_log_density_given_chol_cov(
        log_density_aux, linearization_point, y, chol_R, has_aux=True
    )

    chex.assert_trees_all_close(
        (mat, shift, aux),
        (H, d, {"aux": linearization_point}),
        rtol=1e-7,
    )

    # Test with auxiliary value
    mat, shift, chol_cov, aux = linearize_log_density(
        log_density_aux, linearization_point, y, has_aux=True
    )

    chex.assert_trees_all_close(
        (mat, shift, chol_cov @ chol_cov.T, aux),
        (H, d, chol_R @ chol_R.T, {"aux": linearization_point}),
        rtol=1e-7,
    )
