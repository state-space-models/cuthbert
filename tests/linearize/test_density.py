import pytest
import chex
import numpy as np
import jax
from jax import numpy as jnp
from jax.scipy.stats import multivariate_normal

from linearize import linearize_density, linearize_density_given_chol_cov
from tests.kalman.utils import (
    generate_init_model,
    generate_trans_model,
    generate_obs_model,
)


@pytest.fixture(scope="module", autouse=True)
def config():
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", False)


@pytest.mark.parametrize("seed", [0, 42, 99, 123, 456])
@pytest.mark.parametrize("x_dim", [3])
def test_linearize_init(seed, x_dim) -> None:
    rng = np.random.default_rng(seed)

    mean, chol_P0 = generate_init_model(rng, x_dim)

    linearization_point = rng.normal(size=x_dim)

    def init_log_density(x):
        return jnp.asarray(multivariate_normal.logpdf(x, mean, chol_P0 @ chol_P0.T))

    mat_given_chol_cov, shift_given_chol_cov = linearize_density_given_chol_cov(
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

    mat, shift, chol_cov = linearize_density(
        lambda _, x: init_log_density(x), linearization_point, linearization_point
    )

    chex.assert_trees_all_close(
        (mat, shift, chol_cov @ chol_cov.T),
        (jnp.zeros((x_dim, x_dim)), mean, chol_P0 @ chol_P0.T),
        rtol=1e-7,
    )


@pytest.mark.parametrize("seed", [0, 42, 99, 123, 456])
@pytest.mark.parametrize("x_dim", [3])
def test_linearize_trans_model(seed, x_dim) -> None:
    rng = np.random.default_rng(seed)

    F, c, chol_Q = generate_trans_model(rng, x_dim)

    linearization_point_prev = rng.normal(size=x_dim)
    linearization_point = rng.normal(size=x_dim)

    def dynamics_log_density(x_prev, x):
        return jnp.asarray(
            multivariate_normal.logpdf(x, F @ x_prev + c, chol_Q @ chol_Q.T)
        )

    dynamics_mat_given_chol_cov, dynamics_shift_given_chol_cov = (
        linearize_density_given_chol_cov(
            dynamics_log_density, linearization_point_prev, linearization_point, chol_Q
        )
    )

    chex.assert_trees_all_close(
        (dynamics_mat_given_chol_cov, dynamics_shift_given_chol_cov),
        (F, c),
        rtol=1e-8,
    )

    dynamics_mat, dynamics_shift, dynamics_chol_cov = linearize_density(
        dynamics_log_density, linearization_point_prev, linearization_point
    )

    chex.assert_trees_all_close(
        (dynamics_mat, dynamics_shift, dynamics_chol_cov @ dynamics_chol_cov.T),
        (F, c, chol_Q @ chol_Q.T),
        rtol=1e-7,
    )


@pytest.mark.parametrize("seed", [0, 42, 99, 123, 456])
@pytest.mark.parametrize("x_dim", [3])
@pytest.mark.parametrize("y_dim", [1, 2])
def test_linearize_obs_model(seed, x_dim, y_dim) -> None:
    rng = np.random.default_rng(seed)

    H, d, chol_R, y = generate_obs_model(rng, x_dim, y_dim)

    linearization_point = rng.normal(size=x_dim)

    def obs_log_density(x, y):
        return jnp.asarray(multivariate_normal.logpdf(y, H @ x + d, chol_R @ chol_R.T))

    obs_mat_given_chol_cov, obs_shift_given_chol_cov = linearize_density_given_chol_cov(
        obs_log_density, linearization_point, y, chol_R
    )

    chex.assert_trees_all_close(
        (obs_mat_given_chol_cov, obs_shift_given_chol_cov),
        (H, d),
        rtol=1e-8,
    )

    obs_mat, obs_shift, obs_chol_cov = linearize_density(
        obs_log_density, linearization_point, y
    )

    chex.assert_trees_all_close(
        (obs_mat, obs_shift, obs_chol_cov @ obs_chol_cov.T),
        (H, d, chol_R @ chol_R.T),
        rtol=1e-7,
    )
