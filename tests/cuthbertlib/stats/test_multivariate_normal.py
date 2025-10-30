import jax.numpy as jnp
import pytest
from jax import random
from jax.scipy.stats import multivariate_normal as jax_mvn
from jax.scipy.stats import norm as jax_norm

from cuthbertlib.stats import multivariate_normal


@pytest.mark.parametrize("seed", [0, 42, 99, 123, 456])
@pytest.mark.parametrize("nan_support", [True, False])
def test_multivariate_normal_logpdf(seed, nan_support):
    key = random.key(seed)
    dim = 3

    key, sub_key = random.split(key)
    chol_cov = random.uniform(sub_key, (dim, dim))
    chol_cov = chol_cov.at[jnp.triu_indices(dim, 1)].set(0.0)
    x = random.uniform(key, (dim,)) @ chol_cov.T

    logpdf = multivariate_normal.logpdf(
        x, jnp.zeros(dim), chol_cov, nan_support=nan_support
    )
    cov = chol_cov @ chol_cov.T
    des_logpdf = jax_mvn.logpdf(x, jnp.zeros(dim), cov)
    assert jnp.allclose(logpdf, des_logpdf, rtol=1e-4)


@pytest.mark.parametrize("seed", [0, 42, 99, 123, 456])
@pytest.mark.parametrize("nan_support", [True, False])
def test_multivariate_normal_diag(seed, nan_support):
    key = random.key(seed)
    dim = 3

    key, sub_key = random.split(key)
    chol_cov = random.uniform(sub_key, (dim,))
    x = random.uniform(key, (dim,)) * chol_cov

    logpdf = multivariate_normal.logpdf(
        x, jnp.zeros(dim), chol_cov, nan_support=nan_support
    )
    cov = jnp.diag(chol_cov**2)
    des_logpdf = jax_mvn.logpdf(x, jnp.zeros(dim), cov)
    assert jnp.allclose(logpdf, des_logpdf, rtol=1e-4)


@pytest.mark.parametrize("seed", [0, 42, 99, 123, 456])
@pytest.mark.parametrize("nan_support", [True, False])
def test_multivariate_normal_scalar_cov(seed, nan_support):
    key = random.key(seed)
    dim = 3

    key, sub_key = random.split(key)
    chol_cov = random.uniform(sub_key, ())
    x = random.uniform(key, (dim,)) * chol_cov

    logpdf = multivariate_normal.logpdf(
        x, jnp.zeros(dim), chol_cov, nan_support=nan_support
    )
    des_logpdf = jax_norm.logpdf(x, jnp.zeros(dim), chol_cov).sum()
    assert jnp.allclose(logpdf, des_logpdf, rtol=1e-4)


@pytest.mark.parametrize("seed", [0, 42, 99, 123, 456])
@pytest.mark.parametrize("nan_support", [True, False])
def test_multivariate_normal_scalar(seed, nan_support):
    key = random.key(seed)

    key, sub_key = random.split(key)
    chol_cov = random.uniform(sub_key, ())
    x = random.uniform(key, ()) * chol_cov

    logpdf = multivariate_normal.logpdf(x, 0.0, chol_cov, nan_support=nan_support)
    des_logpdf = jax_norm.logpdf(x, 0.0, chol_cov)
    assert jnp.allclose(logpdf, des_logpdf, rtol=1e-4)


@pytest.mark.parametrize("seed", [0, 42, 99, 123, 456])
@pytest.mark.parametrize("nan_support", [True, False])
def test_multivariate_normal_logpdf_scalar_mean_full_cov(seed, nan_support):
    """Test with multivariate x, scalar mean, and full covariance matrix."""
    key = random.key(seed)
    dim = 3

    key, sub_key = random.split(key)
    chol_cov = random.uniform(sub_key, (dim, dim))
    chol_cov = chol_cov.at[jnp.triu_indices(dim, 1)].set(0.0)
    mean_scalar = random.uniform(key, ())
    x = random.uniform(key, (dim,)) @ chol_cov.T + mean_scalar

    logpdf = multivariate_normal.logpdf(
        x, mean_scalar, chol_cov, nan_support=nan_support
    )
    cov = chol_cov @ chol_cov.T
    # JAX should broadcast scalar mean to match x's shape
    des_logpdf = jax_mvn.logpdf(x, jnp.full(dim, mean_scalar), cov)
    assert jnp.allclose(logpdf, des_logpdf, rtol=1e-4)


@pytest.mark.parametrize("seed", [0, 42, 99, 123, 456])
@pytest.mark.parametrize("nan_support", [True, False])
def test_multivariate_normal_logpdf_scalar_mean_diag_cov(seed, nan_support):
    """Test with multivariate x, scalar mean, and diagonal covariance."""
    key = random.key(seed)
    dim = 3

    key, sub_key = random.split(key)
    chol_cov = random.uniform(sub_key, (dim,))
    mean_scalar = random.uniform(key, ())
    x = random.uniform(key, (dim,)) * chol_cov + mean_scalar

    logpdf = multivariate_normal.logpdf(
        x, mean_scalar, chol_cov, nan_support=nan_support
    )
    cov = jnp.diag(chol_cov**2)
    # JAX should broadcast scalar mean to match x's shape
    des_logpdf = jax_mvn.logpdf(x, jnp.full(dim, mean_scalar), cov)
    assert jnp.allclose(logpdf, des_logpdf, rtol=1e-4)


@pytest.mark.parametrize("seed", [0, 42, 99, 123, 456])
@pytest.mark.parametrize("nan_support", [True, False])
def test_multivariate_normal_logpdf_scalar_mean_scalar_cov(seed, nan_support):
    """Test with multivariate x, scalar mean, and scalar covariance."""
    key = random.key(seed)
    dim = 3

    key, sub_key = random.split(key)
    chol_cov = random.uniform(sub_key, ())
    mean_scalar = random.uniform(key, ())
    x = random.uniform(key, (dim,)) * chol_cov + mean_scalar

    logpdf = multivariate_normal.logpdf(
        x, mean_scalar, chol_cov, nan_support=nan_support
    )
    # With scalar mean and scalar cov, this should be independent normals
    des_logpdf = jax_norm.logpdf(x, mean_scalar, chol_cov).sum()
    assert jnp.allclose(logpdf, des_logpdf, rtol=1e-4)


@pytest.mark.parametrize("seed", [0, 42, 99, 123, 456])
def test_multivariate_normal_logpdf_with_nans(seed):
    key = random.key(seed)
    dim = 3

    key, sub_key = random.split(key)
    chol_cov = random.uniform(sub_key, (dim, dim))
    chol_cov = chol_cov.at[jnp.triu_indices(dim, 1)].set(0.0)
    x = random.uniform(key, (dim,)) @ chol_cov.T
    x = x.at[1].set(jnp.nan)  # Introduce a NaN in the second element

    logpdf = multivariate_normal.logpdf(x, jnp.zeros(dim), chol_cov, nan_support=True)
    cov = chol_cov @ chol_cov.T
    cov = cov[~jnp.isnan(x), :][:, ~jnp.isnan(x)]  # Remove NaN rows and columns
    des_logpdf = jax_mvn.logpdf(x[~jnp.isnan(x)], jnp.zeros(dim - 1), cov)
    assert jnp.allclose(logpdf, des_logpdf, rtol=1e-4)


@pytest.mark.parametrize("seed", [0, 42, 99, 123, 456])
def test_multivariate_normal_logpdf_with_nans_diag(seed):
    key = random.key(seed)
    dim = 3

    key, sub_key = random.split(key)
    chol_cov = random.uniform(sub_key, (dim,))
    x = random.uniform(key, (dim,)) * chol_cov
    x = x.at[1].set(jnp.nan)  # Introduce a NaN in the second element

    logpdf = multivariate_normal.logpdf(x, jnp.zeros(dim), chol_cov, nan_support=True)
    cov = jnp.diag(chol_cov**2)
    cov = cov[~jnp.isnan(x), :][:, ~jnp.isnan(x)]  # Remove NaN rows and columns
    des_logpdf = jax_mvn.logpdf(x[~jnp.isnan(x)], jnp.zeros(dim - 1), cov)
    assert jnp.allclose(logpdf, des_logpdf, rtol=1e-4)


@pytest.mark.parametrize("seed", [0, 42, 99, 123, 456])
def test_multivariate_normal_logpdf_with_nans_scalar_cov(seed):
    key = random.key(seed)
    dim = 3

    key, sub_key = random.split(key)
    chol_cov = random.uniform(sub_key, ())
    x = random.uniform(key, (dim,)) * chol_cov
    x = x.at[1].set(jnp.nan)  # Introduce a NaN in the second element

    logpdf = multivariate_normal.logpdf(x, jnp.zeros(dim), chol_cov, nan_support=True)
    cov = chol_cov**2 * jnp.eye(dim)
    cov = cov[~jnp.isnan(x), :][:, ~jnp.isnan(x)]  # Remove NaN rows and columns
    des_logpdf = jax_mvn.logpdf(x[~jnp.isnan(x)], jnp.zeros(dim - 1), cov)
    assert jnp.allclose(logpdf, des_logpdf, rtol=1e-4)
