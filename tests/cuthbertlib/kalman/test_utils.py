import jax.numpy as jnp
import pytest
from jax import random
from jax.scipy.stats import multivariate_normal

from cuthbertlib.kalman.utils import mvn_logpdf


@pytest.mark.parametrize("seed", [0, 42, 99, 123, 456])
def test_mvn_logpdf(seed):
    key = random.key(seed)
    dim = 3

    key, sub_key = random.split(key)
    chol_cov = random.uniform(sub_key, (dim, dim))
    chol_cov = chol_cov.at[jnp.triu_indices(dim, 1)].set(0.0)
    x = random.uniform(key, (dim,)) @ chol_cov.T

    logpdf = mvn_logpdf(x, chol_cov)
    cov = chol_cov @ chol_cov.T
    des_logpdf = multivariate_normal.logpdf(x, jnp.zeros(cov.shape[0]), cov)
    assert jnp.allclose(logpdf, des_logpdf, rtol=1e-4)
