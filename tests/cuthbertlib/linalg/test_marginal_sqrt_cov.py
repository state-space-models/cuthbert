import jax
import jax.numpy as jnp
import pytest
from jax import random

from cuthbertlib.linalg.marginal_sqrt_cov import marginal_sqrt_cov


@pytest.fixture(scope="module", autouse=True)
def config():
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", False)


@pytest.mark.parametrize("seed", [0, 42, 99])
@pytest.mark.parametrize(
    "n,start,end",
    [
        (6, 0, 3),  # top-left block
        (6, 3, 6),  # bottom-right block
        (8, 2, 5),  # middle block
        (10, 1, 9),  # large block
    ],
)
def test_marginal_sqrt_cov(seed, n, start, end):
    key = random.key(seed)

    # Random lower-triangular joint square root
    L = jnp.tril(random.normal(key, (n, n)))

    # Extract marginal square root
    B = marginal_sqrt_cov(L, start, end)

    # Expected marginal covariance block
    Sigma = L @ L.T
    Sigma_block = Sigma[start:end, start:end]

    # Check B is lower triangular
    assert jnp.allclose(B, jnp.tril(B))

    # Check B B^T reproduces marginal covariance
    assert jnp.allclose(B @ B.T, Sigma_block)
