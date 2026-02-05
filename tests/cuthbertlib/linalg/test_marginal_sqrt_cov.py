import jax
import jax.numpy as jnp
import pytest
from jax import random

from cuthbertlib.linalg.marginal_sqrt_cov import (
    block_marginal_sqrt_cov,
    marginal_sqrt_cov,
)


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
    B = marginal_sqrt_cov(L, start, end - start)

    # Expected marginal covariance block
    Sigma = L @ L.T
    Sigma_block = Sigma[start:end, start:end]

    # Check B is lower triangular
    assert jnp.allclose(B, jnp.tril(B))

    # Check B B^T reproduces marginal covariance
    assert jnp.allclose(B @ B.T, Sigma_block)


@pytest.mark.parametrize("seed", [0, 42])
@pytest.mark.parametrize(
    "n,subdim",
    [
        (6, 2),
        (6, 3),
        (8, 4),
        (9, 3),
    ],
)
def test_block_marginal_sqrt_cov(seed, n, subdim):
    key = random.key(seed)
    L = jnp.tril(random.normal(key, (n, n)))

    blocks = block_marginal_sqrt_cov(L, subdim)

    n_blocks = n // subdim
    assert blocks.shape == (n_blocks, subdim, subdim)

    Sigma = L @ L.T
    for i in range(n_blocks):
        start, end = i * subdim, (i + 1) * subdim
        Sigma_block = Sigma[start:end, start:end]
        assert jnp.allclose(
            blocks[i], jnp.tril(blocks[i])
        )  # Check that blocks are lower triangular
        assert jnp.allclose(
            blocks[i] @ blocks[i].T, Sigma_block
        )  # Check that blocks reproduce the marginal covariance
        assert jnp.allclose(
            blocks[i], marginal_sqrt_cov(L, start, subdim)
        )  # Check that blocks are the same as the marginal square root covariance
