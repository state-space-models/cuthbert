import chex
import jax
import jax.numpy as jnp
import pytest
from absl.testing import parameterized
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


class TestMarginalSqrtCov(chex.TestCase):
    @chex.variants(with_jit=True, without_jit=True)
    @parameterized.product(
        seed=[0, 42, 99],
        block=[
            (6, 0, 3),  # top-left block
            (6, 3, 6),  # bottom-right block
            (8, 2, 5),  # middle block
            (10, 1, 9),  # large block
        ],
    )
    def test_marginal_sqrt_cov(self, seed, block):
        key = random.key(seed)
        n, start, end = block
        size = end - start

        # Random lower-triangular joint square root
        L = jnp.tril(random.normal(key, (n, n)))

        # Extract marginal square root
        B = self.variant(
            marginal_sqrt_cov,
            static_argnames=("size"),
        )(L, start, size)

        # Expected marginal covariance block
        Sigma = L @ L.T
        Sigma_block = Sigma[start:end, start:end]

        # Check B is lower triangular
        chex.assert_trees_all_close(B, jnp.tril(B))

        # Check B B^T reproduces marginal covariance
        chex.assert_trees_all_close(B @ B.T, Sigma_block)


class TestBlockMarginalSqrtCov(chex.TestCase):
    @chex.variants(with_jit=True, without_jit=True)
    @parameterized.product(
        seed=[0, 42],
        n_subdim=[
            (6, 2),
            (6, 3),
            (8, 4),
            (9, 3),
        ],
    )
    def test_block_marginal_sqrt_cov(self, seed, n_subdim):
        n, subdim = n_subdim
        key = random.key(seed)
        L = jnp.tril(random.normal(key, (n, n)))

        blocks = self.variant(
            block_marginal_sqrt_cov,
            static_argnames=("subdim",),
        )(L, subdim=subdim)

        n_blocks = n // subdim
        chex.assert_equal(blocks.shape, (n_blocks, subdim, subdim))

        Sigma = L @ L.T
        for i in range(n_blocks):
            start, end = i * subdim, (i + 1) * subdim
            Sigma_block = Sigma[start:end, start:end]
            chex.assert_trees_all_close(
                blocks[i], jnp.tril(blocks[i])
            )  # Check that blocks are lower triangular
            chex.assert_trees_all_close(
                blocks[i] @ blocks[i].T, Sigma_block
            )  # Check that blocks reproduce the marginal covariance
            chex.assert_trees_all_close(
                blocks[i], marginal_sqrt_cov(L, start, subdim)
            )  # Check that blocks match marginal_sqrt_cov
