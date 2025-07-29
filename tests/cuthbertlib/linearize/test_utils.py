import chex
import jax
import numpy as np
import pytest
from jax import numpy as jnp

from cuthbertlib.linearize import symmetric_inv_sqrt


@pytest.fixture(scope="module", autouse=True)
def config():
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", False)


@pytest.mark.parametrize("seed", [0, 42, 99, 123, 456])
@pytest.mark.parametrize("x_dim", [3])
def test_inv_sqrt_positive_definite(seed, x_dim) -> None:
    rng = np.random.default_rng(seed)

    A = rng.normal(size=(x_dim, x_dim))
    A = A @ A.T

    L = symmetric_inv_sqrt(A)

    chex.assert_trees_all_close(L @ L.T, jnp.linalg.inv(A))
    chex.assert_trees_all_close(jnp.linalg.inv(L @ L.T), A)


@pytest.mark.parametrize("seed", [0, 42, 99, 123, 456])
@pytest.mark.parametrize("x_dim", [3])
def test_inv_sqrt_singular_indefinite(seed, x_dim) -> None:
    rng = np.random.default_rng(seed)

    A = rng.normal(size=(x_dim, x_dim))
    A = A @ A.T

    A[0, 0] = -1
    A[1, 1] = 0.0

    L = symmetric_inv_sqrt(A)

    assert L.shape == (x_dim, x_dim)
    assert jnp.array_equal(jnp.tril(L), L)
