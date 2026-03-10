import jax
import jax.numpy as jnp
import pytest
from jax import random

from cuthbertlib.linalg.tria import tria


@pytest.fixture(scope="module", autouse=True)
def config():
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", False)


@pytest.mark.parametrize("seed", [0, 42, 99])
@pytest.mark.parametrize("shape", [(3, 2), (4, 4), (4, 5)])
def test_tria(seed, shape):
    key = random.key(seed)
    A = random.normal(key, shape)

    R = tria(A)

    # Check that R is lower triangular
    assert jnp.allclose(R, jnp.tril(R))

    # Check that R @ R.T = A @ A.T
    assert jnp.allclose(R @ R.T, A @ A.T)


def test_tria_jvp_preserves_gram_matrix_for_rank_deficient_input():
    A = jnp.array([[1.0, 0.0], [1.0, 0.0], [2.0, 3.0]])
    dA = jnp.array([[0.0, 0.0], [1.0, 2.0], [3.0, 4.0]])

    def gram_via_tria(x):
        R = tria(x)
        return R @ R.T

    def gram_direct(x):
        return x @ x.T

    gram_jvp_from_tria = jax.jvp(gram_via_tria, (A,), (dA,))[1]
    gram_jvp_direct = jax.jvp(gram_direct, (A,), (dA,))[1]

    assert jnp.allclose(gram_jvp_from_tria, gram_jvp_direct)
