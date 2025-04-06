import jax.numpy as jnp
import pytest
from jax import random

from cuthbertlib.linalg.tria import tria


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
