import chex
import jax
import jax.numpy as jnp
import pytest

from cuthbertlib.ensemble_kalman import gaspari_cohn


@pytest.fixture(scope="module", autouse=True)
def config():
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", False)


def test_gaspari_cohn():
    distances = jnp.array([0.0, 1.0, 1.5, 2.0, 3.0])
    expected = jnp.array([1.0, 5 / 24, 19 / 1152, 0.0, 0.0])

    actual = jax.jit(gaspari_cohn)(distances, 2.0)

    chex.assert_trees_all_close(actual, expected, rtol=1e-14, atol=1e-14)
