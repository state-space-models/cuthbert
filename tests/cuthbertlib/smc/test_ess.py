import jax
import jax.numpy as jnp
import pytest
from jax import random

from cuthbertlib.smc.ess import ess


def test_ess_extremes():
    # Uniform weights
    log_weights = jnp.ones(10)
    assert jnp.allclose(ess(log_weights), 10.0, rtol=1e-8, atol=1e-8)

    # Complete degeneracy
    log_weights = jnp.full((10,), -jnp.inf)
    log_weights = log_weights.at[2].set(0.0)
    assert jnp.allclose(ess(log_weights), 1.0, rtol=1e-8, atol=1e-8)


@pytest.mark.parametrize("seed", [0, 42, 99, 123, 456])
def test_ess(seed):
    def ess_basic(weights):
        return 1.0 / jnp.sum(jnp.square(weights))

    key = random.key(seed)
    log_weights = random.normal(key, shape=(100,))
    weights = jax.nn.softmax(log_weights)
    assert jnp.allclose(ess(log_weights), ess_basic(weights), rtol=1e-6, atol=1e-8)
