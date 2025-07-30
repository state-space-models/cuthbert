from functools import partial

import chex
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from cuthbertlib.linearize import linearize_taylor


@pytest.fixture(scope="session", autouse=True)
def config():
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", False)


def quadratic_log_potential(x, m, L):
    return -0.5 * (x - m).T @ jnp.linalg.inv(L @ L.T) @ (x - m)


@pytest.mark.parametrize("dim_x", [1, 3])
@pytest.mark.parametrize("seed", [0, 42])
def test_linearize_taylor(dim_x, seed):
    np.random.seed(seed)

    m = np.random.randn(dim_x)
    L = np.random.randn(dim_x, dim_x)
    L = np.tril(L)

    log_potential = partial(quadratic_log_potential, m=m, L=L)

    x = np.random.randn(dim_x)
    m_lin, L_lin = linearize_taylor(log_potential, x)

    chex.assert_trees_all_close(
        (m_lin, L_lin @ L_lin.T),
        (m, L @ L.T),
        rtol=1e-7,
    )

    # Test with auxiliary value
    def log_potential_aux(x):
        return log_potential(x), {"aux": x}

    m_lin, L_lin, aux = linearize_taylor(log_potential_aux, x, has_aux=True)

    chex.assert_trees_all_close(
        (m_lin, L_lin @ L_lin.T, aux),
        (m, L @ L.T, {"aux": x}),
        rtol=1e-7,
    )
