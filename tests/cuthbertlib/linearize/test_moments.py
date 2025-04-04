from functools import partial
import chex
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from cuthbertlib.linearize import linearize_moments
from cuthbertlib.linearize.utils import tria


@pytest.fixture(scope="session", autouse=True)
def config():
    jax.config.update("jax_enable_x64", True)
    jax.config.update("jax_debug_nans", False)


def linear_function(x, a, c):
    return a @ x + c


def linear_conditional_mean(x, q, a, b, c):
    return a @ x + b @ q + c


def linear_conditional_cov(_x, b, cov_q):
    return b @ cov_q @ b.T


def linear_conditional_chol(_x, b, chol_q):
    ny, nq = b.shape
    if ny > nq:
        res = jnp.concatenate([b @ chol_q, jnp.zeros((ny, ny - nq))], axis=1)
    else:
        res = tria(b @ chol_q)
    return res


@pytest.mark.parametrize("dim_x", [1, 3])
@pytest.mark.parametrize("dim_q", [1, 2, 3])
@pytest.mark.parametrize("seed", [0, 42])
def test_linear_conditional(dim_x, dim_q, seed):
    np.random.seed(seed)
    a = np.random.randn(dim_x, dim_x)
    b = np.random.randn(dim_x, dim_q)
    c = np.random.randn(dim_x)

    m_x = np.random.randn(dim_x)
    m_q = np.random.randn(dim_q)

    chol_x = np.random.rand(dim_x, dim_x)
    chol_x[np.triu_indices(dim_x, 1)] = 0

    chol_q = np.random.rand(dim_q, dim_q)
    chol_q[np.triu_indices(dim_q, 1)] = 0

    E_f = partial(linear_conditional_mean, q=m_q, a=a, b=b, c=c)
    chol_f = partial(linear_conditional_chol, b=b, chol_q=chol_q)

    F_x, remainder, Q_lin = linearize_moments(E_f, chol_f, m_x)
    Q_lin = Q_lin @ Q_lin.T
    x_prime = np.random.randn(dim_x)

    expected = linear_conditional_mean(x_prime, m_q, a, b, c)
    actual = F_x @ x_prime + remainder
    expected_Q = (b @ chol_q) @ (b @ chol_q).T
    chex.assert_trees_all_close(
        (F_x, actual, Q_lin),
        (a, expected, expected_Q),
        rtol=1e-7,
    )
