import itertools
from functools import partial

import chex
import jax
import jax.numpy as jnp
import numpy.testing as npt
from absl.testing import parameterized

from quadrature import cubature, unscented, gauss_hermite
from quadrature.linearize import functional, conditional_moments

QUADRATURES = [
    {"name": "cubature", "params": []},
    {"name": "unscented", "params": (1.0, 0.0, None)},
    {"name": "gauss_hermite", "params": []},
]
DIMENSIONS = [1, 2, 3]


def linear_function(x, a, b):
    return a @ x + b


def linear_conditional_mean(x, a, b):
    return a @ x + b


def linear_conditional_cov(_x, cov_q):
    return cov_q


def linear_conditional_chol(_x, chol_q):
    return chol_q


def get_quadrature(quadrature, dim, *params):
    if quadrature["name"] == "cubature":
        return cubature.weights(dim)
    elif quadrature["name"] == "unscented":
        return unscented.weights(dim, *quadrature["params"])
    elif quadrature["name"] == "gauss_hermite":
        return gauss_hermite.weights(dim)
    else:
        raise ValueError(f'Unknown quadrature method: {quadrature["name"]}')


class TestLinearize(chex.TestCase):
    def setUp(self):
        super().setUp()

    # @chex.all_variants(with_pmap=False, without_jit=False)
    @parameterized.parameters(
        itertools.product(
            [0, 1, 2, 3, 4],
            QUADRATURES,
            itertools.product(DIMENSIONS, DIMENSIONS),
            ["covariance", "sqrt"],
        )
    )
    def test_functional(self, seed, quadrature, dim_xy, mode):
        key = jax.random.key(seed)
        dim_x, dim_y = dim_xy
        key_m, key_P, key_A, key_b, key_Q = jax.random.split(key, 5)
        weights = get_quadrature(quadrature, dim_x)

        m = jax.random.normal(key_m, (dim_x,))
        P = jax.random.normal(key_P, (dim_x, 3 * dim_x))
        P = P @ P.T
        A = jax.random.normal(key_A, (dim_y, dim_x))
        b = jax.random.normal(key_b, (dim_y,))
        Q = jax.random.normal(key_Q, (dim_y, 3 * dim_y))
        Q = Q @ Q.T

        expected_R = Q

        if mode == "sqrt":
            Q = jnp.linalg.cholesky(Q)
            P = jnp.linalg.cholesky(P)

        linear_function_ = partial(linear_function, a=A, b=b)
        F, c, R = functional(linear_function_, Q, m, P, weights, mode)

        npt.assert_allclose(F, A, atol=1e-4)
        npt.assert_allclose(c, b, atol=1e-4)
        if mode == "covariance":
            npt.assert_allclose(R, expected_R, atol=1e-4)
        else:
            npt.assert_allclose(R @ R.T, expected_R, atol=1e-4)

    @parameterized.parameters(
        itertools.product(
            [0, 1, 2, 3, 4],
            QUADRATURES,
            itertools.product(DIMENSIONS, DIMENSIONS),
            ["covariance", "sqrt"],
        )
    )
    def test_conditional(self, seed, quadrature, dim_xy, mode):
        key = jax.random.key(seed)
        dim_x, dim_y = dim_xy
        key_m, key_P, key_A, key_b, key_Q = jax.random.split(key, 5)
        weights = get_quadrature(quadrature, dim_x)

        m = jax.random.normal(key_m, (dim_x,))
        P = jax.random.normal(key_P, (dim_x, 3 * dim_x))
        P = P @ P.T
        A = jax.random.normal(key_A, (dim_y, dim_x))
        b = jax.random.normal(key_b, (dim_y,))
        Q = jax.random.normal(key_Q, (dim_y, 3 * dim_y))
        Q = Q @ Q.T

        expected_R = Q

        if mode == "sqrt":
            Q = jnp.linalg.cholesky(Q)
            P = jnp.linalg.cholesky(P)

        conditional_mean = partial(linear_conditional_mean, a=A, b=b)
        if mode == "covariance":
            conditional_cov = partial(linear_conditional_cov, cov_q=Q)
        else:
            conditional_cov = partial(linear_conditional_chol, chol_q=Q)
        with jax.debug_nans():
            F, c, R = conditional_moments(
                conditional_mean, conditional_cov, m, P, weights, mode
            )

        npt.assert_allclose(F, A, atol=1e-4)
        npt.assert_allclose(c, b, atol=1e-4)
        if mode == "covariance":
            npt.assert_allclose(R, expected_R, atol=1e-4)
        else:
            npt.assert_allclose(R @ R.T, expected_R, atol=1e-4)
