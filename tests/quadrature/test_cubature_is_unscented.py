import itertools

import chex
import numpy as np
from absl.testing import parameterized

from quadrature import cubature, unscented

DIMENSIONS = [1, 2, 3]
SEEDS = [0, 1, 2, 3, 4]


class TestCubatureIsUnscented(chex.TestCase):
    def setUp(self):
        super().setUp()

    @parameterized.parameters(itertools.product(DIMENSIONS, SEEDS))
    def test_cubature_is_unscented(self, dim, seed):
        np.random.seed(seed)
        cubature_quadrature = cubature.weights(dim)
        unscented_quadrature = unscented.weights(dim, 1.0, 0.0, 0.0)

        m = np.random.randn(dim)
        chol = np.random.rand(dim, dim)
        chol = np.tril(chol)

        cubature_sigma_points = cubature_quadrature.get_sigma_points(m, chol)
        unscented_sigma_points = unscented_quadrature.get_sigma_points(m, chol)

        chex.assert_trees_all_close(
            cubature_sigma_points.points,
            unscented_sigma_points.points[1:],
            rtol=1e-5,
            atol=1e-5,
        )
