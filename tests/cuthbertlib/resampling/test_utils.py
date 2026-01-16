import itertools

import chex
import jax
import jax.numpy as jnp
import numpy.testing as npt
import pytest
from absl.testing import parameterized
from jax.scipy.special import logsumexp

from cuthbertlib.resampling.utils import (
    _inverse_cdf_cpu,
    _inverse_cdf_default,
    inverse_cdf,
)


class TestInverseCdf(chex.TestCase):
    def setUp(self):
        super().setUp()
        self.Ms = [10, 100]
        self.Ns = [10, 100]

    @chex.all_variants(with_pmap=False, without_jit=False)
    @pytest.mark.xdist_group(name="inverse_cdf")  # Serialize to avoid OOM
    @parameterized.parameters([0, 1, 2, 3, 4])
    def test_inverse_cdf(self, seed):
        key = jax.random.key(seed)
        for M, N in itertools.product(self.Ms, self.Ns):
            sorted_uniforms = jax.random.uniform(key, (N,))
            sorted_uniforms = jnp.sort(sorted_uniforms)
            log_weights = jnp.linspace(-3.0, 0.0, M)

            indices = self.variant(inverse_cdf)(sorted_uniforms, log_weights)

            # Test for general properties
            self.assertEqual(indices.shape, sorted_uniforms.shape)
            npt.assert_array_less(indices, M)
            npt.assert_array_less(-1, indices)

            # Test for output values
            weights = jnp.exp(log_weights - logsumexp(log_weights))
            cdf = jnp.cumsum(weights)
            expected = jnp.searchsorted(cdf, sorted_uniforms)

            npt.assert_allclose(indices, expected)

    @chex.all_variants(with_pmap=False)
    @parameterized.parameters([0, 1, 2, 3, 4])
    @pytest.mark.xdist_group(name="inverse_cdf")  # Serialize to avoid OOM
    def test_cpu_default_match(self, seed):
        key = jax.random.key(seed)
        for M, N in itertools.product(self.Ms, self.Ns):
            sorted_uniforms = jax.random.uniform(key, (N,))
            sorted_uniforms = jnp.sort(sorted_uniforms)
            log_weights = jnp.linspace(-3.0, 0.0, M)

            indices = self.variant(_inverse_cdf_cpu)(sorted_uniforms, log_weights)
            indices_default = self.variant(_inverse_cdf_default)(
                sorted_uniforms, log_weights
            )

            npt.assert_allclose(indices, indices_default)
