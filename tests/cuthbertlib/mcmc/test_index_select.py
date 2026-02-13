import itertools

import chex
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from absl.testing import parameterized

from cuthbertlib.mcmc.index_select import barker_move, force_move
from cuthbertlib.resampling.utils import normalize

N_PARTICLES = [5, 10]
test_cases = list(itertools.product([0, 42], N_PARTICLES, [barker_move, force_move]))


def _check_dist(indices, expected_probs, n_particles):
    """Checks that the empirical distribution of indices matches the expected one."""
    n_chains, n_iter = indices.shape
    counts = jnp.bincount(jnp.ravel(indices), length=n_particles)
    probs = counts / (n_iter * n_chains)
    tol = 1 / np.sqrt(n_chains)
    chex.assert_trees_all_close(probs, expected_probs, rtol=tol, atol=tol)


class TestIndexSelect(chex.TestCase):
    def setUp(self):
        super().setUp()
        self.K = 10_000  # Number of samples for statistical tests

    @chex.all_variants(with_pmap=False, without_jit=False)
    @pytest.mark.xdist_group(name="mcmc")
    @parameterized.parameters(test_cases)
    def test_mcmc(self, seed, n_particles, move):
        """Tests the functions."""
        key = jax.random.key(seed)
        key_weights, key_test = jax.random.split(key, 2)

        # Generate random weights
        log_weights = jax.random.uniform(key_weights, (n_particles,))
        weights = normalize(log_weights)

        # Set a fixed initial pivot
        pivot = 0

        def run_chain(key):
            def body(p, key_in):
                p, _ = move(key_in, weights, p)
                return p, p

            # Run the chain for a few steps to burn in
            _, pivot_final = jax.lax.scan(body, pivot, jax.random.split(key, self.K))
            return pivot_final

        keys = jax.random.split(key_test, 25)
        final_indices = self.variant(jax.vmap(run_chain))(keys)

        _check_dist(final_indices, weights, n_particles)
