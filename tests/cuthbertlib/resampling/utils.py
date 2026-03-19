import chex
import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.special import logsumexp


def resampling_tester(rng_key, log_weights, resampling, m, k):
    keys = jax.random.split(rng_key, k)

    # create dummy positions matching number of particles
    n_particles = log_weights.shape[0]
    positions = jax.random.normal(jax.random.key(0), (n_particles,))

    def call_one(key):
        idx, logits_out, _ = resampling(key, log_weights, positions)
        return idx

    indices = jax.vmap(call_one)(keys)
    _check_bincounts(indices, log_weights, m, k)


def conditional_resampling_tester(rng_key, log_weights, conditional_resampling, m, k):
    def do_one(key):
        # check that Bayes rule is satisfied:
        # we sample the index of the pivot particle unconditionally,
        # and then the rest conditionally and then return the indices, which
        # should be the same as the unconditional resampling

        key_i, key_resampling, key_conditional = jax.random.split(key, 3)
        pivot_in = jax.random.randint(key_i, (), 0, m)

        p = jnp.exp(log_weights - logsumexp(log_weights))
        pivot_out = jax.random.choice(key_resampling, m, shape=(), p=p)

        # create dummy positions
        positions = jax.random.normal(jax.random.key(0), (m,))

        conditional_indices, _, _ = conditional_resampling(
            key_conditional, log_weights, positions, pivot_in, pivot_out
        )
        return conditional_indices, pivot_out, conditional_indices[pivot_in]

    keys = jax.random.split(rng_key, k)
    indices, expected_pivots_out, pivots_out = jax.vmap(do_one)(keys)

    chex.assert_trees_all_equal(expected_pivots_out, pivots_out)
    _check_bincounts(indices, log_weights, m, k)


def _check_bincounts(indices, log_weights, m, k):
    n = log_weights.shape[0]
    counts = jax.vmap(lambda z: jnp.bincount(z, length=n))(indices)
    probs = jnp.mean(counts, axis=0) / m
    expected_probs = jnp.exp(log_weights - logsumexp(log_weights))
    tol = 1 / np.sqrt(k)
    chex.assert_trees_all_close(probs, expected_probs, rtol=tol, atol=tol)
