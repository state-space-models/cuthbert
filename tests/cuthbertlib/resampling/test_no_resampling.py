import chex
import jax
import jax.numpy as jnp
import pytest

from cuthbertlib.resampling import no_resampling


def test_resampling_identity_outputs():
    n = 5
    key = jax.random.key(0)
    logits = jnp.linspace(-1.0, 1.0, n)
    positions = {
        "x": jnp.arange(n),
        "y": jnp.arange(2 * n).reshape(n, 2),
    }

    idx, logits_out, positions_out = no_resampling.resampling(
        key=key, logits=logits, positions=positions, n=n
    )

    chex.assert_trees_all_equal(idx, jnp.arange(n))
    chex.assert_trees_all_close(logits_out, logits, rtol=0.0, atol=0.0)
    chex.assert_trees_all_close(positions_out, positions, rtol=0.0, atol=0.0)


def test_conditional_resampling_identity_with_matching_pivot():
    n = 6
    key = jax.random.key(1)
    logits = jnp.linspace(-2.0, 2.0, n)
    positions = jnp.arange(n)

    idx, logits_out, positions_out = no_resampling.conditional_resampling(
        key=key,
        logits=logits,
        positions=positions,
        n=n,
        pivot_in=3,
        pivot_out=3,
    )

    chex.assert_trees_all_equal(idx, jnp.arange(n))
    chex.assert_trees_all_close(logits_out, logits, rtol=0.0, atol=0.0)
    chex.assert_trees_all_close(positions_out, positions, rtol=0.0, atol=0.0)


def test_conditional_resampling_enforces_pivot_mapping():
    n = 4
    key = jax.random.key(2)
    logits = jnp.zeros(n)
    positions = jnp.arange(n)

    idx, logits_out, positions_out = no_resampling.conditional_resampling(
        key=key,
        logits=logits,
        positions=positions,
        n=n,
        pivot_in=0,
        pivot_out=1,
    )
    chex.assert_trees_all_equal(idx[0], jnp.array(1))
    chex.assert_trees_all_close(logits_out, logits, rtol=0.0, atol=0.0)
    chex.assert_trees_all_close(positions_out, positions[idx], rtol=0.0, atol=0.0)


def test_conditional_resampling_raises_on_n_mismatch():
    n = 5
    key = jax.random.key(3)
    logits = jnp.zeros(n)
    positions = jnp.arange(n)

    with pytest.raises(AssertionError):
        no_resampling.conditional_resampling(
            key=key,
            logits=logits,
            positions=positions,
            n=n + 1,
            pivot_in=0,
            pivot_out=0,
        )
