"""Index selection methods for MCMC."""

import jax
import jax.numpy as jnp

from cuthbertlib.types import Array, KeyArray, ScalarArray, ScalarArrayLike


def barker_move(key: KeyArray, weights: Array, pivot: ScalarArrayLike) -> tuple[ScalarArray, ScalarArray]:
    """
    A Barker proposal move for a categorical distribution.

    Args:
        key: JAX PRNG key.
        weights: Normalized weights of the categorical distribution.
        pivot: The current index to move from.

    Returns:
        A tuple containing the new index and the probability of a new index being selected
    """
    M = weights.shape[0]
    i = jax.random.choice(key, M, p=weights, shape=())
    return i, 1 - weights[pivot]

def force_move(key: KeyArray, weights: Array, pivot: ScalarArrayLike) -> tuple[ScalarArray, ScalarArray]:
    """A forced-move proposal for a categorical distribution.

    The weights are assumed to be normalised (linear, not log).

    Args:
        key: JAX PRNG key.
        weights: Normalized weights of the categorical distribution.
        pivot: The current index to move from.

    Returns:
        A tuple containing the new index and the overall acceptance probability.
    """
    n_particles = weights.shape[0]
    key_1, key_2 = jax.random.split(key, 2)

    p_pivot = weights[pivot]
    one_minus_p_pivot = 1 - p_pivot

    # Create proposal distribution q(i) = w_i / (1 - w_k) for i != k
    proposal_weights = weights.at[pivot].set(0)
    proposal_weights = proposal_weights / one_minus_p_pivot

    proposal_idx = jax.random.choice(key_1, n_particles, p=proposal_weights, shape=())

    # Acceptance step to make the move valid: u < (1 - w_k) / (1 - w_i)
    u = jax.random.uniform(key_2, shape=())
    accept = u * (1 - weights[proposal_idx]) < one_minus_p_pivot

    new_idx = jax.lax.select(accept, proposal_idx, pivot)

    # The acceptance probability alpha is sum_i q(i) * min(1, (1-w_k)/(1-w_i))
    alpha = jnp.nansum(one_minus_p_pivot * proposal_weights / (1 - weights))
    alpha = jnp.clip(alpha, 0, 1.0)

    return new_idx, alpha
