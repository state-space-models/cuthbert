import itertools

import chex
import jax
import jax.numpy as jnp
import pytest
from jax import random

from cuthbert import filter
from cuthbert.hmm.filter import build_filter


@pytest.fixture(scope="module", autouse=True)
def config():
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", False)


def build_hmm(seed, num_states, num_time_steps):
    """Generate a random HMM.

    Args:
        seed: PRNG seed.
        num_states: Number of hidden states K.
        Nam_time_steps: Number of time steps T (excluding initial), so there are T+1 observations.

    Returns:
        init_prob: Initial state distribution of shape (K,).
        trans_matrices: Transition matrices of shape (T, K, K).
        log_emissions: Log emission likelihoods of shape (T+1, K).
    """
    key = random.key(seed)
    init_key, trans_key, emission_key = random.split(key, 3)

    # Initial distribution
    init_probs = random.uniform(init_key, (num_states,))
    init_probs /= init_probs.sum()

    # Transition matrices
    trans_matrices = random.uniform(trans_key, (num_time_steps, num_states, num_states))
    trans_matrices /= trans_matrices.sum(axis=-1, keepdims=True)

    # Emission likelihoods
    log_emissions = random.normal(emission_key, (num_time_steps + 1, num_states))
    return init_probs, trans_matrices, log_emissions


def sequential_filter(initial_probs, transition_matrix, log_likelihoods):
    """Filter for an HMM log space."""
    num_timesteps = transition_matrix.shape[0]

    log_initial = jnp.log(initial_probs)
    log_trans = jnp.log(transition_matrix)

    # Initialize (time 0 includes the first observation)
    log_alpha = []
    log_marginals = []

    log_alpha_t = log_initial + log_likelihoods[0]
    log_alpha.append(log_alpha_t)
    log_marginals.append(jax.nn.logsumexp(log_alpha_t))

    for t in range(num_timesteps):
        log_alpha_t = log_likelihoods[t + 1] + jax.nn.logsumexp(
            log_alpha[-1][:, None] + log_trans[t], axis=0
        )
        log_alpha.append(log_alpha_t)
        log_marginals.append(jax.nn.logsumexp(log_alpha_t))

    log_alpha = jnp.stack(log_alpha)
    log_marginals = jnp.array(log_marginals)

    alpha = jax.nn.softmax(log_alpha)
    return alpha, log_marginals


def build_filter_object(init_dist, trans_matrices, log_likelihoods):
    T = log_likelihoods.shape[0] - 1

    def get_init_dist(t):
        return init_dist

    def get_trans_matrix(t):
        return trans_matrices[t - 1]

    def get_obs_lls(t):
        return log_likelihoods[t]

    filt = build_filter(
        get_init_dist=get_init_dist,
        get_trans_matrix=get_trans_matrix,
        get_obs_lls=get_obs_lls,
    )
    model_inputs = jnp.arange(T + 1)
    return filt, model_inputs


seeds = [0, 123, 456]
num_states_list = [1, 3]
num_time_steps_list = [1, 25]
common_params = list(itertools.product(seeds, num_states_list, num_time_steps_list))


@pytest.mark.parametrize("seed,num_states,num_time_steps", common_params)
def test_discrete_filter(seed, num_states, num_time_steps):
    init_dist, trans_matrices, log_likelihoods = build_hmm(
        seed, num_states, num_time_steps
    )
    filt_obj, model_inputs = build_filter_object(
        init_dist, trans_matrices, log_likelihoods
    )

    # Run the filter
    states = filter(filt_obj, model_inputs, parallel=True)
    filt_dists, filt_loglik = states.filtered_state, states.log_marginal_ll

    # Reference solution
    des_dists, des_log_marginals = sequential_filter(
        init_dist, trans_matrices, log_likelihoods
    )

    # Check shapes
    assert filt_dists.shape == (num_time_steps + 1, num_states)
    assert filt_loglik.shape[0] == num_time_steps + 1

    # Check filtered distributions and marginal log likelihood
    chex.assert_trees_all_close(filt_dists, des_dists, rtol=1e-10, atol=1e-12)
    chex.assert_trees_all_close(filt_loglik, des_log_marginals, rtol=1e-10, atol=1e-12)
