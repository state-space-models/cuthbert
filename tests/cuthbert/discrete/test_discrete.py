import chex
import jax
import jax.numpy as jnp
import pytest
from absl.testing import parameterized
from jax import random

from cuthbert import filter, smoother
from cuthbert.discrete import build_filter, build_smoother


@pytest.fixture(scope="module", autouse=True)
def config():
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", False)


def build_hmm(seed, num_states, num_time_steps):
    """Generate a random HMM.

    Args:
        seed: PRNG seed.
        num_states: Number of hidden states N.
        Nam_time_steps: Number of time steps T (excluding initial), so there are T+1 observations.

    Returns:
        init_dist: Initial state distribution of shape (N,).
        trans_matrices: Transition matrices of shape (T, N, N).
        log_likelihoods: Observation log likelihoods of shape (T + 1, N).
    """
    key = random.key(seed)
    init_key, trans_key, obs_key = random.split(key, 3)

    # Initial distribution
    init_dist = random.uniform(init_key, (num_states,))
    init_dist /= init_dist.sum()

    # Transition matrices
    trans_matrices = random.uniform(trans_key, (num_time_steps, num_states, num_states))
    trans_matrices /= trans_matrices.sum(axis=-1, keepdims=True)

    # Observation log likelihoods
    log_likelihoods = random.normal(obs_key, (num_time_steps + 1, num_states))
    return init_dist, trans_matrices, log_likelihoods


def std_forward_backward(init_dist, trans_matrix, log_likelihoods):
    """The standard forward-backward algorithm for discrete HMMs.

    Computes everything in the log space.

    References:
        https://en.wikipedia.org/wiki/Forward%E2%80%93backward_algorithm
        https://ieeexplore.ieee.org/document/9512397
    """
    num_timesteps, N = trans_matrix.shape[:2]

    log_initial = jnp.log(init_dist)
    log_trans = jnp.log(trans_matrix)

    # Forward filter
    log_alpha = []
    log_marginals = []

    log_alpha_init = log_initial + log_likelihoods[0]
    log_alpha.append(log_alpha_init)
    log_marginals.append(jax.nn.logsumexp(log_alpha_init))

    for t in range(num_timesteps):
        log_alpha_t = log_likelihoods[t + 1] + jax.nn.logsumexp(
            log_alpha[-1][:, None] + log_trans[t], axis=0
        )
        log_alpha.append(log_alpha_t)
        log_marginals.append(jax.nn.logsumexp(log_alpha_t))

    log_alpha = jnp.stack(log_alpha)
    log_marginals = jnp.array(log_marginals)

    # Backward filter
    log_beta = []  # beta_t = p(y_{t+1:T} | x_t)
    log_beta_final = jnp.zeros(N)
    log_beta.append(log_beta_final)

    for t in range(num_timesteps - 1, -1, -1):
        log_beta_t = jax.nn.logsumexp(
            log_trans[t] + log_likelihoods[t + 1] + log_beta[-1], axis=1
        )
        log_beta.append(log_beta_t)

    log_beta = jnp.stack(log_beta[::-1])
    log_gamma_unnorm = log_alpha + log_beta

    filt_states = jax.nn.softmax(log_alpha)
    smooth_states = jax.nn.softmax(log_gamma_unnorm)

    return filt_states, smooth_states, log_marginals


def build_inference_object(init_dist, trans_matrices, log_likelihoods):
    T = log_likelihoods.shape[0] - 1

    def get_init_dist(model_inputs):
        return init_dist

    def get_trans_matrix(model_inputs):
        return trans_matrices[model_inputs - 1]

    def get_obs_lls(model_inputs):
        return log_likelihoods[model_inputs]

    filter_obj = build_filter(
        get_init_dist=get_init_dist,
        get_trans_matrix=get_trans_matrix,
        get_obs_lls=get_obs_lls,
    )
    smoother_obj = build_smoother(get_trans_matrix)
    model_inputs = jnp.arange(T + 1)
    return filter_obj, smoother_obj, model_inputs


class TestDiscrete(chex.TestCase):
    @chex.variants(with_jit=True, without_jit=True)
    @parameterized.product(
        seed=[1, 123, 456], num_states=[5], num_time_steps=[25], parallel=[True, False]
    )
    def test(self, seed, num_states, num_time_steps, parallel):
        init_dist, trans_matrices, log_likelihoods = build_hmm(
            seed, num_states, num_time_steps
        )
        filter_obj, smoother_obj, model_inputs = build_inference_object(
            init_dist, trans_matrices, log_likelihoods
        )

        # Run the filter and smoother
        filtered_states = self.variant(
            filter, static_argnames=("filter_obj", "parallel")
        )(filter_obj, model_inputs, parallel=parallel)
        filt_dists, log_normalizing_constants = (
            filtered_states.dist,
            filtered_states.log_normalizing_constant,
        )

        smoothed_states = self.variant(
            smoother, static_argnames=("smoother_obj", "parallel")
        )(smoother_obj, filtered_states, None, parallel=parallel)
        smooth_dists = smoothed_states.dist

        # Reference solution
        des_filt_dists, des_smooth_dists, des_log_marginals = std_forward_backward(
            init_dist, trans_matrices, log_likelihoods
        )

        # Check shapes
        assert filt_dists.shape == (num_time_steps + 1, num_states)
        assert log_normalizing_constants.shape == (num_time_steps + 1,)
        assert smooth_dists.shape == (num_time_steps + 1, num_states)

        # Check filtered and smoothed distributions and log marginal likelihoods
        chex.assert_trees_all_close(
            (filt_dists, log_normalizing_constants, smooth_dists),
            (des_filt_dists, des_log_marginals, des_smooth_dists),
            rtol=1e-10,
            atol=0.0,
        )
