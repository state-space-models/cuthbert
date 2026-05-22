import itertools
from functools import partial

import chex
import jax
import jax.numpy as jnp
import pytest
from jax import random
from jax.nn import logsumexp

from cuthbert import factorial
from cuthbert.discrete.filter import DiscreteFilterState, filter_combine, filter_prepare
from cuthbert.inference import Filter
from cuthbertlib.discrete.filtering import FilterScanElement


@pytest.fixture(scope="module", autouse=True)
def config():
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", False)


def _kron_vectors(vecs):
    out = vecs[0]
    for i in range(1, vecs.shape[0]):
        out = jnp.kron(out, vecs[i])
    return out


def _kron_matrices(mats):
    out = mats[0]
    for i in range(1, mats.shape[0]):
        out = jnp.kron(out, mats[i])
    return out


def generate_factorial_hmm(
    seed: int,
    num_states: int,
    num_factors: int,
    num_factors_local: int,
    num_time_steps: int,
):
    key = random.key(seed)
    init_key, trans_key, obs_key, inds_key = random.split(key, 4)

    init_dist = random.uniform(init_key, (num_factors, num_states))
    init_dist /= init_dist.sum(axis=-1, keepdims=True)

    trans_matrices = random.uniform(
        trans_key, (num_time_steps, num_factors, num_states, num_states)
    )
    trans_matrices /= trans_matrices.sum(axis=-1, keepdims=True)

    local_num_states = num_states**num_factors_local
    local_obs_lls = random.normal(obs_key, (num_time_steps, local_num_states))

    factorial_indices = jax.vmap(
        lambda k: random.choice(
            k, jnp.arange(num_factors), (num_factors_local,), replace=False
        )
    )(random.split(inds_key, num_time_steps))

    return init_dist, trans_matrices, local_obs_lls, factorial_indices


def build_factorial_discrete_filter(model_params):
    init_dist, trans_matrices, local_obs_lls, factorial_indices = model_params
    num_states = init_dist.shape[-1]

    def get_init_state(model_inputs, key=None):
        model_inputs = jnp.asarray(model_inputs)
        f = jnp.tile(init_dist[:, None, :], (1, num_states, 1))
        log_g = jnp.zeros((init_dist.shape[0], num_states))
        return DiscreteFilterState(
            elem=FilterScanElement(f=f, log_g=log_g),
            model_inputs=model_inputs,
        )

    def get_local_trans_matrix(model_inputs):
        inds = factorial_indices[model_inputs - 1]
        local_trans = trans_matrices[model_inputs - 1, inds]
        return _kron_matrices(local_trans)

    def get_local_obs_lls(model_inputs):
        return local_obs_lls[model_inputs - 1]

    filter_obj = Filter(
        init_prepare=get_init_state,
        filter_prepare=partial(
            filter_prepare,
            get_trans_matrix=get_local_trans_matrix,
            get_obs_lls=get_local_obs_lls,
        ),
        filter_combine=filter_combine,
        associative=False,
    )
    factorializer = factorial.discrete.build_factorializer(
        lambda model_inputs: factorial_indices[model_inputs - 1]
    )
    model_inputs = jnp.arange(local_obs_lls.shape[0] + 1)
    return filter_obj, factorializer, model_inputs


def reference_factorial_filter(model_params):
    init_dist, trans_matrices, local_obs_lls, factorial_indices = model_params
    num_time_steps = local_obs_lls.shape[0]
    num_factors_local = factorial_indices.shape[1]

    factorial_dist = init_dist
    ell = jnp.array(0.0)
    local_factorial_dists = []
    local_ells = []
    factorial_dists_all = [factorial_dist]

    for t in range(num_time_steps):
        inds = factorial_indices[t]
        joint_trans = _kron_matrices(trans_matrices[t, inds])
        joint_prior = _kron_vectors(factorial_dist[inds])

        joint_predict = joint_prior @ joint_trans
        joint_log_post = jnp.log(joint_predict) + local_obs_lls[t]
        ell = ell + logsumexp(joint_log_post)
        joint_post = jnp.exp(joint_log_post - logsumexp(joint_log_post))

        post_tensor = joint_post.reshape(
            (factorial_dist.shape[-1],) * num_factors_local
        )
        local_marginals = jnp.stack(
            [
                post_tensor.sum(
                    axis=tuple(ax for ax in range(num_factors_local) if ax != i)
                )
                for i in range(num_factors_local)
            ]
        )
        factorial_dist = factorial_dist.at[inds].set(local_marginals)

        local_factorial_dists.append(local_marginals)
        local_ells.append(ell)
        factorial_dists_all.append(factorial_dist)

    return (
        jnp.stack(local_factorial_dists),
        jnp.stack(local_ells),
        jnp.stack(factorial_dists_all),
    )


params = list(
    itertools.product(
        [1, 43],
        [3],  # num_states
        [8],  # num_factors
        [2],  # num_factors_local
        [1, 20],  # num_time_steps
    )
)


@pytest.mark.parametrize(
    "seed,num_states,num_factors,num_factors_local,num_time_steps",
    params,
)
def test_factorial_discrete_filter(
    seed, num_states, num_factors, num_factors_local, num_time_steps
):
    model_params = generate_factorial_hmm(
        seed, num_states, num_factors, num_factors_local, num_time_steps
    )
    filter_obj, factorializer, model_inputs = build_factorial_discrete_filter(
        model_params
    )
    local_dists_ref, local_ells_ref, factorial_dists_ref = reference_factorial_filter(
        model_params
    )

    # output_factorial=False
    init_state, local_filter_states = factorial.filter(
        filter_obj, factorializer, model_inputs, output_factorial=False
    )
    chex.assert_trees_all_close(init_state.dist, model_params[0], rtol=1e-10, atol=0.0)
    chex.assert_trees_all_close(
        local_filter_states.dist, local_dists_ref, rtol=1e-10, atol=0.0
    )
    # log normalizing constants are carried per factor but should be identical.
    chex.assert_trees_all_close(
        local_filter_states.log_normalizing_constant[:, 0],
        local_ells_ref,
        rtol=1e-10,
        atol=0.0,
    )

    # output_factorial=True
    factorial_states = factorial.filter(
        filter_obj, factorializer, model_inputs, output_factorial=True
    )
    chex.assert_trees_all_close(
        factorial_states.dist, factorial_dists_ref, rtol=1e-10, atol=0.0
    )
    chex.assert_trees_all_close(
        factorial_states.log_normalizing_constant[1:, 0],
        local_ells_ref,
        rtol=1e-10,
        atol=0.0,
    )
