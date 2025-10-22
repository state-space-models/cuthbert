"""
Parallel-in-time Bayesian filter for discrete hidden Markov models.

References:
    - https://ieeexplore.ieee.org/document/9512397
    - https://github.com/EEA-sensors/sequential-parallelization-examples/tree/main/python/temporal-parallelization-inference-in-HMMs
    - https://github.com/probml/dynamax/blob/main/dynamax/hidden_markov_model/parallel_inference.py
"""

from functools import partial
from typing import NamedTuple

import jax.numpy as jnp
from jax import tree

from cuthbert.discrete.types import (
    GetInitDist,
    GetObsLogLikelihoods,
    GetTransitionMatrix,
)
from cuthbert.inference import Filter
from cuthbertlib.discrete import filtering
from cuthbertlib.types import Array, ArrayTree, ArrayTreeLike, KeyArray


class DiscreteFilterState(NamedTuple):
    elem: filtering.FilterScanElement
    model_inputs: ArrayTree

    @property
    def dist(self) -> Array:
        return jnp.take(self.elem.f, 0, axis=-2)

    @property
    def log_marginal(self) -> Array:
        return jnp.take(self.elem.log_g, 0, axis=-1)


def build_filter(
    get_init_dist: GetInitDist,
    get_trans_matrix: GetTransitionMatrix,
    get_obs_lls: GetObsLogLikelihoods,
) -> Filter:
    r"""Builds a filter object for discrete hidden Markov models.

    Args:
        get_init_dist: Function to get initial state probabilities $m_i = p(x_0 = i)$.
        get_trans_matrix: Function to get the transition matrix $A_{ij} = p(x_t = j \mid x_{t-1} = i)$.
        get_obs_lls: Function to get observation log likelihoods $b_i = \log p(y_t | x_t = i)$.

    Returns:
        Filter object. Suitable for associative scan.
    """
    return Filter(
        init_prepare=partial(
            init_prepare, get_init_dist=get_init_dist, get_obs_lls=get_obs_lls
        ),
        filter_prepare=partial(
            filter_prepare, get_trans_matrix=get_trans_matrix, get_obs_lls=get_obs_lls
        ),
        filter_combine=filter_combine,
        associative=True,
    )


def init_prepare(
    model_inputs: ArrayTreeLike,
    get_init_dist: GetInitDist,
    get_obs_lls: GetObsLogLikelihoods,
    key: KeyArray | None = None,
) -> DiscreteFilterState:
    """Prepare the initial state for the filter.

    Args:
        model_inputs: Model inputs.
        get_init_dist: Function to get initial state probabilities m_i = p(x_0 = i).
        get_obs_lls: Function to get observation log likelihoods b_i = log p(y_t | x_t = i).
        key: JAX random key - not used.

    Returns:
        Prepared state for the filter.
    """
    model_inputs = tree.map(lambda x: jnp.asarray(x), model_inputs)
    init_dist = get_init_dist(model_inputs)
    obs_lls = get_obs_lls(model_inputs)
    f, log_g = filtering.condition_on_obs(init_dist, obs_lls)
    N = init_dist.shape[0]
    f *= jnp.ones((N, N))
    log_g *= jnp.ones(N)
    return DiscreteFilterState(
        elem=filtering.FilterScanElement(f, log_g), model_inputs=model_inputs
    )


def filter_prepare(
    model_inputs: ArrayTreeLike,
    get_trans_matrix: GetTransitionMatrix,
    get_obs_lls: GetObsLogLikelihoods,
    key: KeyArray | None = None,
) -> DiscreteFilterState:
    """Prepare a state for a filter step.

    Args:
        model_inputs: Model inputs.
        get_trans_matrix: Function to get the transition matrix A_{ij} = p(x_t = j | x_{t-1} = i).
        get_obs_lls: Function to get observation log likelihoods b_i = log p(y_t | x_t = i).
        key: JAX random key - not used.

    Returns:
        Prepared state for the filter.
    """
    model_inputs = tree.map(lambda x: jnp.asarray(x), model_inputs)
    trans_matrix = get_trans_matrix(model_inputs)
    obs_lls = get_obs_lls(model_inputs)
    f, log_g = filtering.condition_on_obs(trans_matrix, obs_lls)
    return DiscreteFilterState(
        elem=filtering.FilterScanElement(f, log_g), model_inputs=model_inputs
    )


def filter_combine(
    state_1: DiscreteFilterState, state_2: DiscreteFilterState
) -> DiscreteFilterState:
    """Combine the filter state from the previous time point with the state
    prepared with the latest model inputs.

    Args:
        state_1: State from the previous time step.
        state_2: State prepared with the latest model inputs.

    Returns:
        Combined filter state.
    """
    combined_elem = filtering.filtering_operator(state_1.elem, state_2.elem)
    return DiscreteFilterState(elem=combined_elem, model_inputs=state_2.model_inputs)
