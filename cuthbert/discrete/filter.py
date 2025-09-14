"""
Parallel-in-time Bayesian filter for discrete hidden Markov models.

References:
    https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9512397
    https://github.com/EEA-sensors/sequential-parallelization-examples/tree/main/python/temporal-parallelization-inference-in-HMMs
"""

from functools import partial
from typing import NamedTuple, Protocol

import jax.numpy as jnp

from cuthbert.inference import Filter
from cuthbertlib.discrete import filtering
from cuthbertlib.types import Array, ArrayTree, ArrayTreeLike, KeyArray


class GetInitDist(Protocol):
    def __call__(self, model_inputs: ArrayTreeLike) -> Array:
        """Get the initial distribution.

        Should return an array of shape (N,) where N is the number of states.
        """
        ...


class GetTransitionMatrix(Protocol):
    def __call__(self, model_inputs: ArrayTreeLike) -> Array:
        """Get the transition matrix.

        Should return an array A of shape (N, N) where N is the number of
        states, with A_{ij} = p(x_t = j | x_{t-1} = i).
        """
        ...


class GetObsLogLikelihoods(Protocol):
    def __call__(self, model_inputs: ArrayTreeLike) -> Array:
        """Get the observation log likelihoods.

        Should return an array b of shape (N,) where N is the number of states,
        with b_i = log p(y_t | x_t = i).
        """
        ...


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
    """Builds a filter object for discrete hidden Markov models."""
    return Filter(
        init_prepare=partial(
            init_prepare,
            get_init_dist=get_init_dist,
            get_obs_lls=get_obs_lls,
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
    """
    Prepare a state for a filter step.

    Arcs:
        model_inputs: Model inputs.
        key: JAX random key - not used.

    Returns:
        Prepared state for the filter.
    """
    trans_matrix = get_trans_matrix(model_inputs)
    obs_lls = get_obs_lls(model_inputs)
    f, log_g = filtering.condition_on_obs(trans_matrix, obs_lls)
    return DiscreteFilterState(
        elem=filtering.FilterScanElement(f, log_g), model_inputs=model_inputs
    )


def filter_combine(
    state_1: DiscreteFilterState, state_2: DiscreteFilterState
) -> DiscreteFilterState:
    """
    Combine filter state from previous time point with state prepared
    with latest model inputs.

    Args:
        state_1: State from previous time step.
        state_2: State prepared with latest model inputs.

    Returns:
        Combined filter state.
    """
    combined_elem = filtering.filtering_operator(state_1.elem, state_2.elem)
    return DiscreteFilterState(elem=combined_elem, model_inputs=state_2.model_inputs)
