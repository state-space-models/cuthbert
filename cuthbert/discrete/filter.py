"""
Parallel filter and smoother for discrete hidden Markov models (HMMs).

Reference:
    https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9512397
"""

from functools import partial
from typing import NamedTuple, Protocol

import jax.numpy as jnp

from cuthbert.inference import Filter
from cuthbertlib.discrete import filtering
from cuthbertlib.types import Array, ArrayTree, ArrayTreeLike, KeyArray


class GetInitDist(Protocol):
    def __call__(self, model_inputs: ArrayTreeLike) -> Array:
        """Get the initial distribution.."""
        ...


class GetTransitionMatrix(Protocol):
    def __call__(self, model_inputs: ArrayTreeLike) -> Array:
        """Get the transition matrix."""
        ...


class GetObservationLikelihoods(Protocol):
    def __call__(self, model_inputs: ArrayTreeLike) -> Array:
        """Get the observation log likelihoods."""
        ...


class HMMFilterState(NamedTuple):
    elem: filtering.FilterScanElement
    model_inputs: ArrayTree

    @property
    def filtered_state(self) -> Array:
        return jnp.take(self.elem.f, 0, axis=-2)

    @property
    def log_marginal_ll(self) -> Array:
        return jnp.take(self.elem.log_g, 0, axis=-1)


def build_filter(
    get_init_dist: GetInitDist,
    get_trans_matrix: GetTransitionMatrix,
    get_obs_lls: GetObservationLikelihoods,
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
    get_obs_lls: GetObservationLikelihoods,
    key: KeyArray | None = None,
) -> HMMFilterState:
    init_dist = get_init_dist(model_inputs)
    obs_log_probs = get_obs_lls(model_inputs)
    f, log_g = filtering.condition_on_obs(init_dist, obs_log_probs)
    K = init_dist.shape[0]
    f *= jnp.ones((K, K))
    log_g *= jnp.ones(K)
    return HMMFilterState(
        elem=filtering.FilterScanElement(f, log_g), model_inputs=model_inputs
    )


def filter_prepare(
    model_inputs: ArrayTreeLike,
    get_trans_matrix: GetTransitionMatrix,
    get_obs_lls: GetObservationLikelihoods,
    key: KeyArray | None = None,
) -> HMMFilterState:
    """
    Prepare a state for a filter step.

    Arcs:
        model_inputs: Model inputs.
        key: JAX random key - not used.

    Returns:
        Prepared state for the filter.
    """
    trans_matrix = get_trans_matrix(model_inputs)
    obs_log_probs = get_obs_lls(model_inputs)
    f, log_g = filtering.condition_on_obs(trans_matrix, obs_log_probs)
    return HMMFilterState(
        elem=filtering.FilterScanElement(f, log_g), model_inputs=model_inputs
    )


def filter_combine(state_1: HMMFilterState, state_2: HMMFilterState) -> HMMFilterState:
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
    return HMMFilterState(elem=combined_elem, model_inputs=state_2.model_inputs)
