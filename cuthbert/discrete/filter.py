"""
Parallel filter and smoother for discrete hidden Markov models~(HMMs).

Reference:
    https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9512397
"""

from functools import partial
from typing import NamedTuple, Protocol

import jax.numpy as jnp

from cuthbert.inference import Filter
from cuthbertlib.types import Array, ArrayTreeLike, KeyArray


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
        """Get the observation matrix."""
        ...


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


class FilterState(NamedTuple):
    f: Array
    log_g: Array

    @property
    def filtered_state(self) -> Array:
        return jnp.take(self.f, 0, axis=-2)

    @property
    def log_marginal_ll(self) -> Array:
        return jnp.take(self.log_g, 0, axis=-1)


def condition(state_probs: Array, log_liks: Array) -> tuple[Array, Array]:
    """Condition a state on an observation.

    Args:
        state_probs: Can either be the state transition probabilities or the
            initial distribution.
        log_liks: Vector of log p(y_t | x_t) for each possible state x_t.

    Returns:
        The conditioned state and the log normalizing constant.
    """
    ll_max = log_liks.max(axis=-1)
    A_cond = state_probs * jnp.exp(log_liks - ll_max)
    norm = A_cond.sum(axis=-1)
    A_cond /= jnp.expand_dims(norm, axis=-1)
    return A_cond, jnp.log(norm) + ll_max


def init_prepare(
    model_inputs: ArrayTreeLike,
    get_init_dist: GetInitDist,
    get_obs_lls: GetObservationLikelihoods,
    key: KeyArray | None = None,
) -> FilterState:
    init_dist = get_init_dist(model_inputs)
    obs_log_probs = get_obs_lls(model_inputs)
    f, log_g = condition(init_dist, obs_log_probs)
    K = init_dist.shape[0]
    f *= jnp.ones((K, K))
    log_g *= jnp.ones(K)
    return FilterState(f, log_g)


def filter_prepare(
    model_inputs: ArrayTreeLike,
    get_trans_matrix: GetTransitionMatrix,
    get_obs_lls: GetObservationLikelihoods,
    key: KeyArray | None = None,
) -> FilterState:
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
    f, log_g = condition(trans_matrix, obs_log_probs)
    return FilterState(f, log_g)


def filter_combine(state_1: FilterState, state_2: FilterState) -> FilterState:
    """
    Combine filter state from previous time point with state prepared
    with latest model inputs.

    Args:
        state_1: State from previous time step.
        state_2: State prepared with latest model inputs.

    Returns:
        Combined filter state.
    """
    f, lognorm = condition(state_1.f, state_2.log_g)
    f = f @ state_2.f
    log_g = state_1.log_g + lognorm
    return FilterState(f, log_g)
