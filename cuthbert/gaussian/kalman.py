"""Implements the square-root, parallel-in-time Kalman filter for linear Gaussian SSMs.

See [Yaghoobi et. al. (2025)](https://doi.org/10.1137/23M156121X).
"""

from functools import partial
from typing import NamedTuple

from jax import numpy as jnp
from jax import tree

from cuthbert.gaussian.types import (
    GetDynamicsParams,
    GetInitParams,
    GetObservationParams,
)
from cuthbert.inference import Filter, Smoother
from cuthbert.utils import dummy_tree_like
from cuthbertlib.kalman import filtering, smoothing
from cuthbertlib.types import Array, ArrayTree, ArrayTreeLike, KeyArray


class KalmanFilterState(NamedTuple):
    """Kalman filter state."""

    elem: filtering.FilterScanElement
    model_inputs: ArrayTree

    @property
    def mean(self) -> Array:
        """Filtering mean."""
        return self.elem.b

    @property
    def chol_cov(self) -> Array:
        """Filtering generalised Cholesky covariance."""
        return self.elem.U

    @property
    def log_normalizing_constant(self) -> Array:
        """Log normalizing constant (cumulative)."""
        return self.elem.ell


class KalmanSmootherState(NamedTuple):
    """Kalman smoother state."""

    elem: smoothing.SmootherScanElement
    model_inputs: ArrayTree
    gain: Array | None = None
    chol_cov_given_next: Array | None = None

    @property
    def mean(self) -> Array:
        """Smoothing mean."""
        return self.elem.g

    @property
    def chol_cov(self) -> Array:
        """Smoothing generalised Cholesky covariance."""
        return self.elem.D


def build_filter(
    get_init_params: GetInitParams,
    get_dynamics_params: GetDynamicsParams,
    get_observation_params: GetObservationParams,
) -> Filter:
    """Builds an exact Kalman filter object for linear Gaussian SSMs.

    Args:
        get_init_params: Function to get m0, chol_P0 to initialize filter state,
            given model inputs sufficient to define p(x_0) = N(m0, chol_P0 @ chol_P0^T).
        get_dynamics_params: Function to get dynamics parameters, F, c, chol_Q
            given model inputs sufficient to define
            p(x_t | x_{t-1}) = N(F @ x_{t-1} + c, chol_Q @ chol_Q^T).
        get_observation_params: Function to get observation parameters, H, d, chol_R, y
            given model inputs sufficient to define
            p(y_t | x_t) = N(H @ x_t + d, chol_R @ chol_R^T).

    Returns:
        Filter object for exact Kalman filter. Suitable for associative scan.
    """
    return Filter(
        init_prepare=partial(
            init_prepare,
            get_init_params=get_init_params,
        ),
        filter_prepare=partial(
            filter_prepare,
            get_dynamics_params=get_dynamics_params,
            get_observation_params=get_observation_params,
        ),
        filter_combine=filter_combine,
        associative=True,
    )


def build_smoother(
    get_dynamics_params: GetDynamicsParams,
    store_gain: bool = False,
    store_chol_cov_given_next: bool = False,
) -> Smoother:
    """Builds an exact Kalman smoother object for linear Gaussian SSMs.

    Args:
        get_dynamics_params: Function to get dynamics parameters, F, c, chol_Q
            given model inputs sufficient to define
            p(x_t | x_{t-1}) = N(F @ x_{t-1} + c, chol_Q @ chol_Q^T).
        store_gain: Whether to store the gain matrix in the smoother state.
        store_chol_cov_given_next: Whether to store the chol_cov_given_next matrix
            in the smoother state.

    Returns:
        Smoother object for exact Kalman smoother. Suitable for associative scan.
    """
    return Smoother(
        convert_filter_to_smoother_state=partial(
            convert_filter_to_smoother_state,
            store_gain=store_gain,
            store_chol_cov_given_next=store_chol_cov_given_next,
        ),
        smoother_prepare=partial(
            smoother_prepare,
            get_dynamics_params=get_dynamics_params,
            store_gain=store_gain,
            store_chol_cov_given_next=store_chol_cov_given_next,
        ),
        smoother_combine=smoother_combine,
        associative=True,
    )


def init_prepare(
    model_inputs: ArrayTreeLike,
    get_init_params: GetInitParams,
    key: KeyArray | None = None,
) -> KalmanFilterState:
    """Prepare the initial state for the Kalman filter.

    Args:
        model_inputs: Model inputs.
        get_init_params: Function to get m0, chol_P0 from model inputs.
        key: JAX random key - not used.

    Returns:
        State for the Kalman filter.
            Contains mean and chol_cov (generalised Cholesky factor of covariance).
    """
    model_inputs = tree.map(lambda x: jnp.asarray(x), model_inputs)
    m0, chol_P0 = get_init_params(model_inputs)
    elem = filtering.FilterScanElement(
        A=jnp.zeros_like(chol_P0),
        b=m0,
        U=chol_P0,
        eta=jnp.zeros_like(m0),
        Z=jnp.zeros_like(chol_P0),
        ell=jnp.array(0.0),
    )
    return KalmanFilterState(elem=elem, model_inputs=model_inputs)


def filter_prepare(
    model_inputs: ArrayTreeLike,
    get_dynamics_params: GetDynamicsParams,
    get_observation_params: GetObservationParams,
    key: KeyArray | None = None,
) -> KalmanFilterState:
    """Prepare a state for an exact Kalman filter step.

    Args:
        model_inputs: Model inputs.
        get_dynamics_params: Function to get dynamics parameters, F, c, chol_Q.
        get_observation_params: Function to get observation parameters, H, d, chol_R, y.
        key: JAX random key - not used.

    Returns:
        Prepared state for Kalman filter.
    """
    model_inputs = tree.map(lambda x: jnp.asarray(x), model_inputs)
    F, c, chol_Q = get_dynamics_params(model_inputs)
    H, d, chol_R, y = get_observation_params(model_inputs)
    elem = filtering.associative_params_single(F, c, chol_Q, H, d, chol_R, y)
    return KalmanFilterState(elem=elem, model_inputs=model_inputs)


def filter_combine(
    state_1: KalmanFilterState,
    state_2: KalmanFilterState,
) -> KalmanFilterState:
    """Combine previous filter state with state prepared with latest model inputs.

    Applies exact Kalman predict + filter update in covariance square root form.
    Suitable for associative scan (as well as sequential scan).

    Args:
        state_1: State from previous time step.
        state_2: State prepared with latest model inputs.

    Returns:
        Combined Kalman filter state.
            Contains mean, chol_cov (generalised Cholesky factor of covariance)
            and log_normalizing_constant.
    """
    combined_elem = filtering.filtering_operator(
        state_1.elem,
        state_2.elem,
    )
    return KalmanFilterState(elem=combined_elem, model_inputs=state_2.model_inputs)


def smoother_prepare(
    filter_state: KalmanFilterState,
    get_dynamics_params: GetDynamicsParams,
    model_inputs: ArrayTreeLike,
    store_gain: bool = False,
    store_chol_cov_given_next: bool = False,
    key: KeyArray | None = None,
) -> KalmanSmootherState:
    """Prepare a state for an exact Kalman smoother step.

    Note that the model_inputs here are different to filter_state.model_inputs.
    The model_inputs required here are for the transition from t to t+1.
    filter_state.model_inputs represents the transition from t-1 to t.

    Args:
        filter_state: State generated by the Kalman filter at time t.
        get_dynamics_params: Function to get dynamics parameters, F, c, chol_Q,
            from model inputs.
        model_inputs: Model inputs for the transition from t to t+1.
        store_gain: Whether to store the gain matrix in the smoother state.
        store_chol_cov_given_next: Whether to store the chol_cov_given_next matrix
            in the smoother state.
        key: JAX random key - not used.

    Returns:
        Prepared state for the Kalman smoother.
    """
    model_inputs = tree.map(lambda x: jnp.asarray(x), model_inputs)
    F, c, chol_Q = get_dynamics_params(model_inputs)
    filter_mean = filter_state.mean
    filter_chol_cov = filter_state.chol_cov
    state = smoothing.associative_params_single(
        filter_mean, filter_chol_cov, F, c, chol_Q
    )
    return KalmanSmootherState(
        elem=state,
        gain=state.E if store_gain else None,
        chol_cov_given_next=state.D if store_chol_cov_given_next else None,
        model_inputs=model_inputs,
    )


def smoother_combine(
    state_1: KalmanSmootherState,
    state_2: KalmanSmootherState,
) -> KalmanSmootherState:
    """Combine smoother state from next time point with state prepared with latest model inputs.

    Remember smoothing iterates backwards in time.

    Applies exact Kalman smoother update in covariance square root form.
    Suitable for associative scan (as well as sequential scan).

    Args:
        state_1: State prepared with model inputs at time t.
        state_2: Smoother state at time t + 1.

    Returns:
        Combined Kalman smoother state.
            Contains mean, chol_cov (generalised Cholesky factor of covariance)
            and gain (which can be used to compute temporal cross-covariance).
    """
    state_elem = smoothing.smoothing_operator(
        state_2.elem,
        state_1.elem,
    )
    return KalmanSmootherState(
        elem=state_elem,
        gain=state_1.gain,
        chol_cov_given_next=state_1.chol_cov_given_next,
        model_inputs=state_1.model_inputs,
    )


def convert_filter_to_smoother_state(
    filter_state: ArrayTreeLike,
    model_inputs: ArrayTreeLike | None = None,
    store_gain: bool = False,
    store_chol_cov_given_next: bool = False,
    key: KeyArray | None = None,
) -> KalmanSmootherState:
    """Convert the filter state to a smoother state.

    Useful for the final filter state which is equivalent to the final smoother state.

    Args:
        filter_state: Filter state.
        model_inputs: Only used to create an empty model_inputs tree
            (the values are ignored).
            Useful so that the final smoother state has the same structure as the rest.
            By default, filter_state.model_inputs is used. So this
            is only needed if the smoother model_inputs have a different tree
            structure to filter_state.model_inputs.
        store_gain: Whether to store the gain matrix in the smoother state.
        store_chol_cov_given_next: Whether to store the chol_cov_given_next matrix
            in the smoother state.
        key: JAX random key - not used.

    Returns:
        Smoother state, same data as filter state just different structure.
            Note that the model_inputs are set to dummy values.
    """
    if model_inputs is None:
        model_inputs = filter_state.model_inputs

    dummy_model_inputs = dummy_tree_like(model_inputs)

    elem = smoothing.SmootherScanElement(
        g=filter_state.mean,
        D=filter_state.chol_cov,
        E=jnp.zeros_like(filter_state.chol_cov),
    )
    return KalmanSmootherState(
        elem=elem,
        gain=dummy_tree_like(filter_state.chol_cov) if store_gain else None,
        chol_cov_given_next=dummy_tree_like(filter_state.chol_cov)
        if store_chol_cov_given_next
        else None,
        model_inputs=dummy_model_inputs,
    )
