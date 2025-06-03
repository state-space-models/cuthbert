from typing import Callable
from functools import partial

from cuthbertlib.types import (
    Array,
    ArrayTreeLike,
    KeyArray,
)
from cuthbertlib.kalman import filtering, smoothing
from cuthbert.inference import SSMInference


# model_inputs -> m0, chol_P0
GetInitParams = Callable[[ArrayTreeLike], tuple[Array, Array]]

# model_inputs -> F, c, chol_Q
GetDynamicsParams = Callable[[ArrayTreeLike], tuple[Array, Array, Array]]

# model_inputs -> H, d, chol_R, y
GetObservationParams = Callable[[ArrayTreeLike], tuple[Array, Array, Array, Array]]


def build(
    get_init_params: GetInitParams,
    get_dynamics_params: GetDynamicsParams,
    get_observation_params: GetObservationParams,
) -> SSMInference:
    """
    Build an exact Kalman filter and smoother.

    Args:
        get_init_params: Function to get m0, chol_P0 to initialize filter state.
            Typically this is non-zero for the first time step and otherwise zero.
        get_dynamics_params: Function to get dynamics parameters, F, c, chol_Q
            given model inputs.
        get_observation_params: Function to get observation parameters, H, d, chol_R, y
            given model inputs.

    Returns:
        SSMInference: Inference object for exact Kalman filter and smoother.
            Suitable for associative scan.
    """
    return SSMInference(
        FilterPrepare=partial(
            filter_prepare,
            get_init_params=get_init_params,
            get_dynamics_params=get_dynamics_params,
            get_observation_params=get_observation_params,
        ),
        FilterCombine=filter_combine,
        SmootherPrepare=partial(
            smoother_prepare, get_dynamics_params=get_dynamics_params
        ),
        SmootherCombine=smoother_combine,
        associative_filter=True,
        associative_smoother=True,
    )


def filter_prepare(
    model_inputs: ArrayTreeLike,
    get_init_params: GetInitParams,
    get_dynamics_params: GetDynamicsParams,
    get_observation_params: GetObservationParams,
    key: KeyArray | None = None,
) -> filtering.FilterScanElement:
    """
    Prepare a state for an exact Kalman filter step.

    Args:
        model_inputs: Model inputs.
        get_init_params: Function to get m0, chol_P0 to initialize filter state.
            Typically this is non-zero for the first time step and otherwise zero.
        get_dynamics_params: Function to get dynamics parameters, F, c, chol_Q.
        get_observation_params: Function to get observation parameters, H, d, chol_R, y.
        key: JAX random key - not used.

    Returns:
        KalmanState: State for the Kalman filter.
            Contains mean and chol_cov (generalised Cholesky factor of covariance).
    """
    m0, chol_P0 = get_init_params(model_inputs)
    F, c, chol_Q = get_dynamics_params(model_inputs)
    H, d, chol_R, y = get_observation_params(model_inputs)
    return filtering._sqrt_associative_params_single(
        m0,
        chol_P0,
        F,
        c,
        chol_Q,
        H,
        d,
        chol_R,
        y,
    )


def filter_combine(
    state_1: filtering.FilterScanElement,
    state_2: filtering.FilterScanElement,
    key: KeyArray | None = None,
) -> filtering.FilterScanElement:
    """
    Combine two Kalman filter states.

    Applies exact Kalman predict + filter update in covariance square root form.
    Suitable for associative scan.

    Args:
        state_1: State from previous time step.
        state_2: State prepared with latest model inputs.
        key: JAX random key - not used.

    Returns:
        FilterScanElement: Combined Kalman filter state.
    """
    return filtering.sqrt_filtering_operator(
        state_1,
        state_2,
    )


def smoother_prepare(
    filter_state: filtering.FilterScanElement,
    model_inputs: ArrayTreeLike,
    get_dynamics_params: GetDynamicsParams,
    key: KeyArray | None = None,
) -> smoothing.SmootherScanElement:
    """
    Prepare a state for an exact Kalman smoother step.

    Args:
        filter_state: Kalman filter state.
        model_inputs: Model inputs.
        get_dynamics_params: Function to get dynamics parameters, F, c, chol_Q.
        key: JAX random key - not used.

    Returns:
        SmootherScanElement: State for the Kalman smoother.
            Contains mean and chol_cov (generalised Cholesky factor of covariance),
            as well as other quantities needed for the smoother update.
    """
    F, c, chol_Q = get_dynamics_params(model_inputs)
    filter_mean = filter_state.mean
    filter_chol_cov = filter_state.chol_cov
    return smoothing._sqrt_associative_params_single(
        filter_mean,
        filter_chol_cov,
        F,
        c,
        chol_Q,
    )


def smoother_combine(
    state_1: smoothing.SmootherScanElement,
    state_2: smoothing.SmootherScanElement,
    key: KeyArray | None = None,
) -> smoothing.SmootherScanElement:
    """
    Combine two Kalman smoother states.

    Applies exact Kalman smoother update in covariance square root form.
    Suitable for associative scan.

    Args:
        state_1: State prepared from current time step.
        state_2: State prepared at next time step.
        key: JAX random key - not used.

    Returns:
        SmootherScanElement: Combined Kalman smoother state.
    """
    return smoothing.sqrt_smoothing_operator(
        state_1,
        state_2,
    )
