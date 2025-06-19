from typing import NamedTuple, Protocol
from functools import partial
from jax import numpy as jnp

from cuthbertlib.types import (
    Array,
    ArrayTreeLike,
    KeyArray,
)
from cuthbertlib.kalman import filtering, smoothing
from cuthbert.inference import Inference


class GetInitParams(Protocol):
    def __call__(self, model_inputs: ArrayTreeLike) -> tuple[Array, Array]:
        """Get initial parameters (m0, chol_P0) from model inputs."""
        ...


class GetDynamicsParams(Protocol):
    def __call__(self, model_inputs: ArrayTreeLike) -> tuple[Array, Array, Array]:
        """Get dynamics parameters (F, c, chol_Q) from model inputs."""
        ...


class GetObservationParams(Protocol):
    def __call__(
        self, model_inputs: ArrayTreeLike
    ) -> tuple[Array, Array, Array, Array]:
        """Get observation parameters (H, d, chol_R, y) from model inputs."""
        ...


class KalmanFilterState(NamedTuple):
    elem: filtering.FilterScanElement

    @property
    def mean(self) -> Array:
        return self.elem.b

    @property
    def chol_cov(self) -> Array:
        return self.elem.U

    @property
    def log_likelihood(self) -> Array:
        return self.elem.ell


class KalmanSmootherState(NamedTuple):
    elem: smoothing.SmootherScanElement
    gain: Array | None = None

    @property
    def mean(self) -> Array:
        return self.elem.g

    @property
    def chol_cov(self) -> Array:
        return self.elem.D


def build(
    get_init_params: GetInitParams,
    get_dynamics_params: GetDynamicsParams,
    get_observation_params: GetObservationParams,
) -> Inference:
    """
    Build exact Kalman inference object for linear Gaussian SSMs.

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
        SSMInference: Inference object for exact Kalman filter and smoother.
            Suitable for associative scan.
    """
    return Inference(
        init_prepare=partial(init_prepare, get_init_params=get_init_params),
        filter_prepare=partial(
            filter_prepare,
            get_dynamics_params=get_dynamics_params,
            get_observation_params=get_observation_params,
        ),
        filter_combine=filter_combine,
        smoother_prepare=partial(
            smoother_prepare, get_dynamics_params=get_dynamics_params
        ),
        smoother_combine=smoother_combine,
        convert_filter_to_smoother_state=convert_filter_to_smoother_state,
        associative_filter=True,
        associative_smoother=True,
    )


def init_prepare(
    model_inputs: ArrayTreeLike,
    get_init_params: GetInitParams,
    key: KeyArray | None = None,
) -> KalmanFilterState:
    """
    Prepare the initial state for the Kalman filter.

    Args:
        model_inputs: Model inputs.
        get_init_params: Function to get m0, chol_P0 from model inputs.
        key: JAX random key - not used.

    Returns:
        State for the Kalman filter.
            Contains mean and chol_cov (generalised Cholesky factor of covariance).
    """
    m0, chol_P0 = get_init_params(model_inputs)
    elem = filtering.FilterScanElement(
        A=jnp.zeros_like(chol_P0),
        b=m0,
        U=chol_P0,
        eta=jnp.zeros_like(m0),
        Z=jnp.zeros_like(chol_P0),
        ell=jnp.array(0.0),
    )
    return KalmanFilterState(elem=elem)


def filter_prepare(
    model_inputs: ArrayTreeLike,
    get_dynamics_params: GetDynamicsParams,
    get_observation_params: GetObservationParams,
    key: KeyArray | None = None,
) -> KalmanFilterState:
    """
    Prepare a state for an exact Kalman filter step.

    Args:
        model_inputs: Model inputs.
        get_dynamics_params: Function to get dynamics parameters, F, c, chol_Q.
        get_observation_params: Function to get observation parameters, H, d, chol_R, y.
        key: JAX random key - not used.

    Returns:
        Prepared state for Kalman filter.
    """
    F, c, chol_Q = get_dynamics_params(model_inputs)
    H, d, chol_R, y = get_observation_params(model_inputs)
    elem = filtering.associative_params_single(F, c, chol_Q, H, d, chol_R, y)
    return KalmanFilterState(elem=elem)


def filter_combine(
    state_1: KalmanFilterState,
    state_2: KalmanFilterState,
) -> KalmanFilterState:
    """
    Combine filter state from previous time point with state prepared
    with latest model inputs.

    Applies exact Kalman predict + filter update in covariance square root form.
    Suitable for associative scan (as well as sequential scan).

    Args:
        state_1: State from previous time step.
        state_2: State prepared with latest model inputs.

    Returns:
        Combined Kalman filter state.
            Contains mean, chol_cov (generalised Cholesky factor of covariance)
            and log_likelihood.
    """
    combined_elem = filtering.filtering_operator(
        state_1.elem,
        state_2.elem,
    )
    return KalmanFilterState(elem=combined_elem)


def smoother_prepare(
    filter_state: KalmanFilterState,
    model_inputs: ArrayTreeLike,
    get_dynamics_params: GetDynamicsParams,
    key: KeyArray | None = None,
) -> KalmanSmootherState:
    """
    Prepare a state for an exact Kalman smoother step.

    Args:
        filter_state: State generated by the Kalman filter at the previous time point.
        model_inputs: Model inputs.
        get_dynamics_params: Function to get dynamics parameters, F, c, chol_Q,
            from model inputs.
        key: JAX random key - not used.

    Returns:
        Prepared state for the Kalman smoother.
    """
    F, c, chol_Q = get_dynamics_params(model_inputs)
    filter_mean = filter_state.mean
    filter_chol_cov = filter_state.chol_cov
    state = smoothing.associative_params_single(
        filter_mean, filter_chol_cov, F, c, chol_Q
    )
    return KalmanSmootherState(elem=state, gain=state.E)


def smoother_combine(
    state_1: KalmanSmootherState,
    state_2: KalmanSmootherState,
) -> KalmanSmootherState:
    """
    Combine smoother state from next time point with state prepared
    with latest model inputs.

    Remember smoothing iterates backwards in time.

    Applies exact Kalman smoother update in covariance square root form.
    Suitable for associative scan (as well as sequential scan).

    Args:
        state_1: State prepared with model inputs at time t.
        state_2: Smoother state at time t + 1.

    Returns:
        Combined Kalman filter state.
            Contains mean, chol_cov (generalised Cholesky factor of covariance)
            and gain (which can be used to compute temporal cross-covariance).
    """
    state_elem = smoothing.smoothing_operator(
        state_2.elem,
        state_1.elem,
    )
    return KalmanSmootherState(elem=state_elem, gain=state_1.gain)


def convert_filter_to_smoother_state(
    filter_state: KalmanFilterState,
) -> KalmanSmootherState:
    """
    Convert the filter state to a smoother state.

    Useful for the final filter state which is equivalent to the final smoother state.

    Args:
        filter_state: Filter state.

    Returns:
        Smoother state, same data as filter state just different structure.
    """
    elem = smoothing.SmootherScanElement(
        g=filter_state.mean,
        D=filter_state.chol_cov,
        E=jnp.zeros_like(filter_state.chol_cov),
    )
    return KalmanSmootherState(
        elem=elem,
        gain=jnp.full_like(filter_state.chol_cov, jnp.nan),
    )
