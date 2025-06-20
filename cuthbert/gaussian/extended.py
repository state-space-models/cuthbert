from typing import NamedTuple, Protocol
from functools import partial
from jax import numpy as jnp, tree

from cuthbertlib.types import (
    Array,
    ArrayTree,
    ArrayTreeLike,
    KeyArray,
)
from cuthbertlib.kalman import filtering, smoothing
from cuthbertlib.linearize import linearize_moments
from cuthbert.inference import Inference
from cuthbert.gaussian.kalman import (
    KalmanFilterState,
    KalmanSmootherState,
    GetInitParams,
    smoother_combine,
    convert_filter_to_smoother_state,
)


class GetDynamicsExtendedParams(Protocol):
    def __call__(
        self,
        x: Array,
        model_inputs: ArrayTreeLike,
    ) -> tuple[Array, Array]:
        """
        Get dynamics conditional mean and (generalised) Cholesky covariance
            from model inputs and linearization point.

        Args:
            x: Linearization point.
            model_inputs: Model inputs.

        Returns:
            Tuple with conditional mean and (generalised) Cholesky covariance.
        """
        ...


class GetObservationExtendedParams(Protocol):
    def __call__(
        self, x: Array, model_inputs: ArrayTreeLike
    ) -> tuple[Array, Array, Array]:
        """
        Get observation conditional mean, (generalised) Cholesky covariance
            and the observation itself from model inputs and linearization point.

        Args:
            x: Linearization point.
            model_inputs: Model inputs.

        Returns:
            Tuple with conditional mean, (generalised) Cholesky covariance
                and observation.
        """
        ...


class ExtendedKalmanFilterState(NamedTuple):
    mean: Array | None
    chol_cov: Array | None
    log_likelihood: Array
    model_inputs: ArrayTree


def build(
    get_init_params: GetInitParams,
    get_dynamics_params: GetDynamicsExtendedParams,
    get_observation_params: GetObservationExtendedParams,
) -> Inference:
    """
    Build extended Kalman inference object for conditionally Gaussian SSMs.

    Args:
        get_init_params: Function to get m0, chol_P0 to initialize filter state,
            given model inputs sufficient to define p(x_0) = N(m0, chol_P0 @ chol_P0^T).
        get_dynamics_params: Function to get dynamics conditional mean and
            (generalised) Cholesky covariance from linearization point and model inputs.
        get_observation_params: Function to get observation conditional mean,
            (generalised) Cholesky covariance and observation from linearization point
            and model inputs.

    Returns:
        Inference object for extended Kalman filter and smoother.
            Filter not suitable for associative scan, smoother suitable.
    """
    return Inference(
        init_prepare=partial(init_prepare, get_init_params=get_init_params),
        filter_prepare=filter_prepare,
        filter_combine=partial(
            filter_combine,
            get_dynamics_params=get_dynamics_params,
            get_observation_params=get_observation_params,
        ),
        smoother_prepare=partial(
            smoother_prepare, get_dynamics_params=get_dynamics_params
        ),
        smoother_combine=smoother_combine,
        convert_filter_to_smoother_state=convert_filter_to_smoother_state,
        associative_filter=False,
        associative_smoother=True,
    )


def init_prepare(
    model_inputs: ArrayTreeLike,
    get_init_params: GetInitParams,
    key: KeyArray | None = None,
) -> ExtendedKalmanFilterState:
    """
    Prepare the initial state for the extended Kalman filter.

    Args:
        model_inputs: Model inputs.
        get_init_params: Function to get m0, chol_P0 from model inputs.
        key: JAX random key - not used.

    Returns:
        State for the extended Kalman filter.
            Contains mean, chol_cov (generalised Cholesky factor of covariance)
            and log_likelihood.
    """
    m0, chol_P0 = get_init_params(model_inputs)
    return ExtendedKalmanFilterState(
        mean=m0,
        chol_cov=chol_P0,
        log_likelihood=jnp.array(0.0),
        model_inputs=tree.map(lambda x: jnp.asarray(x), model_inputs),
    )


def filter_prepare(
    model_inputs: ArrayTreeLike,
    key: KeyArray | None = None,
) -> ExtendedKalmanFilterState:
    """
    Prepare a state for an extended Kalman filter step - just passes through model inputs.

    Args:
        model_inputs: Model inputs.
        key: JAX random key - not used.

    Returns:
        Prepared state for extended Kalman filter.
    """
    return ExtendedKalmanFilterState(
        mean=None,
        chol_cov=None,
        log_likelihood=jnp.array(0.0),
        model_inputs=tree.map(lambda x: jnp.asarray(x), model_inputs),
    )


def filter_combine(
    state_1: ExtendedKalmanFilterState,
    state_2: ExtendedKalmanFilterState,
    get_dynamics_params: GetDynamicsExtendedParams,
    get_observation_params: GetObservationExtendedParams,
) -> ExtendedKalmanFilterState:
    """
    Combine filter state from previous time point with state prepared
    with latest model inputs.

    Applies extended Kalman predict + filter update in covariance square root form.
    Not suitable for associative scan.

    Args:
        state_1: State from previous time step.
        state_2: State prepared (only access model_inputs attribute).
        get_dynamics_params: Function to get dynamics conditional mean and
            (generalised) Cholesky covariance from linearization point and model inputs.
        get_observation_params: Function to get observation conditional mean,
            (generalised) Cholesky covariance and observation from linearization point
            and model inputs.

    Returns:
        Predicted and updated extended Kalman filter state.
            Contains mean, chol_cov (generalised Cholesky factor of covariance)
            and log_likelihood.
    """
    if state_1.mean is None or state_1.chol_cov is None:
        raise ValueError("State from previous time step must have mean and chol_cov.")

    linearization_point = state_1.mean

    def dynamics_mean_and_chol_cov(x):
        return get_dynamics_params(x, state_2.model_inputs)

    F, c, chol_Q = linearize_moments(dynamics_mean_and_chol_cov, linearization_point)

    def observation_mean_and_chol_cov_and_y(x):
        return get_observation_params(x, state_2.model_inputs)

    H, d, chol_R, y = linearize_moments(
        observation_mean_and_chol_cov_and_y, linearization_point, has_aux=True
    )

    predict_mean, predict_chol_cov = filtering.predict(
        state_1.mean, state_1.chol_cov, F, c, chol_Q
    )
    (update_mean, update_chol_cov), log_likelihood = filtering.update(
        predict_mean, predict_chol_cov, H, d, chol_R, y
    )

    return ExtendedKalmanFilterState(
        mean=update_mean,
        chol_cov=update_chol_cov,
        log_likelihood=state_1.log_likelihood + log_likelihood,
        model_inputs=tree.map(lambda x: jnp.asarray(x), state_2.model_inputs),
    )


def smoother_prepare(
    filter_state: KalmanFilterState,
    model_inputs: ArrayTreeLike,
    get_dynamics_params: GetDynamicsExtendedParams,
    key: KeyArray | None = None,
) -> KalmanSmootherState:
    """
    Prepare a state for an exact Kalman smoother step.

    Args:
        filter_state: State generated by the Kalman filter at the previous time point.
        model_inputs: Model inputs.
        get_dynamics_params: Function to get dynamics conditional mean and
            (generalised) Cholesky covariance from linearization point and model inputs.
        key: JAX random key - not used.

    Returns:
        Prepared state for the Kalman smoother.
    """
    filter_mean = filter_state.mean
    filter_chol_cov = filter_state.chol_cov

    def dynamics_mean_and_chol_cov(x):
        return get_dynamics_params(x, model_inputs)

    F, c, chol_Q = linearize_moments(dynamics_mean_and_chol_cov, filter_mean)

    state = smoothing.associative_params_single(
        filter_mean, filter_chol_cov, F, c, chol_Q
    )
    return KalmanSmootherState(elem=state, gain=state.E)
