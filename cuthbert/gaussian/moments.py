from functools import partial

from jax import eval_shape, tree
from jax import numpy as jnp

from cuthbert.gaussian.kalman import (
    GetInitParams,
    KalmanSmootherState,
    convert_filter_to_smoother_state,
    smoother_combine,
)
from cuthbert.gaussian.types import (
    GetDynamicsMoments,
    GetObservationMoments,
    LinearizedKalmanFilterState,
)
from cuthbert.inference import Filter, Smoother
from cuthbertlib.kalman import filtering, smoothing
from cuthbertlib.linearize import linearize_moments
from cuthbertlib.types import (
    ArrayTreeLike,
    KeyArray,
)


def build_filter(
    get_init_params: GetInitParams,
    get_dynamics_params: GetDynamicsMoments,
    get_observation_params: GetObservationMoments,
) -> Filter:
    """
    Build linearized moments Kalman inference filter for conditionally Gaussian SSMs.

    Args:
        get_init_params: Function to get m0, chol_P0 to initialize filter state,
            given model inputs sufficient to define p(x_0) = N(m0, chol_P0 @ chol_P0^T).
        get_dynamics_params: Function to get dynamics conditional mean and
            (generalised) Cholesky covariance function and linearization point.
        get_observation_params: Function to get observation conditional mean,
            (generalised) Cholesky covariance function, linearization point and
            observation.

    Returns:
        Linearized moments Kalman filter object, not suitable for associative scan.
    """
    return Filter(
        init_prepare=partial(
            init_prepare,
            get_init_params=get_init_params,
            get_observation_params=get_observation_params,
        ),
        filter_prepare=partial(
            filter_prepare,
            get_init_params=get_init_params,
        ),
        filter_combine=partial(
            filter_combine,
            get_dynamics_params=get_dynamics_params,
            get_observation_params=get_observation_params,
        ),
        associative=False,
    )


def build_smoother(
    get_dynamics_params: GetDynamicsMoments,
) -> Smoother:
    """
    Build linearized moments Kalman inference smoother for conditionally Gaussian SSMs.

    Args:
        get_dynamics_params: Function to get dynamics conditional mean and
            (generalised) Cholesky covariance from linearization point and model inputs.

    Returns:
        Linearized moments Kalman smoother object, suitable for associative scan.
    """
    return Smoother(
        smoother_prepare=partial(
            smoother_prepare, get_dynamics_params=get_dynamics_params
        ),
        smoother_combine=smoother_combine,
        convert_filter_to_smoother_state=convert_filter_to_smoother_state,
        associative=True,
    )


def init_prepare(
    model_inputs: ArrayTreeLike,
    get_init_params: GetInitParams,
    get_observation_params: GetObservationMoments,
    key: KeyArray | None = None,
) -> LinearizedKalmanFilterState:
    """
    Prepare the initial state for the linearized moments Kalman filter.

    Args:
        model_inputs: Model inputs.
        get_init_params: Function to get m0, chol_P0 from model inputs.
        get_observation_params: Function to get observation conditional mean,
            (generalised) Cholesky covariance function, linearization point and
            observation.
        key: JAX random key - not used.

    Returns:
        State for the linearized moments Kalman filter.
            Contains mean, chol_cov (generalised Cholesky factor of covariance)
            and log_likelihood.
    """
    model_inputs = tree.map(lambda x: jnp.asarray(x), model_inputs)
    m0, chol_P0 = get_init_params(model_inputs)

    predict_state = LinearizedKalmanFilterState(
        mean=m0,
        chol_cov=chol_P0,
        log_likelihood=jnp.array(0.0),
        model_inputs=model_inputs,
        mean_prev=jnp.full_like(m0, jnp.nan),
    )

    mean_and_chol_cov_func, linearization_point, y = get_observation_params(
        predict_state, model_inputs
    )

    H, d, chol_R = linearize_moments(mean_and_chol_cov_func, linearization_point)
    (m, chol_P), ell = filtering.update(m0, chol_P0, H, d, chol_R, y)

    return LinearizedKalmanFilterState(
        mean=m,
        chol_cov=chol_P,
        log_likelihood=ell,
        model_inputs=model_inputs,
        mean_prev=jnp.full_like(m0, jnp.nan),
    )


def filter_prepare(
    model_inputs: ArrayTreeLike,
    get_init_params: GetInitParams,
    key: KeyArray | None = None,
) -> LinearizedKalmanFilterState:
    """
    Prepare a state for an linearized moments Kalman filter step,
    just passes through model inputs.

    Args:
        model_inputs: Model inputs.
        get_init_params: Function to get m0, chol_P0 from model inputs, just used to
            infer shape of mean and chol_cov.
        key: JAX random key - not used.

    Returns:
        Prepared state for linearized moments Kalman filter.
    """
    model_inputs = tree.map(lambda x: jnp.asarray(x), model_inputs)
    dummy_mean, dummy_chol_cov = eval_shape(get_init_params, model_inputs)
    mean = jnp.empty_like(dummy_mean)
    chol_cov = jnp.empty_like(dummy_chol_cov)
    return LinearizedKalmanFilterState(
        mean=mean,
        chol_cov=chol_cov,
        log_likelihood=jnp.array(0.0),
        model_inputs=model_inputs,
        mean_prev=jnp.full_like(mean, jnp.nan),
    )


def filter_combine(
    state_1: LinearizedKalmanFilterState,
    state_2: LinearizedKalmanFilterState,
    get_dynamics_params: GetDynamicsMoments,
    get_observation_params: GetObservationMoments,
) -> LinearizedKalmanFilterState:
    """
    Combine filter state from previous time point with state prepared
    with latest model inputs.

    Applies linearized moments Kalman predict + filter update in covariance square root form.
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
        Predicted and updated linearized moments Kalman filter state.
            Contains mean, chol_cov (generalised Cholesky factor of covariance)
            and log_likelihood.
    """
    dynamics_mean_and_chol_cov_func, dynamics_linearization_point = get_dynamics_params(
        state_1, state_2.model_inputs
    )

    F, c, chol_Q = linearize_moments(
        dynamics_mean_and_chol_cov_func, dynamics_linearization_point
    )

    observation_mean_and_chol_cov_func, observation_linearization_point, y = (
        get_observation_params(state_1, state_2.model_inputs)
    )

    H, d, chol_R = linearize_moments(
        observation_mean_and_chol_cov_func, observation_linearization_point
    )

    predict_mean, predict_chol_cov = filtering.predict(
        state_1.mean, state_1.chol_cov, F, c, chol_Q
    )
    (update_mean, update_chol_cov), log_likelihood = filtering.update(
        predict_mean, predict_chol_cov, H, d, chol_R, y
    )

    return LinearizedKalmanFilterState(
        mean=update_mean,
        chol_cov=update_chol_cov,
        log_likelihood=state_1.log_likelihood + log_likelihood,
        model_inputs=state_2.model_inputs,
        mean_prev=state_1.mean,
    )


def smoother_prepare(
    filter_state: LinearizedKalmanFilterState,
    get_dynamics_params: GetDynamicsMoments,
    model_inputs: ArrayTreeLike,
    key: KeyArray | None = None,
) -> KalmanSmootherState:
    """
    Prepare a state for an extended Kalman smoother step.

    Note that the model_inputs here are different to filter_state.model_inputs.
    The model_inputs required here are for the transition from t to t+1.
    filter_state.model_inputs represents the transition from t-1 to t.

    Args:
        filter_state: State generated by the extended Kalman filter at time t.
        get_dynamics_params: Function to get dynamics conditional mean and
            (generalised) Cholesky covariance from linearization point and model inputs.
        model_inputs: Model inputs for the transition from t to t+1.
        key: JAX random key - not used.

    Returns:
        Prepared state for the Kalman smoother.
    """
    model_inputs = tree.map(lambda x: jnp.asarray(x), model_inputs)
    filter_mean = filter_state.mean
    filter_chol_cov = filter_state.chol_cov

    dynamics_mean_and_chol_cov_func, dynamics_linearization_point = get_dynamics_params(
        filter_state, model_inputs
    )

    F, c, chol_Q = linearize_moments(
        dynamics_mean_and_chol_cov_func, dynamics_linearization_point
    )

    state = smoothing.associative_params_single(
        filter_mean, filter_chol_cov, F, c, chol_Q
    )
    return KalmanSmootherState(elem=state, gain=state.E, model_inputs=model_inputs)
