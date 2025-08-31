from functools import partial
from typing import Callable, NamedTuple, Protocol

from jax import eval_shape, tree
from jax import numpy as jnp

from cuthbert.gaussian.kalman import (
    KalmanSmootherState,
    _convert_filter_to_smoother_state,
    smoother_combine,
)
from cuthbert.inference import Filter, Smoother
from cuthbertlib.kalman import filtering, smoothing
from cuthbertlib.linearize import linearize_log_density, linearize_taylor
from cuthbertlib.types import (
    Array,
    ArrayTree,
    ArrayTreeLike,
    KeyArray,
    LogConditionalDensity,
    LogDensity,
    ScalarArray,
)

LogPotential = Callable[[ArrayTreeLike], ScalarArray]


class LogDensityKalmanFilterState(NamedTuple):
    mean: Array
    chol_cov: Array
    log_likelihood: Array
    model_inputs: ArrayTree


class GetInitLogDensity(Protocol):
    def __call__(self, model_inputs: ArrayTreeLike) -> tuple[LogDensity, Array]:
        """Get the initial log density and initial linearization point."""
        ...


class GetDynamicsLogDensity(Protocol):
    def __call__(
        self, state: LogDensityKalmanFilterState, model_inputs: ArrayTreeLike
    ) -> tuple[LogConditionalDensity, Array, Array]:
        """Get the dynamics log density and linearization points
        (for the previous and current time points)"""
        ...


class GetObservationLogDensity(Protocol):
    def __call__(
        self, state: LogDensityKalmanFilterState | None, model_inputs: ArrayTreeLike
    ) -> tuple[LogConditionalDensity, Array, Array]:
        """Extract observation log density, linearization point and observation.
        At first time point, state is None, otherwise it is the predicted state."""
        ...


class GetLogPotential(Protocol):
    def __call__(
        self, state: LogDensityKalmanFilterState | None, model_inputs: ArrayTreeLike
    ) -> tuple[LogPotential, Array]:
        """Extract log potential and linearization point.
        At first time point, state is None, otherwise it is the predicted state."""
        ...


def build_filter(
    get_init_log_density: GetInitLogDensity,
    get_dynamics_log_density: GetDynamicsLogDensity,
    get_observation_func: GetObservationLogDensity | GetLogPotential,
) -> Filter:
    """
    Build linearized log density Kalman inference filter.

    Args:
        get_init_log_density: Function to get log density log p(x_0)
            and linearization point.
        get_dynamics_log_density: Function to get dynamics log density log p(x_t+1 | x_t)
            and linearization points (for the previous and current time points)
        get_observation_func: Function to get observation log density log p(y_t | x_t)
            and linearization point and observation.

    Returns:
        Log density Kalman filter object, not suitable for associative scan.
    """
    return Filter(
        init_prepare=partial(
            init_prepare,
            get_init_log_density=get_init_log_density,
            get_observation_func=get_observation_func,
        ),
        filter_prepare=partial(
            filter_prepare,
            get_init_log_density=get_init_log_density,
        ),
        filter_combine=partial(
            filter_combine,
            get_dynamics_log_density=get_dynamics_log_density,
            get_observation_func=get_observation_func,
        ),
        associative=False,
    )


def build_smoother(
    get_dynamics_log_density: GetDynamicsLogDensity,
) -> Smoother:
    """
    Build linearized log density Kalman inference smoother.

    Args:
        get_dynamics_log_density: Function to get dynamics log density log p(x_t+1 | x_t)
            and linearization points (for the previous and current time points)

    Returns:
        Log density Kalman smoother object, suitable for associative scan.
    """
    return Smoother(
        smoother_prepare=partial(
            smoother_prepare, get_dynamics_log_density=get_dynamics_log_density
        ),
        smoother_combine=smoother_combine,
        convert_filter_to_smoother_state=convert_filter_to_smoother_state,
        associative=True,
    )


def process_observation(
    state: LogDensityKalmanFilterState | None,
    get_observation_func: GetObservationLogDensity | GetLogPotential,
    model_inputs: ArrayTreeLike,
) -> tuple[Array, Array, Array, Array]:
    """Process observation for log density Kalman filter."""
    observation_output = get_observation_func(state, model_inputs)
    if len(observation_output) == 3:
        observation_log_density_func, linearization_point, observation = (
            observation_output
        )
        H, d, chol_R = linearize_log_density(
            observation_log_density_func, linearization_point, observation
        )
    else:
        log_potential, linearization_point = observation_output
        d, chol_R = linearize_taylor(log_potential, linearization_point)
        # dummy mat and observation as potential is unconditional
        # Note the minus sign as linear potential is -0.5 (x - d)^T (R R^T)^{-1} (x - d)
        # and kalman expects -0.5 (y - H @ x - d)^T (R R^T)^{-1} (y - H @ x - d)
        H = -jnp.eye(d.shape[0])
        observation = jnp.zeros_like(d)
    return H, d, chol_R, observation


def init_prepare(
    model_inputs: ArrayTreeLike,
    get_init_log_density: GetInitLogDensity,
    get_observation_func: GetObservationLogDensity | GetLogPotential,
    key: KeyArray | None = None,
) -> LogDensityKalmanFilterState:
    """
    Prepare the initial state for the log density Kalman filter.

    Args:
        model_inputs: Model inputs.
        get_init_log_density: Function that returns log density log p(x_0)
            and linearization point.
        get_observation_func: Function that returns either
            - An observation log density
                function log p(y_0 | x_0) as well as points x_0 and y_0
                to linearize around.
            - A log potential function log G(None, x_0) and a linearization point x_0.
        key: JAX random key - not used.

    Returns:
        State for the log density Kalman filter.
            Contains mean, chol_cov (generalised Cholesky factor of covariance)
            and log_likelihood.
    """
    model_inputs = tree.map(lambda x: jnp.asarray(x), model_inputs)
    init_log_density, linearization_point = get_init_log_density(model_inputs)

    _, m0, chol_P0 = linearize_log_density(
        lambda _, x: init_log_density(x), linearization_point, linearization_point
    )

    H, d, chol_R, observation = process_observation(
        None, get_observation_func, model_inputs
    )

    (m, chol_P), ell = filtering.update(m0, chol_P0, H, d, chol_R, observation)

    return LogDensityKalmanFilterState(
        mean=m,
        chol_cov=chol_P,
        log_likelihood=ell,
        model_inputs=model_inputs,
    )


def filter_prepare(
    model_inputs: ArrayTreeLike,
    get_init_log_density: GetInitLogDensity,
    key: KeyArray | None = None,
) -> LogDensityKalmanFilterState:
    """
    Prepare a state for a log density Kalman filter step,
    just passes through model inputs.

    Args:
        model_inputs: Model inputs.
        get_init_log_density: Function that returns log density log p(x_0)
            and linearization point. Only used to infer shape of mean and chol_cov.
        key: JAX random key - not used.

    Returns:
        Prepared state for log density Kalman filter.
    """
    model_inputs = tree.map(lambda x: jnp.asarray(x), model_inputs)
    dummy_mean = eval_shape(lambda mi: get_init_log_density(mi)[1], model_inputs)
    dummy_chol_cov = jnp.cov(dummy_mean[..., None])
    mean = jnp.empty_like(dummy_mean)
    chol_cov = jnp.empty_like(dummy_chol_cov)

    return LogDensityKalmanFilterState(
        mean=mean,
        chol_cov=chol_cov,
        log_likelihood=jnp.array(0.0),
        model_inputs=model_inputs,
    )


def filter_combine(
    state_1: LogDensityKalmanFilterState,
    state_2: LogDensityKalmanFilterState,
    get_dynamics_log_density: GetDynamicsLogDensity,
    get_observation_func: GetObservationLogDensity | GetLogPotential,
) -> LogDensityKalmanFilterState:
    """
    Combine filter state from previous time point with state prepared
    with latest model inputs.

    Applies linearized log density Kalman predict + filter update in covariance square
    root form.
    Not suitable for associative scan.

    Args:
        state_1: State from previous time step.
        state_2: State prepared (only access model_inputs attribute).
        get_dynamics_log_density: Function to get dynamics log density log p(x_t+1 | x_t)
            and linearization points (for the previous and current time points)
        get_observation_func: Function to get observation log density log p(y_t | x_t)
            and linearization point and observation.

    Returns:
        Predicted and updated log density Kalman filter state.
            Contains mean, chol_cov (generalised Cholesky factor of covariance)
            and log_likelihood.
    """

    log_dynamics_density, linearization_point_prev, linearization_point_curr = (
        get_dynamics_log_density(state_1, state_2.model_inputs)
    )

    F, c, chol_Q = linearize_log_density(
        log_dynamics_density, linearization_point_prev, linearization_point_curr
    )

    H, d, chol_R, observation = process_observation(
        None, get_observation_func, state_2.model_inputs
    )

    predict_mean, predict_chol_cov = filtering.predict(
        state_1.mean, state_1.chol_cov, F, c, chol_Q
    )
    (update_mean, update_chol_cov), log_likelihood = filtering.update(
        predict_mean, predict_chol_cov, H, d, chol_R, observation
    )

    return LogDensityKalmanFilterState(
        mean=update_mean,
        chol_cov=update_chol_cov,
        log_likelihood=state_1.log_likelihood + log_likelihood,
        model_inputs=state_2.model_inputs,
    )


def smoother_prepare(
    filter_state: LogDensityKalmanFilterState,
    get_dynamics_log_density: GetDynamicsLogDensity,
    model_inputs: ArrayTreeLike | None = None,
    key: KeyArray | None = None,
) -> KalmanSmootherState:
    """
    Prepare a state for an exact Kalman smoother step.

    Args:
        filter_state: State generated by the log density Kalman filter at the previous
            time point.
        get_dynamics_log_density: Function to get dynamics log density log p(x_t+1 | x_t)
            and linearization points (for the previous and current time points)
        model_inputs: Model inputs at the next time point.
            Optional, if None then filter_state.model_inputs are used.
        key: JAX random key - not used.

    Returns:
        Prepared state for the Kalman smoother.
    """
    if model_inputs is None:
        model_inputs = filter_state.model_inputs
    else:
        model_inputs = tree.map(lambda x: jnp.asarray(x), model_inputs)

    filter_mean = filter_state.mean
    filter_chol_cov = filter_state.chol_cov

    log_dynamics_density, linearization_point_prev, linearization_point_curr = (
        get_dynamics_log_density(filter_state, model_inputs)
    )  ###### Might want to allow this to see filter_state_prev as well for linearization purposes

    F, c, chol_Q = linearize_log_density(
        log_dynamics_density, linearization_point_prev, linearization_point_curr
    )

    state = smoothing.associative_params_single(
        filter_mean, filter_chol_cov, F, c, chol_Q
    )
    return KalmanSmootherState(elem=state, gain=state.E, model_inputs=model_inputs)


def convert_filter_to_smoother_state(
    filter_state: LogDensityKalmanFilterState,
    model_inputs: ArrayTreeLike | None = None,
    key: KeyArray | None = None,
) -> KalmanSmootherState:
    """
    Convert the filter state to a smoother state.

    Useful for the final filter state which is equivalent to the final smoother state.

    Args:
        filter_state: Filter state.
        model_inputs: Model inputs at the final time point.
            Optional, if None then filter_state.model_inputs are used.
        key: JAX random key - not used.

    Returns:
        Smoother state, same data as filter state just different structure.
    """
    if model_inputs is None:
        model_inputs = filter_state.model_inputs
    else:
        model_inputs = tree.map(lambda x: jnp.asarray(x), model_inputs)

    return _convert_filter_to_smoother_state(
        filter_state.mean, filter_state.chol_cov, model_inputs
    )
