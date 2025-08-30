from functools import partial
from typing import NamedTuple, Protocol

from jax import numpy as jnp
from jax import tree

from cuthbert.gaussian.kalman import (
    GetInitParams,
    KalmanFilterState,
    KalmanSmootherState,
    convert_filter_to_smoother_state,
    smoother_combine,
)
from cuthbert.inference import Filter, Smoother
from cuthbertlib.kalman import filtering, smoothing
from cuthbertlib.linearize import linearize_log_density, linearize_taylor
from cuthbertlib.types import LogDensity, LogConditionalDensity
from cuthbertlib.types import (
    Array,
    ArrayLike,
    ArrayTree,
    ArrayTreeLike,
    KeyArray,
    ScalarArray,
)

LogPotential = LogDensity


class LogDensityKalmanFilterState(NamedTuple):
    mean: Array | None
    chol_cov: Array | None
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
        """Get the dynamics log density and inearization points
        (for the previous and current time points)"""
        ...


class GetObservationLogDensity(Protocol):
    def __call__(
        self, state: LogDensityKalmanFilterState | None, model_inputs: ArrayTreeLike
    ) -> tuple[LogConditionalDensity, Array, Array]:
        """Extract observation log density, linearization point and observation.
        At first time point, state is None, otherwise it is predicted state."""
        ...


class GetLogPotential(Protocol):
    def __call__(
        self, state: LogDensityKalmanFilterState | None, model_inputs: ArrayTreeLike
    ) -> tuple[LogPotential, Array]:
        """Extract log potential and linearization point.
        At first time point, state is None, otherwise it is predicted state."""
        ...


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
        get_init_log_density: Function that returns a log density and linearization point.
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
    key: KeyArray | None = None,
) -> LogDensityKalmanFilterState:
    """
    Prepare a state for a log density Kalman filter step - just passes through model inputs.

    Args:
        model_inputs: Model inputs.
        key: JAX random key - not used.

    Returns:
        Prepared state for extended Kalman filter.
    """
    model_inputs = tree.map(lambda x: jnp.asarray(x), model_inputs)
    return LogDensityKalmanFilterState(
        mean=None,
        chol_cov=None,
        log_likelihood=jnp.array(0.0),
        model_inputs=model_inputs,
    )


# def filter_combine(
#     state_1: LogDensityKalmanFilterState,
#     state_2: LogDensityKalmanFilterState,
#     dynamics_log_density: DynamicsLogDensity,
#     observation_func: GetObservationLogDensityAndObservation | GetLogPotential,
# ) -> LogDensityKalmanFilterState:
#     """
#     Combine filter state from previous time point with state prepared
#     with latest model inputs.

#     Applies extended Kalman predict + filter update in covariance square root form.
#     Not suitable for associative scan.

#     Args:
#         state_1: State from previous time step.
#         state_2: State prepared (only access model_inputs attribute).
#         dynamics_log_density: Function computing dynamics log density
#             log p(x | x_prev, model_inputs).
#         observation_func: Either
#             - Function that takes model_inputs and returns an observation log density
#                 function log p(y | x, model_inputs) and an observation.
#             - Function that takes model_inputs and returns a log potential function
#                 log G(x_prev, x).

#     Returns:
#         Predicted and updated extended Kalman filter state.
#             Contains mean, chol_cov (generalised Cholesky factor of covariance)
#             and log_likelihood.
#     """
#     if state_1.mean is None or state_1.chol_cov is None:
#         raise ValueError("State from previous time step must have mean and chol_cov.")

#     linearization_point = state_1.mean

#     F, c, chol_Q = linearize_log_density(
#         lambda x_prev, x: dynamics_log_density(x_prev, x, state_2.model_inputs),
#         state_1.mean,
#         state_1.mean,  # Should these both be state_1.mean?
#     )

#     # observation_output = observation_func(state_2.model_inputs)
#     # if isinstance(observation_output, tuple):
#     #     observation_log_density_func, observation = observation_output
#     #     H, d, chol_R = linearize_log_density(
#     #         observation_log_density_func, m0, observation, has_aux=True
#     #     )
#     # else:
#     #     d, chol_R = linearize_taylor(lambda x: observation_output(None, x), m0)
#     #     # dummy mat and observation as potential is unconditional
#     #     # Note the minus sign as linear potential is -0.5 (x - d)^T (R R^T)^{-1} (x - d)
#     #     # and kalman expects -0.5 (y - H @ x - d)^T (R R^T)^{-1} (y - H @ x - d)
#     #     H = -jnp.eye(d.shape[0])
#     #     observation = jnp.zeros_like(d)

#     # predict_mean, predict_chol_cov = filtering.predict(
#     #     state_1.mean, state_1.chol_cov, F, c, chol_Q
#     # )
#     # (update_mean, update_chol_cov), log_likelihood = filtering.update(
#     #     predict_mean, predict_chol_cov, H, d, chol_R, y
#     # )

#     # return ExtendedKalmanFilterState(
#     #     mean=update_mean,
#     #     chol_cov=update_chol_cov,
#     #     log_likelihood=state_1.log_likelihood + log_likelihood,
#     #     model_inputs=state_2.model_inputs,
#     # )
