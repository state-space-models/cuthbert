from functools import partial
from typing import NamedTuple, Protocol, runtime_checkable

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
from cuthbertlib.types import (
    Array,
    ArrayLike,
    ArrayTree,
    ArrayTreeLike,
    KeyArray,
    ScalarArray,
)


@runtime_checkable
class InitLogDensity(Protocol):
    def __call__(self, x: ArrayLike, model_inputs: ArrayTreeLike) -> ScalarArray:
        """Compute the log density of the initial distribution."""
        ...


@runtime_checkable
class DynamicsLogDensity(Protocol):
    def __call__(
        self, x_prev: ArrayLike, x: ArrayLike, model_inputs: ArrayTreeLike
    ) -> ScalarArray:
        """Compute the log density of the dynamics."""
        ...


@runtime_checkable
class ObservationLogDensity(Protocol):
    def __call__(self, x: ArrayLike, y: ArrayLike) -> ScalarArray:
        """Compute the log density of the observation."""
        ...


class GetObservationLogDensityAndObservation(Protocol):
    def __call__(
        self, model_inputs: ArrayTreeLike
    ) -> tuple[ObservationLogDensity, Array]:
        """Extract observation log density and observation from model inputs."""
        ...


@runtime_checkable
class LogPotential(Protocol):
    def __call__(self, x_prev: ArrayLike | None, x: ArrayLike) -> ScalarArray:
        """Compute the log potential."""
        ...


class GetLogPotential(Protocol):
    def __call__(self, model_inputs: ArrayTreeLike) -> LogPotential:
        """Extract log potential from model inputs."""
        ...


class LogDensityKalmanFilterState(NamedTuple):
    mean: Array | None
    chol_cov: Array | None
    log_likelihood: Array
    model_inputs: ArrayTree


# def process_observation(
#     observation_func: GetObservationLogDensityAndObservation | GetLogPotential,
#     model_inputs: ArrayTreeLike,
#     m0: ArrayLike,
# ) -> tuple[Array, Array, Array, Array]:
#     """Process observation for log density Kalman filter."""
#     observation_output = observation_func(model_inputs)
#     if isinstance(observation_output, tuple):
#         observation_log_density_func, observation = observation_output
#         H, d, chol_R = linearize_log_density(
#             observation_log_density_func, m0, observation, has_aux=True
#         )
#     else:
#         d, chol_R = linearize_taylor(lambda x: observation_output(None, x), m0)
#         # dummy mat and observation as potential is unconditional
#         # Note the minus sign as linear potential is -0.5 (x - d)^T (R R^T)^{-1} (x - d)
#         # and kalman expects -0.5 (y - H @ x - d)^T (R R^T)^{-1} (y - H @ x - d)
#         H = -jnp.eye(d.shape[0])
#         observation = jnp.zeros_like(d)

#     return H, d, chol_R, observation


def init_prepare(
    model_inputs: ArrayTreeLike,
    init_log_density: InitLogDensity,
    observation_func: GetObservationLogDensityAndObservation | GetLogPotential,
    init_x: ArrayLike,
    key: KeyArray | None = None,
) -> LogDensityKalmanFilterState:
    """
    Prepare the initial state for the log density Kalman filter.

    Args:
        model_inputs: Model inputs.
        init_log_density: Function to compute the log density of the initial
            distribution log p(x_0 | model_inputs).
        observation_func: Either
            - Function that takes model_inputs and returns an observation log density
                function log p(y_0 | x_0, model_inputs) and an observation.
            - Function that takes model_inputs and returns a log potential function
                log G(None, x_0).
        init_x: Initial point to linearize around.
        key: JAX random key - not used.

    Returns:
        State for the log density Kalman filter.
            Contains mean, chol_cov (generalised Cholesky factor of covariance)
            and log_likelihood.
    """
    model_inputs = tree.map(lambda x: jnp.asarray(x), model_inputs)

    _, m0, chol_P0 = linearize_log_density(
        lambda _, x: init_log_density(x, model_inputs), init_x, init_x
    )

    observation_output = observation_func(model_inputs)
    if isinstance(observation_output, tuple):
        observation_log_density_func, observation = observation_output
        H, d, chol_R = linearize_log_density(
            observation_log_density_func, m0, observation, has_aux=True
        )
    else:
        d, chol_R = linearize_taylor(lambda x: observation_output(None, x), m0)
        # dummy mat and observation as potential is unconditional
        # Note the minus sign as linear potential is -0.5 (x - d)^T (R R^T)^{-1} (x - d)
        # and kalman expects -0.5 (y - H @ x - d)^T (R R^T)^{-1} (y - H @ x - d)
        H = -jnp.eye(d.shape[0])
        observation = jnp.zeros_like(d)

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


def filter_combine(
    state_1: LogDensityKalmanFilterState,
    state_2: LogDensityKalmanFilterState,
    dynamics_log_density: DynamicsLogDensity,
    observation_func: GetObservationLogDensityAndObservation | GetLogPotential,
) -> LogDensityKalmanFilterState:
    """
    Combine filter state from previous time point with state prepared
    with latest model inputs.

    Applies extended Kalman predict + filter update in covariance square root form.
    Not suitable for associative scan.

    Args:
        state_1: State from previous time step.
        state_2: State prepared (only access model_inputs attribute).
        dynamics_log_density: Function computing dynamics log density
            log p(x | x_prev, model_inputs).
        observation_func: Either
            - Function that takes model_inputs and returns an observation log density
                function log p(y | x, model_inputs) and an observation.
            - Function that takes model_inputs and returns a log potential function
                log G(x_prev, x).

    Returns:
        Predicted and updated extended Kalman filter state.
            Contains mean, chol_cov (generalised Cholesky factor of covariance)
            and log_likelihood.
    """
    if state_1.mean is None or state_1.chol_cov is None:
        raise ValueError("State from previous time step must have mean and chol_cov.")

    linearization_point = state_1.mean

    F, c, chol_Q = linearize_log_density(
        lambda x_prev, x: dynamics_log_density(x_prev, x, state_2.model_inputs),
        state_1.mean,
        state_1.mean,  # Should these both be state_1.mean?
    )

    # observation_output = observation_func(state_2.model_inputs)
    # if isinstance(observation_output, tuple):
    #     observation_log_density_func, observation = observation_output
    #     H, d, chol_R = linearize_log_density(
    #         observation_log_density_func, m0, observation, has_aux=True
    #     )
    # else:
    #     d, chol_R = linearize_taylor(lambda x: observation_output(None, x), m0)
    #     # dummy mat and observation as potential is unconditional
    #     # Note the minus sign as linear potential is -0.5 (x - d)^T (R R^T)^{-1} (x - d)
    #     # and kalman expects -0.5 (y - H @ x - d)^T (R R^T)^{-1} (y - H @ x - d)
    #     H = -jnp.eye(d.shape[0])
    #     observation = jnp.zeros_like(d)

    # predict_mean, predict_chol_cov = filtering.predict(
    #     state_1.mean, state_1.chol_cov, F, c, chol_Q
    # )
    # (update_mean, update_chol_cov), log_likelihood = filtering.update(
    #     predict_mean, predict_chol_cov, H, d, chol_R, y
    # )

    # return ExtendedKalmanFilterState(
    #     mean=update_mean,
    #     chol_cov=update_chol_cov,
    #     log_likelihood=state_1.log_likelihood + log_likelihood,
    #     model_inputs=state_2.model_inputs,
    # )
