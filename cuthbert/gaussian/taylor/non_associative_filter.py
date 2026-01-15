"""Implements the non-associative linearized Taylor Kalman filter."""

from jax import eval_shape, tree
from jax import numpy as jnp

from cuthbert.gaussian.taylor.types import (
    GetDynamicsLogDensity,
    GetInitLogDensity,
    GetObservationFunc,
    LogConditionalDensity,
    LogPotential,
)
from cuthbert.gaussian.types import LinearizedKalmanFilterState
from cuthbert.gaussian.utils import linearized_kalman_filter_state_dummy_elem
from cuthbert.utils import dummy_tree_like
from cuthbertlib.kalman import filtering
from cuthbertlib.linearize import linearize_log_density, linearize_taylor
from cuthbertlib.types import (
    Array,
    ArrayTreeLike,
    KeyArray,
)


def process_observation(
    observation_output: tuple[LogConditionalDensity, Array, Array]
    | tuple[LogPotential, Array],
    rtol: float | None = None,
    ignore_nan_dims: bool = False,
) -> tuple[Array, Array, Array, Array]:
    """Process observation for linearized Taylor Kalman filter."""
    if len(observation_output) == 3:
        observation_cond_log_density, linearization_point, observation = (
            observation_output
        )
        H, d, chol_R = linearize_log_density(
            observation_cond_log_density,
            linearization_point,
            observation,
            rtol=rtol,
            ignore_nan_dims=ignore_nan_dims,
        )
    else:
        observation_log_potential, linearization_point = observation_output
        d, chol_R = linearize_taylor(
            observation_log_potential,
            linearization_point,
            rtol=rtol,
            ignore_nan_dims=ignore_nan_dims,
        )
        # dummy mat and observation as potential is unconditional
        # Note the minus sign as linear potential is -0.5 (x - d)^T (R R^T)^{-1} (x - d)
        # and kalman expects -0.5 (y - H @ x - d)^T (R R^T)^{-1} (y - H @ x - d)
        H = -jnp.eye(d.shape[0])
        observation = jnp.where(
            jnp.isnan(jnp.diag(chol_R)) * ignore_nan_dims, jnp.nan, 0.0
        )  # Tell the cuthbertlib.kalman to skip these dimensions
    return H, d, chol_R, observation


def init_prepare(
    model_inputs: ArrayTreeLike,
    get_init_log_density: GetInitLogDensity,
    get_observation_func: GetObservationFunc,
    rtol: float | None = None,
    ignore_nan_dims: bool = False,
    key: KeyArray | None = None,
) -> LinearizedKalmanFilterState:
    """Prepare the initial state for the linearized Taylor Kalman filter.

    Args:
        model_inputs: Model inputs.
        get_init_log_density: Function that returns log density log p(x_0)
            and linearization point.
        get_observation_func: Function that returns either
            - An observation log density
                function log p(y_0 | x_0) as well as points x_0 and y_0
                to linearize around.
            - A log potential function log G(x_0) and a linearization point x_0.
        rtol: The relative tolerance for the singular values of precision matrices
            when passed to `symmetric_inv_sqrt` during linearization.
            Cutoff for small singular values; singular values smaller than
            `rtol * largest_singular_value` are treated as zero.
            The default is determined based on the floating point precision of the dtype.
            See https://docs.jax.dev/en/latest/_autosummary/jax.numpy.linalg.pinv.html.
        ignore_nan_dims: Whether to treat dimensions with NaN on the diagonal of the
            precision matrices (found via linearization) as missing and ignore all rows
            and columns associated with them.
        key: JAX random key - not used.

    Returns:
        State for the linearized Taylor Kalman filter.
            Contains mean, chol_cov (generalised Cholesky factor of covariance)
            and log_normalizing_constant.
    """
    model_inputs = tree.map(lambda x: jnp.asarray(x), model_inputs)
    init_log_density, linearization_point = get_init_log_density(model_inputs)

    _, m0, chol_P0 = linearize_log_density(
        lambda _, x: init_log_density(x),
        linearization_point,
        linearization_point,
        rtol=rtol,
        ignore_nan_dims=ignore_nan_dims,
    )

    prior_state = linearized_kalman_filter_state_dummy_elem(
        mean=m0,
        chol_cov=chol_P0,
        log_normalizing_constant=jnp.array(0.0),
        model_inputs=model_inputs,
        mean_prev=dummy_tree_like(m0),
    )

    observation_output = get_observation_func(prior_state, model_inputs)

    H, d, chol_R, observation = process_observation(
        observation_output,
        rtol=rtol,
        ignore_nan_dims=ignore_nan_dims,
    )

    (m, chol_P), ell = filtering.update(m0, chol_P0, H, d, chol_R, observation)

    return linearized_kalman_filter_state_dummy_elem(
        mean=m,
        chol_cov=chol_P,
        log_normalizing_constant=ell,
        model_inputs=model_inputs,
        mean_prev=dummy_tree_like(m),
    )


def filter_prepare(
    model_inputs: ArrayTreeLike,
    get_init_log_density: GetInitLogDensity,
    key: KeyArray | None = None,
) -> LinearizedKalmanFilterState:
    """Prepare a state for a linearized Taylor Kalman filter step.

    Just passes through model inputs.

    Args:
        model_inputs: Model inputs.
        get_init_log_density: Function that returns log density log p(x_0)
            and linearization point. Only used to infer shape of mean and chol_cov.
        key: JAX random key - not used.

    Returns:
        Prepared state for linearized Taylor Kalman filter.
    """
    model_inputs = tree.map(lambda x: jnp.asarray(x), model_inputs)
    dummy_mean_struct = eval_shape(lambda mi: get_init_log_density(mi)[1], model_inputs)
    dummy_mean = dummy_tree_like(dummy_mean_struct)
    dummy_chol_cov = dummy_tree_like(jnp.cov(dummy_mean[..., None]))

    return linearized_kalman_filter_state_dummy_elem(
        mean=dummy_mean,
        chol_cov=dummy_chol_cov,
        log_normalizing_constant=jnp.array(0.0),
        model_inputs=model_inputs,
        mean_prev=dummy_mean,
    )


def filter_combine(
    state_1: LinearizedKalmanFilterState,
    state_2: LinearizedKalmanFilterState,
    get_dynamics_log_density: GetDynamicsLogDensity,
    get_observation_func: GetObservationFunc,
    rtol: float | None = None,
    ignore_nan_dims: bool = False,
) -> LinearizedKalmanFilterState:
    """Combine previous filter state with state prepared from latest model inputs.

    Applies linearized Taylor Kalman predict + filter update in covariance square
    root form.
    Not suitable for associative scan.

    Args:
        state_1: State from previous time step.
        state_2: State prepared (only access model_inputs attribute).
        get_dynamics_log_density: Function to get dynamics log density log p(x_t+1 | x_t)
            and linearization points (for the previous and current time points)
        get_observation_func: Function to get observation function (either conditional
            log density or log potential), linearization point and optional observation
            (not required for log potential functions).
        rtol: The relative tolerance for the singular values of precision matrices
            when passed to `symmetric_inv_sqrt` during linearization.
            Cutoff for small singular values; singular values smaller than
            `rtol * largest_singular_value` are treated as zero.
            The default is determined based on the floating point precision of the dtype.
            See https://docs.jax.dev/en/latest/_autosummary/jax.numpy.linalg.pinv.html.
        ignore_nan_dims: Whether to treat dimensions with NaN on the diagonal of the
            precision matrices (found via linearization) as missing and ignore all rows
            and columns associated with them.

    Returns:
        Predicted and updated linearized Taylor Kalman filter state.
            Contains mean, chol_cov (generalised Cholesky factor of covariance)
            and log_normalizing_constant.
    """
    log_dynamics_density, linearization_point_prev, linearization_point_curr = (
        get_dynamics_log_density(state_1, state_2.model_inputs)
    )

    F, c, chol_Q = linearize_log_density(
        log_dynamics_density,
        linearization_point_prev,
        linearization_point_curr,
        rtol=rtol,
        ignore_nan_dims=ignore_nan_dims,
    )

    predict_mean, predict_chol_cov = filtering.predict(
        state_1.mean, state_1.chol_cov, F, c, chol_Q
    )

    predict_state = linearized_kalman_filter_state_dummy_elem(
        mean=predict_mean,
        chol_cov=predict_chol_cov,
        log_normalizing_constant=state_1.log_normalizing_constant,
        model_inputs=state_2.model_inputs,
        mean_prev=state_1.mean,
    )

    observation_output = get_observation_func(predict_state, state_2.model_inputs)

    H, d, chol_R, observation = process_observation(
        observation_output,
        rtol=rtol,
        ignore_nan_dims=ignore_nan_dims,
    )

    (update_mean, update_chol_cov), log_normalizing_constant = filtering.update(
        predict_mean, predict_chol_cov, H, d, chol_R, observation
    )

    return linearized_kalman_filter_state_dummy_elem(
        mean=update_mean,
        chol_cov=update_chol_cov,
        log_normalizing_constant=state_1.log_normalizing_constant
        + log_normalizing_constant,
        model_inputs=state_2.model_inputs,
        mean_prev=state_1.mean,
    )
