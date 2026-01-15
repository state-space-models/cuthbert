"""Implements the associative linearized Taylor Kalman filter."""

from jax import eval_shape, tree
from jax import numpy as jnp

from cuthbert.gaussian.taylor.non_associative_filter import process_observation
from cuthbert.gaussian.taylor.types import (
    GetDynamicsLogDensity,
    GetInitLogDensity,
    GetObservationFunc,
)
from cuthbert.gaussian.types import (
    LinearizedKalmanFilterState,
)
from cuthbert.utils import dummy_tree_like
from cuthbertlib.kalman import filtering
from cuthbertlib.linearize import linearize_log_density
from cuthbertlib.types import (
    ArrayTreeLike,
    KeyArray,
)


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

    prior_state = LinearizedKalmanFilterState(
        elem=filtering.FilterScanElement(
            A=jnp.zeros_like(chol_P0),
            b=m0,
            U=chol_P0,
            eta=jnp.zeros_like(m0),
            Z=jnp.zeros_like(chol_P0),
            ell=jnp.array(0.0),
        ),
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

    elem = filtering.FilterScanElement(
        A=jnp.zeros_like(chol_P),
        b=m,
        U=chol_P,
        eta=jnp.zeros_like(m),
        Z=jnp.zeros_like(chol_P),
        ell=ell,
    )

    return LinearizedKalmanFilterState(
        elem=elem,
        model_inputs=model_inputs,
        mean_prev=dummy_tree_like(m),
    )


def filter_prepare(
    model_inputs: ArrayTreeLike,
    get_init_log_density: GetInitLogDensity,
    get_dynamics_log_density: GetDynamicsLogDensity,
    get_observation_func: GetObservationFunc,
    rtol: float | None = None,
    ignore_nan_dims: bool = False,
    key: KeyArray | None = None,
) -> LinearizedKalmanFilterState:
    """Prepare a state for a linearized Taylor Kalman filter step.

    `associative_scan` is supported but only accurate when `state` is ignored
    in `get_dynamics_log_density` and `get_observation_func`.

    Args:
        model_inputs: Model inputs.
        get_init_log_density: Function that returns log density log p(x_0)
            and linearization point. Only used to infer shape of mean and chol_cov.
        get_dynamics_log_density: Function to get dynamics log density log p(x_t+1 | x_t)
            and linearization points (for the previous and current time points)
            `associative_scan` only supported when `state` is ignored.
        get_observation_func: Function to get observation function (either conditional
            log density or log potential), linearization point and optional observation
            (not required for log potential functions).
            `associative_scan` only supported when `state` is ignored.
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
        Prepared state for linearized Taylor Kalman filter.
    """
    model_inputs = tree.map(lambda x: jnp.asarray(x), model_inputs)
    dummy_mean_struct = eval_shape(lambda mi: get_init_log_density(mi)[1], model_inputs)
    dummy_mean = dummy_tree_like(dummy_mean_struct)
    dummy_chol_cov = dummy_tree_like(jnp.cov(dummy_mean[..., None]))

    dummy_state = LinearizedKalmanFilterState(
        elem=filtering.FilterScanElement(
            A=dummy_chol_cov,
            b=dummy_mean,
            U=dummy_chol_cov,
            eta=dummy_mean,
            Z=dummy_chol_cov,
            ell=jnp.array(0.0),
        ),
        model_inputs=model_inputs,
        mean_prev=dummy_mean,
    )

    log_dynamics_density, linearization_point_prev, linearization_point_curr = (
        get_dynamics_log_density(dummy_state, model_inputs)
    )

    F, c, chol_Q = linearize_log_density(
        log_dynamics_density,
        linearization_point_prev,
        linearization_point_curr,
        rtol=rtol,
        ignore_nan_dims=ignore_nan_dims,
    )

    observation_output = get_observation_func(dummy_state, model_inputs)
    H, d, chol_R, observation = process_observation(
        observation_output,
        rtol=rtol,
        ignore_nan_dims=ignore_nan_dims,
    )

    elem = filtering.associative_params_single(F, c, chol_Q, H, d, chol_R, observation)

    return LinearizedKalmanFilterState(
        elem=elem,
        model_inputs=model_inputs,
        mean_prev=dummy_mean,
    )


def filter_combine(
    state_1: LinearizedKalmanFilterState,
    state_2: LinearizedKalmanFilterState,
) -> LinearizedKalmanFilterState:
    """Combine previous filter state with state prepared with latest model inputs.

    `associative_scan` is supported but only accurate when `state` is ignored
    in `get_dynamics_log_density` and `get_observation_func`.

    Applies standard associative Kalman filtering operator since dynamics and observation
    parameters are extracted in filter_prepare.

    Args:
        state_1: State from previous time step.
        state_2: State prepared (only access model_inputs attribute).

    Returns:
        Predicted and updated linearized Taylor Kalman filter state.
            Contains mean, chol_cov (generalised Cholesky factor of covariance)
            and log_normalizing_constant.
    """
    combined_elem = filtering.filtering_operator(
        state_1.elem,
        state_2.elem,
    )
    return LinearizedKalmanFilterState(
        elem=combined_elem,
        model_inputs=state_2.model_inputs,
        mean_prev=state_1.mean,
    )
