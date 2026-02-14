"""Implements the associative linearized moments Kalman filter."""

from jax import eval_shape, tree
from jax import numpy as jnp

from cuthbert.gaussian.kalman import GetInitParams
from cuthbert.gaussian.moments.types import GetDynamicsMoments, GetObservationMoments
from cuthbert.gaussian.types import LinearizedKalmanFilterState
from cuthbert.utils import dummy_tree_like
from cuthbertlib.kalman import filtering
from cuthbertlib.linearize import linearize_moments
from cuthbertlib.types import ArrayTreeLike, KeyArray


def init_prepare(
    model_inputs: ArrayTreeLike,
    get_init_params: GetInitParams,
    key: KeyArray | None = None,
) -> LinearizedKalmanFilterState:
    """Prepare the initial state for the linearized moments Kalman filter.

    Args:
        model_inputs: Model inputs.
        get_init_params: Function to get m0, chol_P0 from model inputs.
        key: JAX random key - not used.

    Returns:
        State for the linearized moments Kalman filter.
            Contains mean, chol_cov (generalised Cholesky factor of covariance)
            and log_normalizing_constant.
    """
    model_inputs = tree.map(lambda x: jnp.asarray(x), model_inputs)
    m0, chol_P0 = get_init_params(model_inputs)

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

    return prior_state


def filter_prepare(
    model_inputs: ArrayTreeLike,
    get_init_params: GetInitParams,
    get_dynamics_params: GetDynamicsMoments,
    get_observation_params: GetObservationMoments,
    key: KeyArray | None = None,
) -> LinearizedKalmanFilterState:
    """Prepare a state for a linearized moments Kalman filter step.

    Just passes through model inputs.

    `associative_scan` is supported but only accurate when `state` is ignored
    in `get_dynamics_params` and `get_observation_params`.

    Args:
        model_inputs: Model inputs.
        get_init_params: Function to get m0, chol_P0 from model inputs.
            Only used to infer shape of mean and chol_cov.
        get_dynamics_params: Function to get dynamics conditional mean and
            (generalised) Cholesky covariance from linearization point and model inputs.
            `associative_scan` only supported when `state` is ignored.
        get_observation_params: Function to get observation conditional mean,
            (generalised) Cholesky covariance and observation from linearization point
            and model inputs.
            `associative_scan` only supported when `state` is ignored.
        key: JAX random key - not used.

    Returns:
        Prepared state for linearized moments Kalman filter.
    """
    model_inputs = tree.map(lambda x: jnp.asarray(x), model_inputs)
    dummy_mean_struct = eval_shape(lambda mi: get_init_params(mi)[0], model_inputs)
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

    dynamics_mean_and_chol_cov_func, dynamics_linearization_point = get_dynamics_params(
        dummy_state, model_inputs
    )
    F, c, chol_Q = linearize_moments(
        dynamics_mean_and_chol_cov_func, dynamics_linearization_point
    )

    observation_mean_and_chol_cov_func, observation_linearization_point, y = (
        get_observation_params(dummy_state, model_inputs)
    )
    H, d, chol_R = linearize_moments(
        observation_mean_and_chol_cov_func, observation_linearization_point
    )

    elem = filtering.associative_params_single(F, c, chol_Q, H, d, chol_R, y)

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
    in `get_dynamics_params` and `get_observation_params`.

    Applies standard associative Kalman filtering operator since dynamics and observation
    parameters are extracted in filter_prepare.

    Args:
        state_1: State from previous time step.
        state_2: State prepared (only access model_inputs attribute).

    Returns:
        Predicted and updated linearized moments Kalman filter state.
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
