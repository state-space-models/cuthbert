from jax import eval_shape, tree
from jax import numpy as jnp

from cuthbert.gaussian.moments.types import GetDynamicsMoments, GetObservationMoments
from cuthbert.gaussian.types import GetInitParams, LinearizedKalmanFilterState
from cuthbert.gaussian.utils import linearized_kalman_filter_state_dummy_elem
from cuthbert.utils import dummy_tree_like
from cuthbertlib.kalman import filtering
from cuthbertlib.linearize import linearize_moments
from cuthbertlib.types import ArrayTreeLike, KeyArray


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

    mean_and_chol_cov_func, linearization_point, y = get_observation_params(
        prior_state, model_inputs
    )

    H, d, chol_R = linearize_moments(mean_and_chol_cov_func, linearization_point)
    (m, chol_P), ell = filtering.update(m0, chol_P0, H, d, chol_R, y)
    return linearized_kalman_filter_state_dummy_elem(
        mean=m,
        chol_cov=chol_P,
        log_normalizing_constant=ell,
        model_inputs=model_inputs,
        mean_prev=dummy_tree_like(m),
    )


def filter_prepare(
    model_inputs: ArrayTreeLike,
    get_init_params: GetInitParams,
    key: KeyArray | None = None,
) -> LinearizedKalmanFilterState:
    """
    Prepare a state for a linearized moments Kalman filter step,
    just passes through model inputs.

    Args:
        model_inputs: Model inputs.
        get_init_params: Function to get m0, chol_P0 from model inputs.
            Only used to infer shape of mean and chol_cov.
        key: JAX random key - not used.

    Returns:
        Prepared state for linearized moments Kalman filter.
    """
    model_inputs = tree.map(lambda x: jnp.asarray(x), model_inputs)
    dummy_mean_struct = eval_shape(lambda mi: get_init_params(mi)[0], model_inputs)
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
    get_dynamics_params: GetDynamicsMoments,
    get_observation_params: GetObservationMoments,
) -> LinearizedKalmanFilterState:
    """
    Combine filter state from previous time point with state prepared
    with latest model inputs.

    Applies linearized moments Kalman predict + filter update in covariance square
    root form.
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
            and log_normalizing_constant.
    """

    dynamics_mean_and_chol_cov_func, dynamics_linearization_point = get_dynamics_params(
        state_1, state_2.model_inputs
    )
    F, c, chol_Q = linearize_moments(
        dynamics_mean_and_chol_cov_func, dynamics_linearization_point
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

    observation_mean_and_chol_cov_func, observation_linearization_point, y = (
        get_observation_params(predict_state, state_2.model_inputs)
    )
    H, d, chol_R = linearize_moments(
        observation_mean_and_chol_cov_func, observation_linearization_point
    )

    (update_mean, update_chol_cov), log_normalizing_constant = filtering.update(
        predict_mean, predict_chol_cov, H, d, chol_R, y
    )

    return linearized_kalman_filter_state_dummy_elem(
        mean=update_mean,
        chol_cov=update_chol_cov,
        log_normalizing_constant=state_1.log_normalizing_constant
        + log_normalizing_constant,
        model_inputs=state_2.model_inputs,
        mean_prev=state_1.mean,
    )
