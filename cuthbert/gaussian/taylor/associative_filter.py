"""Associative linearized Kalman filter that uses automatic differentiation to extract
conditonally Gaussian parameters from log densities of the dynamics and observation
distributions. This differs from gaussian/moments which requires mean and chol_cov
functions as input rather than log densities.

I.e. we approximate conditional log densities as

log p(y | x) ≈ N(y | H x + d, L L^T)

and log potentials as

log G(x) ≈ N(x | m, L L^T)

where L is the cholesky factor of the covariance matrix.

See `cuthbertlib.linearize` for more details.

This variant assumes linearization points are predefined or can be extracted from model
inputs, therefore is suitable for associative scan.
"""

from jax import tree
from jax import numpy as jnp

from cuthbert.gaussian.types import (
    AssociativeLinearizedKalmanFilterState,
)
from cuthbert.gaussian.taylor.types import (
    GetInitLogDensity,
    AssociativeGetDynamicsLogDensity,
    AssociativeGetObservationFunc,
)
from cuthbertlib.kalman import filtering
from cuthbertlib.linearize import linearize_log_density
from cuthbertlib.types import (
    ArrayTreeLike,
    KeyArray,
)
from cuthbert.gaussian.taylor.non_associative_filter import process_observation
from cuthbert.utils import dummy_tree_like


def init_prepare(
    model_inputs: ArrayTreeLike,
    get_init_log_density: GetInitLogDensity,
    get_observation_func: AssociativeGetObservationFunc,
    key: KeyArray | None = None,
) -> AssociativeLinearizedKalmanFilterState:
    """
    Prepare the initial state for the linearized Taylor Kalman filter.

    Args:
        model_inputs: Model inputs.
        get_init_log_density: Function that returns log density log p(x_0)
            and linearization point.
        get_observation_func: Function that returns either
            - An observation log density
                function log p(y_0 | x_0) as well as points x_0 and y_0
                to linearize around.
            - A log potential function log G(x_0) and a linearization point x_0.
        key: JAX random key - not used.

    Returns:
        State for the linearized Taylor Kalman filter.
            Contains mean, chol_cov (generalised Cholesky factor of covariance)
            and log_likelihood.
    """
    model_inputs = tree.map(lambda x: jnp.asarray(x), model_inputs)
    init_log_density, linearization_point = get_init_log_density(model_inputs)

    _, m0, chol_P0 = linearize_log_density(
        lambda _, x: init_log_density(x), linearization_point, linearization_point
    )

    observation_output = get_observation_func(model_inputs)
    H, d, chol_R, observation = process_observation(observation_output)

    (m, chol_P), ell = filtering.update(m0, chol_P0, H, d, chol_R, observation)

    elem = filtering.FilterScanElement(
        A=jnp.zeros_like(chol_P),
        b=m,
        U=chol_P,
        eta=jnp.zeros_like(m),
        Z=jnp.zeros_like(chol_P),
        ell=ell,
    )

    return AssociativeLinearizedKalmanFilterState(
        elem=elem,
        model_inputs=model_inputs,
        mean_prev=dummy_tree_like(m),
    )


def filter_prepare(
    model_inputs: ArrayTreeLike,
    get_dynamics_log_density: AssociativeGetDynamicsLogDensity,
    get_observation_func: AssociativeGetObservationFunc,
    key: KeyArray | None = None,
) -> AssociativeLinearizedKalmanFilterState:
    """
    Prepare a state for a linearized Taylor Kalman filter step,
    just passes through model inputs.

    Args:
        model_inputs: Model inputs.
        get_dynamics_log_density: Function to get dynamics log density log p(x_t+1 | x_t)
            and linearization points (for the previous and current time points)
            Only has `model_inputs` as input.
        get_observation_func: Function to get observation function (either conditional
            log density or log potential), linearization point and optional observation
            (not required for log potential functions).
            Only has `model_inputs` as input.
        key: JAX random key - not used.

    Returns:
        Prepared state for linearized Taylor Kalman filter.
    """
    model_inputs = tree.map(lambda x: jnp.asarray(x), model_inputs)

    log_dynamics_density, linearization_point_prev, linearization_point_curr = (
        get_dynamics_log_density(model_inputs)
    )

    F, c, chol_Q = linearize_log_density(
        log_dynamics_density, linearization_point_prev, linearization_point_curr
    )

    observation_output = get_observation_func(model_inputs)
    H, d, chol_R, observation = process_observation(observation_output)

    elem = filtering.associative_params_single(F, c, chol_Q, H, d, chol_R, observation)

    return AssociativeLinearizedKalmanFilterState(
        elem=elem,
        model_inputs=model_inputs,
        mean_prev=dummy_tree_like(c),
    )


def filter_combine(
    state_1: AssociativeLinearizedKalmanFilterState,
    state_2: AssociativeLinearizedKalmanFilterState,
) -> AssociativeLinearizedKalmanFilterState:
    """
    Combine filter state from previous time point with state prepared
    with latest model inputs.

    Applies standard associative Kalman filtering operator since dynamics and observation
    parameters are extracted in filter_prepare.
    Suitable for associative scan.

    Args:
        state_1: State from previous time step.
        state_2: State prepared (only access model_inputs attribute).

    Returns:
        Predicted and updated linearized Taylor Kalman filter state.
            Contains mean, chol_cov (generalised Cholesky factor of covariance)
            and log_likelihood.
    """
    combined_elem = filtering.filtering_operator(
        state_1.elem,
        state_2.elem,
    )
    return AssociativeLinearizedKalmanFilterState(
        elem=combined_elem,
        model_inputs=state_2.model_inputs,
        mean_prev=state_1.mean,
    )
