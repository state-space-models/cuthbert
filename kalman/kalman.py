from typing import NamedTuple
from jax import Array, numpy as jnp, vmap, tree
from jax.typing import ArrayLike
from jax.lax import associative_scan

from kalman.protocols import (
    ArrayTree,
    ArrayTreeLike,
    LinearGaussianInit,
    LinearGaussianDynamics,
    LinearGaussianObservation,
)


class KalmanState(NamedTuple):
    """Gaussian state with mean, covariance and potentially temporal cross-covariance.

    May or may not have a temporal axis as first axis.

    Attributes:
        mean: Mean of the Gaussian state.
        cov: Covariance of the Gaussian state.
        cross_cov: Cross-covariance between times k and k + 1.
            Only applies if temporal axis present. None otherwise.
    """

    mean: Array
    cov: Array
    cross_cov: Array | None = None
    # TODO: Include log_norm_constant???


def init(
    inputs: ArrayTreeLike,
    init_params: LinearGaussianInit,
) -> KalmanState:
    """Generate Kalman state for initial time.

    Args:
        inputs: The input PyTree (with ArrayLike leaves) at the initial state.
        init_params: The callable to generate the initial Gaussian state.

    Returns:
        Initial state with Array mean and covariance.
    """
    init_mean, init_cov = init_params(inputs)
    return KalmanState(init_mean, init_cov)


def predict(
    state: KalmanState,
    inputs: ArrayTreeLike,
    dynamics_params: LinearGaussianDynamics,
) -> KalmanState:
    """Propagate the next state at time t given the previous state at time t_prev.

    Args:
        state: The previous Gaussian state.
        inputs: The input PyTree (with ArrayLike leaves) to the dynamics.
        dynamics_params: The callable to generate the dynamics parameters.

    Returns:
        Propagated Gaussian state with Array mean and covariance.
    """
    shift, mat_x, cov = dynamics_params(inputs)
    new_mean = shift + mat_x @ state.mean
    new_cov = mat_x @ state.cov @ mat_x.T + cov
    return KalmanState(new_mean, new_cov)


def update(
    state: KalmanState,
    inputs: ArrayTreeLike,
    observation: ArrayLike,
    observation_params: LinearGaussianObservation,
) -> KalmanState:
    """Update step for a linear Gaussian state-space model.

    Args:
        state: The previous Gaussian state.
        inputs: The input PyTree (with ArrayLike leaves) to the observation
                distribution.
        observation: The ArrayLike observation at the current state.
        observation_params: The callable to generate the observation parameters.

    Returns:
        The filtered Gaussian state at time t.
    """
    obs_shift, obs_mat_x, obs_cov = observation_params(inputs)

    obs_mean = obs_shift + obs_mat_x @ state.mean
    v = observation - obs_mean
    S = obs_mat_x @ state.cov @ obs_mat_x.T + obs_cov
    S_inv = jnp.linalg.inv(S)
    K = state.cov @ obs_mat_x.T @ S_inv

    new_mean = state.mean + K @ v
    new_cov = state.cov - K @ S @ K.T
    return KalmanState(new_mean, new_cov)


def offline_filter(
    inputs: ArrayTreeLike,
    observations: ArrayLike,
    init_params: LinearGaussianInit,
    dynamics_params: LinearGaussianDynamics,
    observation_params: LinearGaussianObservation,
) -> KalmanState:
    """Offline filter for linear Gaussian models via associative scan.

    Args:
        inputs: Inputs to init, dynamics and observations,
            ArrayTreeLike with all leaves Arrays each with
            first axis = temporal axis with length K + 1.
        observations: ArrayLike, first axis = temporal axis with length K.
        init_params: The callable to generate the initial Gaussian parameters.
        dynamics_params: The callable to generate the dynamics parameters.
        observation_params: The callable to generate the observation parameters.

    Returns:
        Filtered Gaussian states containing mean (length K + 1) and cov (length K + 1),
            cross_cov left as None.
    """
    # From https://arxiv.org/abs/1905.13002
    # Also code here https://colab.research.google.com/github/EEA-sensors/sequential-parallelization-examples/blob/main/python/temporal-parallelization-bayes-smoothers/parallel_kalman_jax.ipynb
    # TODO: Make more efficient with jax.scipy.linalg.cho_factor

    observations = jnp.asarray(observations)
    K = len(observations)

    def inputs_k(k: int) -> ArrayTree:
        return tree.map(lambda u: u[k], inputs)

    init_state = init(inputs_k(0), init_params)

    class FilterScanElement(NamedTuple):
        A: Array
        b: Array
        C: Array
        eta: Array
        J: Array

    def get_pre_filter_1() -> FilterScanElement:
        k = 1
        dynamics_shift, F, Q = dynamics_params(inputs_k(k))
        observation_shift, H, R = observation_params(inputs_k(k))

        m1minus = F @ init_state.mean + dynamics_shift
        P1minus = F @ init_state.cov @ F.T + Q
        S1 = H @ P1minus @ H.T + R
        S1inv = jnp.linalg.inv(S1)
        K1 = P1minus @ H.T @ S1inv
        A1 = jnp.zeros_like(F)
        b1 = m1minus + K1 @ (observations[k - 1] - H @ m1minus - observation_shift)
        C1 = P1minus - K1 @ S1 @ K1.T
        eta = (
            F.T
            @ H.T
            @ S1inv
            @ (observations[k - 1] - H @ dynamics_shift - observation_shift)
        )
        J = F.T @ H.T @ S1inv @ H @ F
        return FilterScanElement(A1, b1, C1, eta, J)

    def get_pre_filter(k) -> FilterScanElement:
        dynamics_shift, F, Q = dynamics_params(inputs_k(k))
        observation_shift, H, R = observation_params(inputs_k(k))

        S = H @ Q @ H.T + R
        Sinv = jnp.linalg.inv(S)
        K = Q @ H.T @ Sinv
        A = (jnp.eye(Q.shape[0]) - K @ H) @ F
        b = dynamics_shift + K @ (
            observations[k - 1] - H @ dynamics_shift - observation_shift
        )
        C = (jnp.eye(Q.shape[0]) - K @ H) @ Q

        eta = (
            F.T
            @ H.T
            @ Sinv
            @ (observations[k - 1] - H @ dynamics_shift - observation_shift)
        )
        J = F.T @ H.T @ Sinv @ H @ F
        return FilterScanElement(A, b, C, eta, J)

    pre_filter_1 = get_pre_filter_1()
    pre_filter_rest = vmap(get_pre_filter)(jnp.arange(2, K + 1))
    pre_filter = tree.map(
        lambda a, b: jnp.vstack([a[jnp.newaxis], b]),
        pre_filter_1,
        pre_filter_rest,
    )

    @vmap
    def filter_ascan_body(elem_i: FilterScanElement, elem_j: FilterScanElement):
        IpCJ_inv = jnp.linalg.inv(jnp.eye(elem_j.J.shape[0]) + elem_i.C @ elem_j.J)
        Aij = elem_j.A @ IpCJ_inv @ elem_i.A
        bij = elem_j.A @ IpCJ_inv @ (elem_i.b + elem_i.C @ elem_j.eta) + elem_j.b
        Cij = elem_j.A @ IpCJ_inv @ elem_i.C @ elem_j.A.T + elem_j.C

        IpJC_inv = jnp.linalg.inv(jnp.eye(elem_j.J.shape[0]) + elem_j.J @ elem_i.C)
        etaij = elem_i.A.T @ IpJC_inv @ (elem_j.eta - elem_j.J @ elem_i.b) + elem_i.eta
        Jij = elem_i.A.T @ IpJC_inv @ elem_j.J @ elem_i.A + elem_i.J
        return FilterScanElement(Aij, bij, Cij, etaij, Jij)

    filtered_elements = associative_scan(filter_ascan_body, pre_filter)

    means = jnp.vstack([init_state.mean[jnp.newaxis], filtered_elements.b])
    covs = jnp.vstack([init_state.cov[jnp.newaxis], filtered_elements.C])

    return KalmanState(mean=means, cov=covs)


def smoother(
    filter_states: KalmanState,
    inputs: ArrayTreeLike,
    dynamics_params: LinearGaussianDynamics,
) -> KalmanState:
    """Kalman smoother for linear Gaussian models via associative scan.

    Args:
        filter_states: Filtered states, first axis = temporal axis with length K + 1.
        inputs: Inputs, first axis = temporal axis with length K + 1.
        dynamics_params: The callable to generate the dynamics parameters.

    Returns:
        Smoothed Gaussian states
            containing mean (length K + 1), cov (length K + 1) and cross_cov (length K),
            where cross_cov is the covariances between times k and k + 1.
    """

    K = len(filter_states.mean) - 1

    def inputs_k(k: int) -> ArrayTree:
        return tree.map(lambda u: u[k], inputs)

    class SmootherScanElement(NamedTuple):
        E: Array
        g: Array
        L: Array

    def get_pre_smoother_final() -> SmootherScanElement:
        g = filter_states.mean[-1]
        L = filter_states.cov[-1]
        E = jnp.zeros_like(L)
        return SmootherScanElement(E, g, L)

    def get_pre_smoother(k) -> SmootherScanElement:
        shift, F, Q = dynamics_params(inputs_k(k + 1))

        filter_mean = filter_states.mean[k]
        filter_cov = filter_states.cov[k]

        S = F @ filter_cov @ F.T + Q
        Sinv = jnp.linalg.inv(S)
        E = filter_cov @ F.T @ Sinv
        g = filter_mean - E @ (F @ filter_mean + shift)
        # L = filter_cov - E @ F @ filter_cov
        L = filter_cov - E @ S @ E.T  # Different to https://arxiv.org/abs/1905.13002
        return SmootherScanElement(E, g, L)

    pre_smoother_rest = vmap(get_pre_smoother)(jnp.arange(K))
    pre_smoother_final = get_pre_smoother_final()
    pre_smoother = tree.map(
        lambda a, b: jnp.vstack([a, b[jnp.newaxis]]),
        pre_smoother_rest,
        pre_smoother_final,
    )

    @vmap
    def smoother_ascan_body(elem_i: SmootherScanElement, elem_j: SmootherScanElement):
        # Eij = elem_i.E @ elem_j.E
        # gij = elem_i.E @ elem_j.g + elem_i.g
        # Lij = elem_i.E @ elem_j.L @ elem_i.E.T + elem_i.L
        Eij = elem_j.E @ elem_i.E  # Different to https://arxiv.org/abs/1905.13002
        gij = elem_j.E @ elem_i.g + elem_j.g
        Lij = elem_j.E @ elem_i.L @ elem_j.E.T + elem_j.L
        return SmootherScanElement(Eij, gij, Lij)

    smoother_elements = associative_scan(
        smoother_ascan_body, pre_smoother, reverse=True
    )

    # Get cross covariances, post-hoc. Can we do this more efficiently within the associative_scan?
    def get_cross_cov(k):
        K = pre_smoother.E[k]
        smoother_cov = smoother_elements.L[k + 1]
        return K @ smoother_cov

    cross_covs = vmap(get_cross_cov)(jnp.arange(K))

    return KalmanState(
        mean=smoother_elements.g, cov=smoother_elements.L, cross_cov=cross_covs
    )
