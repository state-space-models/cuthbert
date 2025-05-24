from functools import partial
from jax import random, numpy as jnp
from jax.lax import cond

from cuthbertlib.kalman import (
    KalmanState,
    KalmanFilterInfo,
)
from cuthbertlib import kalman

from cuthbert.types import ArrayTreeLike, KeyArray
from cuthbert.inference import Inference
from cuthbert.generalised_kalman.linear_gaussian_ssm import (
    InitParams,
    DynamicsParams,
    LikelihoodParams,
    LinearGaussianSSM,
)


def build_inference(
    linear_gaussian_ssm: LinearGaussianSSM,
) -> Inference:
    _, _, init_params, dynamics_params, likelihood_params = linear_gaussian_ssm
    return Inference(
        init=partial(init, init_params=init_params),
        predict=partial(predict, dynamics_params=dynamics_params),
        filter_update=partial(
            filter_update,
            dynamics_params=dynamics_params,
            likelihood_params=likelihood_params,
        ),
        smoother_combine=partial(smoother_combine, dynamics_params=dynamics_params),
        associative_filter_init=partial(
            associative_filter_init, lgssm=linear_gaussian_ssm
        ),
        associative_filter_combine=associative_filter_combine,
        associative_smoother_init=partial(
            associative_smoother_init, lgssm=linear_gaussian_ssm
        ),
        associative_smoother_combine=associative_smoother_combine,
    )


def init(
    inputs: ArrayTreeLike,
    init_params: InitParams,
    key: KeyArray | None = None,
) -> KalmanState:
    init_mean, init_chol_cov = init_params(inputs, key)
    return KalmanState(init_mean, init_chol_cov)


def predict(
    state_prev: KalmanState,
    inputs: ArrayTreeLike,
    dynamics_params: DynamicsParams,
    key: KeyArray | None = None,
) -> KalmanState:
    F, c, chol_P = dynamics_params(state_prev.mean, state_prev.chol_cov, inputs, key)
    return kalman.predict(state_prev.mean, state_prev.chol_cov, F, c, chol_P)


def _update(
    state: KalmanState,
    observation: ArrayTreeLike,
    inputs: ArrayTreeLike,
    likelihood_params: LikelihoodParams,
    key: KeyArray | None = None,
) -> tuple[kalman.KalmanState, kalman.KalmanFilterInfo]:
    likelihood_or_potential_params = likelihood_params(
        state.mean, state.chol_cov, observation, inputs, key
    )
    H, d, chol_R = likelihood_or_potential_params
    return kalman.filter_update(state.mean, state.chol_cov, H, d, chol_R, observation)


def filter_update(
    state: KalmanState,
    observation: ArrayTreeLike,
    inputs: ArrayTreeLike,
    dynamics_params: DynamicsParams,
    likelihood_params: LikelihoodParams,
    key: KeyArray | None = None,
) -> tuple[KalmanState, KalmanFilterInfo]:
    pred_state = predict(state, inputs, dynamics_params, key)
    return cond(
        observation is None or jnp.isnan(observation).any(),
        lambda: (pred_state, kalman.KalmanFilterInfo(jnp.array(0.0))),
        lambda: _update(pred_state, observation, inputs, likelihood_params, key),
    )


def smoother_combine(
    state_1: kalman.KalmanState,
    state_2: kalman.KalmanState,
    inputs: ArrayTreeLike,
    dynamics_params: DynamicsParams,
    key: KeyArray | None = None,
) -> tuple[kalman.KalmanState, kalman.KalmanSmootherInfo]:
    F, c, chol_Q = dynamics_params(state_1.mean, state_1.chol_cov, inputs, key)
    return kalman.smoother_update(
        state_1.mean,
        state_1.chol_cov,
        state_2.mean,
        state_2.chol_cov,
        F,
        c,
        chol_Q,
    )


def associative_filter_init(
    observation: ArrayTreeLike,
    inputs: ArrayTreeLike,
    lgssm: LinearGaussianSSM,
    key: KeyArray | None = None,
) -> kalman.filtering.FilterScanElement:
    if key:
        init_key, dynamics_key, likelihood_key = random.split(key, 3)
    else:
        init_key, dynamics_key, likelihood_key = None, None, None

    if (
        observation is None or jnp.isnan(observation).any()
    ):  # User has to provide observation=None or jnp.nan for first time step
        m0, chol_P0 = lgssm.init_params(inputs, init_key)
    else:
        m0 = jnp.zeros(lgssm.dim_state)
        chol_P0 = jnp.zeros((lgssm.dim_state, lgssm.dim_state))

    # This is stupid, but we only really support associative filter in the
    # linear Gaussian case i.e. without linearization
    # and in this case the linearization mean and chol_cov are ignored anyway
    linearization_mean = m0
    linearization_chol_cov = chol_P0

    m0, chol_P0 = lgssm.init_params(inputs, init_key)
    F, c, chol_Q = lgssm.dynamics_params(
        linearization_mean, linearization_chol_cov, inputs, dynamics_key
    )
    H, d, chol_R = lgssm.likelihood_params(
        linearization_mean,
        linearization_chol_cov,
        observation,
        inputs,
        likelihood_key,
    )

    return kalman.filtering._sqrt_associative_params_single(
        m0, chol_P0, F, c, chol_Q, H, d, chol_R, observation
    )


def associative_filter_combine(
    state_1: kalman.filtering.FilterScanElement,
    state_2: kalman.filtering.FilterScanElement,
    key: KeyArray | None = None,
) -> tuple[kalman.filtering.FilterScanElement, kalman.KalmanFilterInfo]:
    state = kalman.filtering.sqrt_filtering_operator(state_1, state_2)
    return state, kalman.KalmanFilterInfo(-state.ell)


def associative_smoother_init(
    filter_state: ArrayTreeLike,
    inputs: ArrayTreeLike,
    lgssm: LinearGaussianSSM,
    key: KeyArray | None = None,
) -> kalman.smoothing.SmootherScanElement:
    F, c, chol_Q = lgssm.dynamics_params(
        filter_state.mean, filter_state.chol_cov, inputs, key
    )
    return kalman.smoothing._sqrt_associative_params_single(
        filter_state.mean,
        filter_state.chol_cov,
        F,
        c,
        chol_Q,
    )


def associative_smoother_combine(
    state_1: kalman.smoothing.SmootherScanElement,
    state_2: kalman.smoothing.SmootherScanElement,
    key: KeyArray | None = None,
) -> tuple[kalman.smoothing.SmootherScanElement, kalman.smoothing.KalmanSmootherInfo]:
    state = kalman.smoothing.sqrt_smoothing_operator(state_1, state_2)
    return state, kalman.smoothing.KalmanSmootherInfo(state_1.E)
