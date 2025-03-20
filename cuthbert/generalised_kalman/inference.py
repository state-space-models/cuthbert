from functools import partial
from jax import vmap
from jax.lax import scan

import kalman

from cuthbert.types import ArrayTreeLike
from cuthbert.inference import Inference
from cuthbert.generalised_kalman.ssm import (
    InitParams,
    DynamicsParams,
    LikelihoodParams,
    LinearGaussianSSM,
)


### TODO: How to interface with FeynmanKac? Or maybe not needed?
### TODO: support parallel for filter?
### TODO: work out how to do smoothing between observations?
### TODO: support ArrayTree for states? Difficult for covariances, but possible see https://docs.jax.dev/en/latest/_autosummary/jax.hessian.html
### TODO: Add optional but not used random key to unified inference to support MCKF etc


def build_inference(
    linear_gaussian_ssm: LinearGaussianSSM,
    parallel_smoothing: bool = True,
) -> Inference:
    init_params, dynamics_params, likelihood_params = linear_gaussian_ssm
    return Inference(
        init=partial(init, init_params=init_params),
        predict=partial(predict, dynamics_params=dynamics_params),
        update=partial(update, likelihood_params=likelihood_params),
        filter=partial(
            filter,
            init_params=init_params,
            dynamics_params=dynamics_params,
            likelihood_params=likelihood_params
        ),
        smoother=partial(
            smoother,
            dynamics_params=dynamics_params,
            parallel=parallel_smoothing,
        ),
    )


def init(
    inputs: ArrayTreeLike,
    init_params: InitParams,
) -> kalman.KalmanState:
    init_mean, init_chol_cov = init_params(inputs)
    return kalman.KalmanState(init_mean, init_chol_cov)


def predict(
    state: kalman.KalmanState,
    inputs: ArrayTreeLike,
    dynamics_params: DynamicsParams,
) -> kalman.KalmanState:
    F, c, chol_P = dynamics_params(state.mean, state.chol_cov, inputs)
    return kalman.predict(state.mean, state.chol_cov, F, c, chol_P)


def update(
    state: kalman.KalmanState,
    inputs: ArrayTreeLike,
    observation: ArrayTreeLike,
    likelihood_params: LikelihoodParams,
) -> tuple[kalman.KalmanState, kalman.KalmanFilterInfo]:
    H, d, chol_R = likelihood_params(state.mean, state.chol_cov, observation, inputs)
    return kalman.filter_update(state.mean, state.chol_cov, H, d, chol_R, observation)


#### How to support parallel? As the params generator needs to know the previous state
#### which we don't have access to for all time steps at the start.
def filter(
    inputs: ArrayTreeLike,
    observations: ArrayTreeLike,
    init_params: InitParams,
    dynamics_params: DynamicsParams,
    likelihood_params: LikelihoodParams,
) -> tuple[kalman.KalmanState, kalman.KalmanFilterInfo]:
    def body(state, inputs_and_observations):
        inputs, observations = inputs_and_observations
        state = predict(state, inputs, dynamics_params)
        state, info = update(state, inputs, observations, likelihood_params)
        return state, (state, info)

    init_state = init(inputs, init_params)
    return scan(body, init_state, (inputs, observations))[1]


def smoother(
    filter_states: kalman.KalmanState,
    inputs: ArrayTreeLike,
    dynamics_params: DynamicsParams,
    parallel: bool = True,
) -> tuple[kalman.KalmanState, kalman.KalmanSmootherInfo]:
    filter_ms = filter_states.mean
    filter_chol_Ps = filter_states.chol_cov
    Fs, cs, chol_Qs = vmap(dynamics_params)(filter_ms, filter_chol_Ps, inputs)
    return kalman.smoother(filter_ms, filter_chol_Ps, Fs, cs, chol_Qs, parallel)
