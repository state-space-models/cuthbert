from functools import partial
from jax import vmap
from jax.lax import scan

import kalman

from cuthbert.types import ArrayTreeLike
from cuthbert.inference import Inference
from cuthbert.generalised_kalman.conditional_moments import (
    InitMoments,
    DynamicsMoments,
    LikelihoodMoments,
    ConditionalMomentsSSM,
)


### TODO: How to interface with FeynmanKac? Or maybe not needed?
### TODO: support parallel for filter?
### TODO: work out how to do smoothing between observations?
### TODO: support ArrayTree for states? Difficult for covariances, but possible see https://docs.jax.dev/en/latest/_autosummary/jax.hessian.html


def build_inference(
    conditional_moments_ssm: ConditionalMomentsSSM,
    parallel_smoothing: bool = True,
) -> Inference:
    init_moments, dynamics_moments, likelihood_moments = conditional_moments_ssm
    return Inference(
        init=partial(init, init_moments=init_moments),
        predict=partial(predict, dynamics_moments=dynamics_moments),
        update=partial(update, likelihood_moments=likelihood_moments),
        filter=partial(filter, likelihood_moments=likelihood_moments),
        smoother=partial(
            smoother,
            dynamics_moments=dynamics_moments,
            parallel=parallel_smoothing,
        ),
    )


def init(
    inputs: ArrayTreeLike,
    init_moments: InitMoments,
) -> kalman.KalmanState:
    init_mean, init_chol_cov = init_moments(inputs)
    return kalman.KalmanState(init_mean, init_chol_cov)


def predict(
    state: kalman.KalmanState,
    inputs: ArrayTreeLike,
    dynamics_moments: DynamicsMoments,
) -> kalman.KalmanState:
    F, c, chol_P = dynamics_moments(state.mean, state.chol_cov, inputs)
    return kalman.predict(state.mean, state.chol_cov, F, c, chol_P)


def update(
    state: kalman.KalmanState,
    inputs: ArrayTreeLike,
    observation: ArrayTreeLike,
    likelihood_moments: LikelihoodMoments,
) -> tuple[kalman.KalmanState, kalman.KalmanFilterInfo]:
    H, d, chol_R = likelihood_moments(state.mean, state.chol_cov, observation, inputs)
    return kalman.filter_update(state.mean, state.chol_cov, H, d, chol_R, observation)


#### How to support parallel? As the moments generator needs to know the previous state
#### which we don't have access to for all time steps at the start.
def filter(
    inputs: ArrayTreeLike,
    observations: ArrayTreeLike,
    init_moments: InitMoments,
    dynamics_moments: DynamicsMoments,
    likelihood_moments: LikelihoodMoments,
) -> tuple[kalman.KalmanState, kalman.KalmanFilterInfo]:
    def body(state, inputs_and_observations):
        inputs, observations = inputs_and_observations
        state = predict(state, inputs, dynamics_moments)
        state, info = update(state, inputs, observations, likelihood_moments)
        return state, (state, info)

    init_state = init(inputs, init_moments)
    return scan(body, init_state, (inputs, observations))[1]


def smoother(
    filter_states: kalman.KalmanState,
    inputs: ArrayTreeLike,
    dynamics_moments: DynamicsMoments,
    parallel: bool = True,
) -> tuple[kalman.KalmanState, kalman.KalmanSmootherInfo]:
    filter_ms = filter_states.mean
    filter_chol_Ps = filter_states.chol_cov
    Fs, cs, chol_Qs = vmap(dynamics_moments)(filter_ms, filter_chol_Ps, inputs)
    return kalman.smoother(filter_ms, filter_chol_Ps, Fs, cs, chol_Qs, parallel)
