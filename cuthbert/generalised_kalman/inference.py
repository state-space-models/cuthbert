from functools import partial
from jax import vmap, random, numpy as jnp
from jax.lax import scan

from cuthbertlib import kalman

from cuthbert.types import ArrayTreeLike, KeyArray
from cuthbert.inference import SSMInference
from cuthbert.generalised_kalman.linear_gaussian_ssm import (
    InitParams,
    DynamicsParams,
    LikelihoodParams,
    PotentialParams,
    LinearGaussianSSM,
)


### TODO: work out how to do smoothing between observations?
### TODO: support ArrayTree for states? Difficult for covariances, but possible see https://docs.jax.dev/en/latest/_autosummary/jax.hessian.html


def build_inference(
    linear_gaussian_ssm: LinearGaussianSSM,
    parallel_smoothing: bool = True,
) -> SSMInference:
    init_params, dynamics_params, likelihood_params = linear_gaussian_ssm
    return SSMInference(
        init=partial(init, init_params=init_params),
        predict=partial(predict, dynamics_params=dynamics_params),
        update=partial(update, likelihood_params=likelihood_params),
        filter=partial(
            filter,
            init_params=init_params,
            dynamics_params=dynamics_params,
            likelihood_params=likelihood_params,
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
    key: KeyArray | None = None,
) -> kalman.KalmanState:
    init_mean, init_chol_cov = init_params(inputs, key)
    return kalman.KalmanState(init_mean, init_chol_cov)


def predict(
    state_prev: kalman.KalmanState,
    inputs: ArrayTreeLike,
    dynamics_params: DynamicsParams,
    key: KeyArray | None = None,
) -> kalman.KalmanState:
    F, c, chol_P = dynamics_params(state_prev.mean, state_prev.chol_cov, inputs, key)
    return kalman.predict(state_prev.mean, state_prev.chol_cov, F, c, chol_P)


def update(
    state: kalman.KalmanState,
    observation: ArrayTreeLike,
    inputs: ArrayTreeLike,
    likelihood_params: LikelihoodParams | PotentialParams,
    key: KeyArray | None = None,
) -> tuple[kalman.KalmanState, kalman.KalmanFilterInfo]:
    likelihood_or_potential_params = likelihood_params(
        state.mean, state.chol_cov, observation, inputs, key
    )

    if len(likelihood_or_potential_params) == 3:
        H, d, chol_R = likelihood_or_potential_params
    else:
        d, chol_R = likelihood_or_potential_params

        # dummy mat and observation as potential is unconditional
        H = jnp.eye(d.shape[0])
        observation = jnp.zeros_like(d)

    return kalman.filter_update(state.mean, state.chol_cov, H, d, chol_R, observation)


def filter(
    observations: ArrayTreeLike,
    inputs: ArrayTreeLike,
    init_params: InitParams,
    dynamics_params: DynamicsParams,
    likelihood_params: LikelihoodParams | PotentialParams,
    key: KeyArray | None = None,
) -> tuple[kalman.KalmanState, kalman.KalmanFilterInfo]:
    if key is not None:
        keys = random.split(key, len(observations) + 1)
    else:
        keys = [None] * (len(observations) + 1)

    def body(state, inputs_and_observations_and_key):
        inputs, observations, key = inputs_and_observations_and_key
        state = predict(state, inputs, dynamics_params, key)
        state, info = update(state, inputs, observations, likelihood_params, key)
        return state, (state, info)

    init_state = init(inputs, init_params, keys[0])
    return scan(body, init_state, (inputs, observations, keys[1:]))[1]


def smoother(
    filter_states: kalman.KalmanState,
    inputs: ArrayTreeLike,
    dynamics_params: DynamicsParams,
    parallel: bool = True,
    key: KeyArray | None = None,
) -> tuple[kalman.KalmanState, kalman.KalmanSmootherInfo]:
    if key is not None:
        key = random.split(key, len(filter_states.mean))

    filter_ms = filter_states.mean
    filter_chol_Ps = filter_states.chol_cov
    Fs, cs, chol_Qs = vmap(dynamics_params)(filter_ms, filter_chol_Ps, inputs, key)
    return kalman.smoother(filter_ms, filter_chol_Ps, Fs, cs, chol_Qs, parallel)
