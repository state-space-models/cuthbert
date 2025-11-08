from typing import NamedTuple

import jax
import jax.numpy as jnp
import optax
from jax import random
from jax.nn import softplus

from cuthbert import filter, smoother
from cuthbert.gaussian import moments
from cuthbertlib.quadrature.gauss_hermite import weights
from cuthbertlib.types import Array


class ParamStruct(NamedTuple):
    a: Array
    b: Array


##### Hyperparameters #####
X_DIM = 6
Y_DIM = 78
SIGMA = 0.3
DELTA = 0.01
T = 25


def get_dynamics_params():
    A = jnp.kron(jnp.array([[1, DELTA], [0, 1]]), jnp.eye(3))
    _chol_Q = SIGMA * jnp.linalg.cholesky(
        jnp.array([[DELTA**3 / 3, DELTA**2 / 2], [DELTA**2 / 2, DELTA]])
    )
    chol_Q = jnp.kron(_chol_Q, jnp.eye(3))
    return A, chol_Q


def get_observation_rate(params: ParamStruct, state: Array) -> Array:
    rate = params.a + params.b @ state
    rate = softplus(rate)  # Ensure rates are > 0
    return rate


def model_factory(params: ParamStruct, ys: Array):
    def get_init_params(model_inputs: int) -> tuple[Array, Array]:
        return jnp.zeros(X_DIM), jnp.eye(X_DIM) * 1e-8

    def get_dynamics_moments(state, model_inputs: int):
        def dynamics_mean_and_chol_cov_func(x):
            A, chol_Q = get_dynamics_params()
            return A @ x, chol_Q

        return dynamics_mean_and_chol_cov_func, state.mean

    def get_observation_moments(state, model_inputs: int):
        def observation_mean_and_chol_cov_func(x):
            rate = get_observation_rate(params, x)
            return rate, jnp.diag(jnp.sqrt(rate))

        return (observation_mean_and_chol_cov_func, state.mean, ys[model_inputs])

    filter_obj = moments.build_filter(
        get_init_params,
        get_dynamics_moments,
        get_observation_moments,
        associative=False,
    )
    smoother_obj = moments.build_smoother(get_dynamics_moments)

    return filter_obj, smoother_obj


def sim_data(num_time_steps: int):
    state = jnp.zeros(X_DIM)
    ys = []
    A, chol_Q = get_dynamics_params()

    key = random.key(0)

    # Sample the true parameters
    key, a_key, b_key = random.split(key, 3)
    true_a = 2.5 + random.normal(a_key, (Y_DIM,))
    # bs are uniformly distributed on the 6-dimensional sphere
    true_b = random.normal(b_key, (Y_DIM, X_DIM))
    b_norms = jnp.linalg.norm(true_b, axis=1, keepdims=True)
    true_b /= b_norms + 1e-12
    true_params = ParamStruct(true_a, true_b)

    for _ in range(num_time_steps):
        key, dyn_key, obs_key = random.split(key, 3)
        state = A @ state + chol_Q @ random.normal(dyn_key, (X_DIM,))

        rate = get_observation_rate(true_params, state)

        if jnp.any(rate < 0):
            raise ValueError("Observation rate must be positive.")

        y = random.poisson(obs_key, rate)
        ys.append(y)

    ys = jnp.array(ys).astype(float)
    # No observation at time 0
    ys = jnp.concatenate([jnp.full((1, Y_DIM), jnp.nan), ys], axis=0)

    return ys, true_params


ys, true_params = sim_data(T)
model_inputs = jnp.arange(T + 1)

# Initialize the parameters
key = random.key(99)
key, a_key, b_key = random.split(key, 3)
init_a = random.normal(a_key, (Y_DIM,))
init_b = random.normal(b_key, (Y_DIM, X_DIM))
params = ParamStruct(init_a, init_b)


def loss_fn(params: ParamStruct, ys: Array, smooth_dist):
    def loss_single_time_step(m, chol_cov, y):
        quadrature = weights(X_DIM, order=3)
        sigma_points = quadrature.get_sigma_points(m, chol_cov)

        @jax.vmap
        def fn_to_integrate(x):
            return jnp.log(get_observation_rate(params, x))

        expected_log_rate = jnp.dot(
            sigma_points.wm, fn_to_integrate(sigma_points.points)
        )
        loss = -1.0 * jnp.sum(y * expected_log_rate - params.a - params.b @ m)
        return loss

    total_loss = jnp.sum(
        jax.vmap(loss_single_time_step)(
            smooth_dist.mean[1:], smooth_dist.chol_cov[1:], ys[1:]
        )  # Skip NaN observation at t=0
    )
    return total_loss


# Set up optimizer
solver = optax.lbfgs()
opt_state = solver.init(params)


@jax.jit
def train_step(params, opt_state, ys, smooth_states):
    def obj_fn(p):
        return loss_fn(p, ys, smooth_states)

    value, grad = optax.value_and_grad_from_state(obj_fn)(params, state=opt_state)
    updates, opt_state = solver.update(
        grad, opt_state, params, value=value, grad=grad, value_fn=obj_fn
    )
    params = optax.apply_updates(params, updates)
    return params, opt_state


# EM loop
for epoch in range(10):
    # E-step: run filter and smoother
    filter_obj, smoother_obj = model_factory(params, ys)
    filt_states = filter(filter_obj, model_inputs)
    smooth_states = smoother(smoother_obj, filt_states)

    # M-step: optimize parameters
    for _ in range(2):
        params, opt_state = train_step(params, opt_state, ys, smooth_states)

    current_loss = loss_fn(params, ys, smooth_states)
    a_diff = jnp.linalg.norm(params.a - true_params.a)
    b_diff = jnp.linalg.norm(params.b - true_params.b)
    print(
        f"Epoch {epoch + 1}, loss: {current_loss:10.3f} ||a - a_true||: {a_diff:8.3f}, ||b - b_true||: {b_diff:8.3f}"
    )
