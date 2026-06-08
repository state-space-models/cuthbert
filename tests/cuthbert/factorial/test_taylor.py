import jax
import jax.numpy as jnp

from cuthbert import factorial
from cuthbert.gaussian import taylor
from cuthbertlib.stats import multivariate_normal
from tests.cuthbert.factorial.gaussian_utils import generate_factorial_kalman_model


def test_factorial_taylor_filter_jit():
    x_dim = 2
    y_dim = 1
    num_factors = 3
    num_factors_local = 2
    num_time_steps = 3

    model_params = generate_factorial_kalman_model(
        seed=0,
        x_dim=x_dim,
        y_dim=y_dim,
        num_factors=num_factors,
        num_factors_local=num_factors_local,
        num_time_steps=num_time_steps,
    )
    m0s, chol_P0s, Fs, cs, chol_Qs, Hs, ds, chol_Rs, ys, _ = model_params
    factorial_indices = jnp.array([[0, 1], [1, 2], [0, 2]])

    # Users have to specify a initial log density that acts on the full factorial state.
    def get_init_log_density(model_inputs):
        def _init_log_density(x):
            return jnp.sum(jax.vmap(multivariate_normal.logpdf)(x, m0s, chol_P0s))

        return _init_log_density, jnp.zeros((num_factors, x_dim))

    def get_dynamics_log_density(state, model_inputs):
        F = Fs[model_inputs - 1]
        c = cs[model_inputs - 1]
        chol_Q = chol_Qs[model_inputs - 1]

        def dynamics_log_density(x_prev, x):
            return multivariate_normal.logpdf(x, F @ x_prev + c, chol_Q)

        return dynamics_log_density, state.mean, F @ state.mean + c

    def get_observation_log_density(state, model_inputs):
        H = Hs[model_inputs - 1]
        d = ds[model_inputs - 1]
        chol_R = chol_Rs[model_inputs - 1]

        def observation_log_density(x, y):
            return multivariate_normal.logpdf(y, H @ x + d, chol_R)

        return observation_log_density, state.mean, ys[model_inputs - 1]

    filter_obj = taylor.build_filter(
        get_init_log_density,
        get_dynamics_log_density,
        get_observation_log_density,
    )
    factorializer = factorial.gaussian.build_factorializer(
        lambda model_inputs: factorial_indices[model_inputs - 1]
    )
    filter_model_inputs = jnp.arange(num_time_steps + 1)

    run_filter = jax.jit(
        lambda model_inputs: factorial.filter(
            filter_obj,
            factorializer,
            model_inputs,
            output_factorial=False,
        )
    )
    init_state, local_filter_states = run_filter(filter_model_inputs)

    assert init_state.mean.shape == (num_factors, x_dim)
    assert init_state.chol_cov.shape == (num_factors, x_dim, x_dim)
    assert jnp.allclose(init_state.mean, m0s, rtol=1e-4, atol=1e-4)
    assert jnp.allclose(
        init_state.chol_cov @ init_state.chol_cov.transpose(0, 2, 1),
        chol_P0s @ chol_P0s.transpose(0, 2, 1),
        rtol=1e-4,
        atol=1e-4,
    )
    assert local_filter_states.mean.shape == (
        num_time_steps,
        num_factors_local,
        x_dim,
    )
    assert local_filter_states.chol_cov.shape == (
        num_time_steps,
        num_factors_local,
        x_dim,
        x_dim,
    )
    assert local_filter_states.log_normalizing_constant.shape == (num_time_steps,)
