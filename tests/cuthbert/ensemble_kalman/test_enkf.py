import chex
import jax
import jax.numpy as jnp
import pytest
from absl.testing import parameterized
from jax import random

from cuthbert import filter
from cuthbert.ensemble_kalman import ensemble_kalman_filter
from cuthbertlib.ensemble_kalman import (
    construct_tapered_chol_innovation_covariance,
    gaussian,
)
from cuthbertlib.kalman.generate import generate_lgssm
from cuthbertlib.linalg import tria
from tests.cuthbert.gaussian.test_kalman import std_kalman_filter


@pytest.fixture(scope="module", autouse=True)
def config():
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", False)


def load_enkf_inference(
    m0,
    chol_P0,
    Fs,
    cs,
    chol_Qs,
    Hs,
    ds,
    chol_Rs,
    ys,
    noop=False,
    modify_cross_covariance=ensemble_kalman_filter.no_covariance_modifier,
    construct_chol_innovation_covariance=None,
    n_particles=100_000,
    perturbed_obs=True,
):
    x_dim = m0.shape[0]

    def init_sample(key, model_inputs):
        return m0 + chol_P0 @ random.normal(key, m0.shape)

    if noop:
        y_dim = ys.shape[1] if ys.ndim > 1 else 1

        def dynamics_fn(x, key):
            return x

        def get_dynamics(model_inputs):
            return dynamics_fn

        def observation_fn(x):
            return jnp.zeros(y_dim)

        def get_observations(model_inputs):
            return observation_fn, jnp.zeros((y_dim, y_dim)), jnp.full(y_dim, jnp.nan)

    else:

        def get_dynamics(model_inputs):
            idx = model_inputs - 1
            return lambda x, key: (
                Fs[idx] @ x + cs[idx] + chol_Qs[idx] @ random.normal(key, (x_dim,))
            )

        def get_observations(model_inputs):
            idx = model_inputs - 1
            return lambda x: Hs[idx] @ x + ds[idx], chol_Rs[idx], ys[idx]

    inference = ensemble_kalman_filter.build_filter(
        init_sample=init_sample,
        get_dynamics=get_dynamics,
        get_observations=get_observations,
        n_particles=n_particles,
        perturbed_obs=perturbed_obs,
        modify_cross_covariance=modify_cross_covariance,
        construct_chol_innovation_covariance=(construct_chol_innovation_covariance),
    )

    model_inputs = jnp.arange(len(ys) + 1)
    return inference, model_inputs


class Test(chex.TestCase):
    @chex.variants(with_jit=True, without_jit=True)
    @parameterized.named_parameters(
        *((f"seed_{seed}", seed, None) for seed in [0, 41, 99, 123, 456]),
        ("cross_taper", 0, False),
        ("cross_and_marginal_tapers", 0, True),
    )
    def test(self, seed, localize_marginal):
        x_dim = 3
        y_dim = 2
        num_time_steps = 20
        m0, chol_P0, Fs, cs, chol_Qs, Hs, ds, chol_Rs, ys = generate_lgssm(
            seed, x_dim, y_dim, num_time_steps
        )

        modify_cross_covariance = ensemble_kalman_filter.no_covariance_modifier
        construct_chol_innovation_covariance = None
        if localize_marginal is not None:

            def modify_cross_covariance(C_xy, model_inputs):
                return C_xy

            if localize_marginal:

                def construct_chol_innovation_covariance(Y, chol_R, model_inputs):
                    return tria(jnp.concatenate([Y, chol_R], axis=1))

        # Run the EnKF.
        inference, model_inputs = load_enkf_inference(
            m0,
            chol_P0,
            Fs,
            cs,
            chol_Qs,
            Hs,
            ds,
            chol_Rs,
            ys,
            modify_cross_covariance=modify_cross_covariance,
            construct_chol_innovation_covariance=(construct_chol_innovation_covariance),
        )
        init_key, filter_key = random.split(random.key(seed + 1))
        init_state = inference.init_prepare(model_inputs[0], key=init_key)
        states = self.variant(filter, static_argnames=("filter_obj", "parallel"))(
            inference, model_inputs[1:], init_state, parallel=False, key=filter_key
        )
        assert states.predicted_ensemble is None
        means = states.mean
        chol_covs = states.chol_cov
        covs = chol_covs @ chol_covs.transpose(0, 2, 1)
        ells = states.log_normalizing_constant

        # Run the standard Kalman filter.
        P0 = chol_P0 @ chol_P0.T
        Qs = chol_Qs @ chol_Qs.transpose(0, 2, 1)
        Rs = chol_Rs @ chol_Rs.transpose(0, 2, 1)
        des_means, des_covs, des_ells = std_kalman_filter(
            m0, P0, Fs, cs, Qs, Hs, ds, Rs, ys
        )

        chex.assert_trees_all_close(
            (ells, means, covs),
            (des_ells, des_means, des_covs),
            rtol=1e-2,
            atol=1e-2,
        )

    @chex.variants(with_jit=True, without_jit=True)
    def test_nonlinear_dynamics(self):
        seed = 42
        x_dim = 3
        y_dim = 2
        num_time_steps = 5

        m0, chol_P0, Fs, cs, chol_Qs, Hs, ds, chol_Rs, ys = generate_lgssm(
            seed, x_dim, y_dim, num_time_steps
        )

        def init_sample(key, model_inputs):
            return m0 + chol_P0 @ random.normal(key, m0.shape)

        def dynamics_fn(x, key):
            return jnp.tanh(x)

        def get_dynamics(model_inputs):
            return dynamics_fn

        def get_observations(model_inputs):
            idx = model_inputs - 1
            return lambda x: Hs[idx] @ x + ds[idx], chol_Rs[idx], ys[idx]

        inference = ensemble_kalman_filter.build_filter(
            init_sample=init_sample,
            get_dynamics=get_dynamics,
            get_observations=get_observations,
            n_particles=1_000,
        )

        model_inputs = jnp.arange(num_time_steps + 1)

        init_key, filter_key = random.split(random.key(seed + 1))
        init_state = inference.init_prepare(model_inputs[0], key=init_key)
        states = self.variant(filter, static_argnames=("filter_obj", "parallel"))(
            inference, model_inputs[1:], init_state, parallel=False, key=filter_key
        )

        # Check shapes
        chex.assert_shape(states.mean, (num_time_steps + 1, x_dim))
        chex.assert_shape(states.chol_cov, (num_time_steps + 1, x_dim, x_dim))
        assert jnp.all(jnp.isfinite(states.log_normalizing_constant))

        # Check autodiff works (differentiate w.r.t. a parameter)
        def log_nc(m0_):
            def init_sample_(key, model_inputs):
                return m0_ + chol_P0 @ random.normal(key, m0_.shape)

            inference_ = ensemble_kalman_filter.build_filter(
                init_sample=init_sample_,
                get_dynamics=get_dynamics,
                get_observations=get_observations,
                n_particles=1_000,
            )
            init_key, filter_key = random.split(random.key(seed + 1))
            init_state = inference_.init_prepare(model_inputs[0], key=init_key)
            states = filter(
                inference_,
                model_inputs[1:],
                init_state,
                parallel=False,
                key=filter_key,
            )
            return states.log_normalizing_constant[-1]

        grad_val = jax.grad(log_nc)(m0)
        assert jnp.all(jnp.isfinite(grad_val))


def test_gaussian_taper_log_likelihood_gradient():
    """A localized EnKF gradient should agree with fixed-randomness differences."""
    x_dim = 20
    y_dim = 10
    num_time_steps = 4
    n_particles = 6

    state_locations = jnp.arange(x_dim, dtype=float)
    observation_indices = jnp.arange(0, x_dim, 2)
    observation_locations = state_locations[observation_indices]

    dynamics_matrix = (
        0.82 * jnp.eye(x_dim) + 0.08 * jnp.eye(x_dim, k=1) + 0.08 * jnp.eye(x_dim, k=-1)
    )
    observation_matrix = jnp.eye(x_dim)[observation_indices]

    m0 = 0.15 * jnp.sin(state_locations / 3)
    chol_P0 = 0.45 * jnp.eye(x_dim)
    Fs = jnp.broadcast_to(dynamics_matrix, (num_time_steps, x_dim, x_dim))
    cs = jnp.zeros((num_time_steps, x_dim))
    chol_Qs = jnp.broadcast_to(0.08 * jnp.eye(x_dim), (num_time_steps, x_dim, x_dim))
    Hs = jnp.broadcast_to(observation_matrix, (num_time_steps, y_dim, x_dim))
    ds = jnp.zeros((num_time_steps, y_dim))
    chol_Rs = jnp.broadcast_to(0.20 * jnp.eye(y_dim), (num_time_steps, y_dim, y_dim))

    initial_truth = 0.9 * jnp.sin(state_locations / 3) + 0.25 * jnp.cos(
        state_locations / 2
    )

    def simulate_step(truth, time_index):
        truth = dynamics_matrix @ truth + 0.03 * jnp.cos(
            state_locations / 4 + time_index
        )
        observation = observation_matrix @ truth + 0.02 * jnp.sin(
            observation_locations + time_index
        )
        return truth, observation

    _, ys = jax.lax.scan(simulate_step, initial_truth, jnp.arange(num_time_steps))

    cross_distances = state_locations[:, None] - observation_locations[None, :]
    marginal_distances = observation_locations[:, None] - observation_locations[None, :]
    init_key, filter_key = random.split(random.key(314))

    @jax.jit
    def log_marginal_likelihood(log_length_scale):
        length_scale = jnp.exp(log_length_scale)

        def modify_cross_covariance(C_xy, model_inputs):
            return gaussian(cross_distances, length_scale) * C_xy

        chol_marginal_taper = jnp.linalg.cholesky(
            gaussian(marginal_distances, length_scale)
        )

        def construct_chol_innovation_covariance(Y, chol_R, model_inputs):
            return construct_tapered_chol_innovation_covariance(
                Y, chol_marginal_taper, chol_R
            )

        inference, model_inputs = load_enkf_inference(
            m0,
            chol_P0,
            Fs,
            cs,
            chol_Qs,
            Hs,
            ds,
            chol_Rs,
            ys,
            modify_cross_covariance=modify_cross_covariance,
            construct_chol_innovation_covariance=(construct_chol_innovation_covariance),
            n_particles=n_particles,
            perturbed_obs=False,
        )
        init_state = inference.init_prepare(model_inputs[0], key=init_key)
        states = filter(
            inference,
            model_inputs[1:],
            init_state,
            parallel=False,
            key=filter_key,
        )
        return states.log_normalizing_constant[-1]

    log_length_scale = jnp.log(3.0)
    epsilon = 1e-4
    log_likelihood, autodiff_gradient = jax.value_and_grad(log_marginal_likelihood)(
        log_length_scale
    )
    finite_difference_gradient = (
        log_marginal_likelihood(log_length_scale + epsilon)
        - log_marginal_likelihood(log_length_scale - epsilon)
    ) / (2 * epsilon)

    assert jnp.isfinite(log_likelihood)
    assert jnp.isfinite(autodiff_gradient)
    assert jnp.abs(autodiff_gradient) > 1e-3
    chex.assert_trees_all_close(
        autodiff_gradient,
        finite_difference_gradient,
        rtol=1e-5,
        atol=1e-7,
    )


@pytest.mark.parametrize("seed", [1, 43, 99, 123, 456])
@pytest.mark.parametrize("x_dim", [1, 10])
@pytest.mark.parametrize("y_dim", [1, 5])
def test_filter_noop(seed, x_dim, y_dim):
    lgssm = generate_lgssm(seed, x_dim, y_dim, 0)

    inference, _ = load_enkf_inference(*lgssm, noop=True)

    init_state = inference.init_prepare(jnp.array(0), key=random.key(seed + 1))
    prep_state = inference.filter_prepare(jnp.array(1), key=random.key(seed + 2))
    filtered_state = inference.filter_combine(init_state, prep_state)

    filtered_cov = filtered_state.chol_cov @ filtered_state.chol_cov.T
    init_cov = init_state.chol_cov @ init_state.chol_cov.T

    # With identity dynamics, zero noise, and NaN observations,
    # the ensemble, covariance, and log-likelihood should be exactly preserved
    chex.assert_trees_all_close(
        (
            filtered_state.mean,
            filtered_cov,
            filtered_state.log_normalizing_constant,
        ),
        (
            init_state.mean,
            init_cov,
            init_state.log_normalizing_constant,
        ),
        rtol=1e-10,
        atol=1e-10,
    )


def test_build_filter_requires_at_least_two_particles():
    """EnKF should fail fast when configured with fewer than two particles."""

    def init_sample(key, model_inputs):
        return jnp.zeros(1) + jnp.eye(1) @ random.normal(key, (1,))

    with pytest.raises(ValueError, match="at least 2"):
        ensemble_kalman_filter.build_filter(
            init_sample=init_sample,
            get_dynamics=lambda _: lambda x, key: x,
            get_observations=lambda _: (lambda x: x, jnp.eye(1), jnp.zeros(1)),
            n_particles=1,
        )
