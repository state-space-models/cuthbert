from functools import partial

import chex
import jax
import jax.numpy as jnp
import pytest
from absl.testing import parameterized
from jax import random

from cuthbert import filter
from cuthbert.inference import Filter
from cuthbert.smc import marginal_particle_filter, particle_filter
from cuthbertlib.kalman.generate import generate_lgssm
from cuthbertlib.resampling import ess_decorator, stop_gradient_decorator, systematic
from cuthbertlib.stats.multivariate_normal import logpdf
from tests.cuthbert.gaussian.test_kalman import std_kalman_filter


@pytest.fixture(scope="module", autouse=True)
def config():
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", False)


def generate_random_walk_model(seed: int, x_dim: int, y_dim: int, num_time_steps: int):
    r"""Generates data from the simple LTI SSM
    x_t = x_{t-1} + \epsilon_t
    y_t = x_t + \eta_t
    where \epsilon_t ~ N(0, 0.05^2) and \eta_t ~ N(0, 0.5^2).
    """
    key = random.key(seed)

    m0 = jnp.zeros((x_dim,))
    chol_P0 = jnp.eye(x_dim)

    F = jnp.eye(x_dim)
    c = jnp.zeros((x_dim,))
    chol_Q = 0.5 * jnp.eye(x_dim)

    H = jnp.zeros((y_dim, x_dim))
    diag_len = min(x_dim, y_dim)
    H = H.at[jnp.arange(diag_len), jnp.arange(diag_len)].set(1.0)
    d = jnp.zeros((y_dim,))
    chol_R = 0.5 * jnp.eye(y_dim)

    key, x0_key = random.split(key)
    x0 = m0 + chol_P0 @ random.normal(x0_key, (x_dim,))

    def body(x_prev, k):
        k_state, k_obs = random.split(k)
        x = F @ x_prev + c + chol_Q @ random.normal(k_state, (x_dim,))
        y = H @ x + d + chol_R @ random.normal(k_obs, (y_dim,))
        return x, y

    scan_keys = random.split(key, num_time_steps)
    _, ys = jax.lax.scan(body, x0, scan_keys)

    # Broadcast the parameters to (N_t, ...) for use with, e.g., std_kalman_filter.
    Fs = jnp.broadcast_to(F, (num_time_steps, x_dim, x_dim))
    cs = jnp.broadcast_to(c, (num_time_steps, x_dim))
    chol_Qs = jnp.broadcast_to(chol_Q, (num_time_steps, x_dim, x_dim))
    Hs = jnp.broadcast_to(H, (num_time_steps, y_dim, x_dim))
    ds = jnp.broadcast_to(d, (num_time_steps, y_dim))
    chol_Rs = jnp.broadcast_to(chol_R, (num_time_steps, y_dim, y_dim))

    return m0, chol_P0, Fs, cs, chol_Qs, Hs, ds, chol_Rs, ys


def load_inference(
    m0,
    chol_P0,
    Fs,
    cs,
    chol_Qs,
    Hs,
    ds,
    chol_Rs,
    ys,
    method,
    use_differentiable_resampling=True,
    noop=False,
    n_filter_particles=None,
    ess_threshold=0.7,
):
    # Maybe make this more flexible in the future if we want to support other methods.
    if method == "bootstrap":
        if n_filter_particles is None:
            n_filter_particles = 1_000_000
        algo = particle_filter
    elif method == "marginal":
        if n_filter_particles is None:
            n_filter_particles = 3_000

        algo = marginal_particle_filter
    else:
        raise ValueError(f"Unknown method: {method}")

    def init_sample(key, model_inputs):
        return m0 + chol_P0 @ random.normal(key, m0.shape)

    if noop:

        def propagate_sample(key, state, model_inputs: int):
            return state

        def log_potential(state_prev, state, model_inputs: int):
            return jnp.zeros(())

    else:

        def propagate_sample(key, state, model_inputs: int):
            idx = model_inputs - 1
            mean_sample = Fs[idx] @ state + cs[idx]
            return mean_sample + chol_Qs[idx] @ random.normal(key, mean_sample.shape)

        def log_potential(state_prev, state, model_inputs: int):
            idx = model_inputs - 1
            return logpdf(
                Hs[idx] @ state + ds[idx], ys[idx], chol_Rs[idx], nan_support=False
            )

    # Decorate the resampling with adaptive behaviour and pass that to the filter
    resampling_fn = systematic.resampling
    if use_differentiable_resampling:
        resampling_fn = stop_gradient_decorator(resampling_fn)
    adaptive_systematic = ess_decorator(resampling_fn, ess_threshold)

    inference = Filter(
        init_prepare=partial(
            algo.init_prepare,
            init_sample=init_sample,
            n_filter_particles=n_filter_particles,
        ),
        filter_prepare=partial(
            algo.filter_prepare,
            init_sample=init_sample,
            n_filter_particles=n_filter_particles,
        ),
        filter_combine=partial(
            algo.filter_combine,
            propagate_sample=propagate_sample,
            log_potential=log_potential,
            resampling_fn=adaptive_systematic,
        ),
        associative=False,
    )

    model_inputs = jnp.arange(len(ys) + 1)
    return inference, model_inputs


class Test(chex.TestCase):
    @chex.variants(with_jit=True, without_jit=True)
    @parameterized.product(
        seed=[0, 41, 99, 123, 456],
        x_dim=[3],
        y_dim=[2],
        num_time_steps=[20],
        method=["bootstrap", "marginal"],
        use_differentiable_resampling=[True, False],
    )
    def test(
        self, seed, x_dim, y_dim, num_time_steps, method, use_differentiable_resampling
    ):
        m0, chol_P0, Fs, cs, chol_Qs, Hs, ds, chol_Rs, ys = generate_lgssm(
            seed, x_dim, y_dim, num_time_steps
        )

        # Run the particle filter.
        inference, model_inputs = load_inference(
            m0,
            chol_P0,
            Fs,
            cs,
            chol_Qs,
            Hs,
            ds,
            chol_Rs,
            ys,
            method,
            use_differentiable_resampling,
        )
        key = random.key(seed + 1)
        states = self.variant(filter, static_argnames=("filter_obj", "parallel"))(
            inference, model_inputs, parallel=False, key=key
        )
        weights = jax.nn.softmax(states.log_weights)
        means = jnp.sum(states.particles * weights[..., None], axis=1)
        covs = jax.vmap(lambda particles, w: jnp.cov(particles.T, aweights=w))(
            states.particles, weights
        )
        ells = states.log_normalizing_constant

        # Run the standard Kalman filter.
        P0 = chol_P0 @ chol_P0.T
        Qs = chol_Qs @ chol_Qs.transpose(0, 2, 1)
        Rs = chol_Rs @ chol_Rs.transpose(0, 2, 1)
        des_means, des_covs, des_ells = std_kalman_filter(
            m0, P0, Fs, cs, Qs, Hs, ds, Rs, ys
        )
        if method == "marginal":
            chex.assert_trees_all_close(
                (ells, means), (des_ells, des_means), atol=4e-1, rtol=0.25
            )
            chex.assert_trees_all_close(covs, des_covs, atol=6e-1, rtol=0.25)

        else:
            chex.assert_trees_all_close(
                (ells, means, covs),
                (des_ells, des_means, des_covs),
                rtol=2e-2,
                atol=2e-2,
            )

    @chex.variants(with_jit=True, without_jit=True)
    @parameterized.product(method=["bootstrap", "marginal"])
    def test_pytree_particles(self, method):
        """Test that the pf handles pytree states correctly."""

        def init_sample(key, model_inputs):
            keys = random.split(key, 2)
            position = random.normal(keys[0], (2,))
            velocity = random.normal(keys[1], (2,))
            return (position, velocity)

        def propagate_sample(key, state, model_inputs):
            position, velocity = state
            new_position = position + velocity * 0.1
            new_velocity = velocity + random.normal(key, (2,))
            return (new_position, new_velocity)

        def log_potential(state_prev, state, model_inputs):
            return jnp.zeros(())

        ess_threshold = 0.7
        # Decorate the resampler for adaptive resampling behaviour
        adaptive_systematic = ess_decorator(systematic.resampling, ess_threshold)

        if method == "bootstrap":
            n_filter_particles = 1_000
            algo = particle_filter
        elif method == "marginal":
            n_filter_particles = 100
            algo = marginal_particle_filter
        else:
            raise ValueError(f"Unknown method: {method}")

        inference = Filter(
            init_prepare=partial(
                algo.init_prepare,
                init_sample=init_sample,
                n_filter_particles=n_filter_particles,
            ),
            filter_prepare=partial(
                algo.filter_prepare,
                init_sample=init_sample,
                n_filter_particles=n_filter_particles,
            ),
            filter_combine=partial(
                algo.filter_combine,
                propagate_sample=propagate_sample,
                log_potential=log_potential,
                resampling_fn=adaptive_systematic,
            ),
            associative=False,
        )

        key = random.key(0)
        num_time_steps = 5

        # Run the particle filter
        model_inputs = jnp.empty(num_time_steps + 1)
        key, subkey = random.split(key)
        states = self.variant(filter, static_argnames=("filter_obj", "parallel"))(
            inference, model_inputs, parallel=False, key=subkey
        )

        # Verify that the pytree structure is preserved
        particles = states.particles
        assert isinstance(particles, tuple) and len(particles) == 2
        expected_shape = (num_time_steps + 1, n_filter_particles, 2)
        chex.assert_shape(particles, expected_shape)

    @parameterized.product(
        seed=[0, 41], x_dim=[2, 1], y_dim=[2, 1], num_time_steps=[10]
    )
    def test_stop_gradient_resampling(self, seed, x_dim, y_dim, num_time_steps):
        """Tests that the stop-gradient DPF provides estimates of the gradient that are
        close (within 20% relative error) of the gradient provided by the Kalman filter when the median
        of 10 trials is taken.
        """
        y_dim = max(x_dim, y_dim)
        m0, chol_P0, Fs, cs, chol_Qs, Hs, ds, chol_Rs, ys = generate_random_walk_model(
            seed, x_dim, y_dim, num_time_steps
        )

        F = Fs[0]
        n_particles = 10_000

        P0 = chol_P0 @ chol_P0.T
        Qs = chol_Qs @ chol_Qs.transpose(0, 2, 1)
        Rs = chol_Rs @ chol_Rs.transpose(0, 2, 1)

        def kalman_mll(F_param):
            Fs_param = jnp.tile(F_param[None, :, :], (num_time_steps, 1, 1))
            _, _, ells = std_kalman_filter(m0, P0, Fs_param, cs, Qs, Hs, ds, Rs, ys)
            return ells[-1]

        def pf_mll(F_param, key):
            Fs_param = jnp.tile(F_param[None, :, :], (num_time_steps, 1, 1))
            filt, model_inputs = load_inference(
                m0,
                chol_P0,
                Fs_param,
                cs,
                chol_Qs,
                Hs,
                ds,
                chol_Rs,
                ys,
                "bootstrap",
                use_differentiable_resampling=True,
                n_filter_particles=n_particles,
                ess_threshold=1.01,  # always resample
            )
            states = filter(filt, model_inputs, parallel=False, key=key)
            return states.log_normalizing_constant[-1]

        kalman_grad = jax.grad(kalman_mll)
        pf_grad = jax.grad(pf_mll, argnums=0)

        grad_key = random.key(seed + 1)
        pf_grads = []
        n_trials = 10
        for _ in range(n_trials):
            grad_key, trial_key = random.split(grad_key)
            pf_grads.append(pf_grad(F, trial_key))

        pf_grad_median = jnp.median(jnp.array(pf_grads), axis=0)
        kf_grad = kalman_grad(F)
        rel_err = jnp.linalg.norm(pf_grad_median - kf_grad) / jnp.linalg.norm(kf_grad)
        assert rel_err < 0.2


@pytest.mark.parametrize("seed", [1, 43, 99, 123, 456])
@pytest.mark.parametrize("x_dim", [1, 10])
@pytest.mark.parametrize("y_dim", [1, 5])
@pytest.mark.parametrize("method", ["bootstrap", "marginal"])
def test_filter_noop(seed, x_dim, y_dim, method):
    lgssm = generate_lgssm(seed, x_dim, y_dim, 0)

    inference, _ = load_inference(*lgssm, method=method, noop=True)

    init_state = inference.init_prepare(None, key=random.key(seed + 1))
    prep_state = inference.filter_prepare(None, key=random.key(seed + 2))
    filtered_state = inference.filter_combine(init_state, prep_state)

    chex.assert_trees_all_close(
        filtered_state._replace(key=None),
        init_state._replace(key=None),
        rtol=1e-10,
        atol=1e-10,
    )
