from functools import partial

import chex
import jax.numpy as jnp
from absl.testing import parameterized
from jax import random

from cuthbert import filter, smoother
from cuthbert.smc.backward_sampler import build_smoother
from cuthbert.smc.particle_filter import build_filter
from cuthbertlib.kalman.generate import generate_lgssm
from cuthbertlib.resampling import systematic
from cuthbertlib.smc.smoothing.exact_sampling import simulate as exact
from cuthbertlib.smc.smoothing.mcmc import simulate as mcmc
from cuthbertlib.smc.smoothing.tracing import simulate as tracing
from cuthbertlib.stats.multivariate_normal import logpdf
from tests.cuthbert.gaussian.test_kalman import std_kalman_filter
from tests.cuthbertlib.kalman.test_smoothing import std_kalman_smoother


def load_inference(m0, chol_P0, Fs, cs, chol_Qs, Hs, ds, chol_Rs, ys):
    def init_sample(key, model_inputs):
        return m0 + chol_P0 @ random.normal(key, m0.shape)

    def propagate_sample(key, state, model_inputs: int):
        idx = model_inputs - 1
        mean_sample = Fs[idx] @ state + cs[idx]
        return mean_sample + chol_Qs[idx] @ random.normal(key, mean_sample.shape)

    def log_potential(state_prev, state, model_inputs: int):
        idx = model_inputs
        return logpdf(
            Hs[idx] @ state + ds[idx], ys[idx], chol_Rs[idx], nan_support=False
        )

    n_filter_particles = 5000
    resampling_fn = systematic.resampling
    ess_threshold = 0.7
    filter_obj = build_filter(
        init_sample,
        propagate_sample,
        log_potential,
        n_filter_particles,
        resampling_fn,
        ess_threshold,
    )
    model_inputs = jnp.arange(len(ys))
    return filter_obj, model_inputs, log_potential


class Test(chex.TestCase):
    @chex.variants(with_jit=True, without_jit=True)
    @parameterized.product(
        seed=[0, 123, 455],
        x_dim=[3],
        y_dim=[2],
        num_time_steps=[20],
        method=["tracing", "exact", "mcmc"],
    )
    def test(self, seed, x_dim, y_dim, num_time_steps, method):
        m0, chol_P0, Fs, cs, chol_Qs, Hs, ds, chol_Rs, ys = generate_lgssm(
            seed, x_dim, y_dim, num_time_steps
        )

        # Run the particle filter.
        filter_obj, model_inputs, log_potential = load_inference(
            m0, chol_P0, Fs, cs, chol_Qs, Hs, ds, chol_Rs, ys
        )
        key = random.key(seed + 1)
        filtered_states = filter(filter_obj, model_inputs, False, key)

        if method == "tracing":
            bs_fn = tracing
        elif method == "exact":
            bs_fn = exact
        elif method == "mcmc":
            bs_fn = partial(mcmc, n_steps=10)
        else:
            raise ValueError(f"{method} is not a valid backward sampling method.")

        # Run the particle smoother.
        n_smoother_particles = 1000
        smoother_obj = build_smoother(
            log_potential, bs_fn, systematic.resampling, n_smoother_particles
        )
        key, smoother_key = random.split(key)
        smoothed_states = self.variant(
            smoother, static_argnames=("smoother_obj", "parallel")
        )(smoother_obj, filtered_states, model_inputs, False, smoother_key)

        # Run the standard Kalman filter and smoother.
        P0 = chol_P0 @ chol_P0.T
        Qs = chol_Qs @ chol_Qs.transpose(0, 2, 1)
        Rs = chol_Rs @ chol_Rs.transpose(0, 2, 1)
        filtered_means, filtered_covs, _ = std_kalman_filter(
            m0, P0, Fs, cs, Qs, Hs, ds, Rs, ys
        )
        (smoothed_means, _), _ = std_kalman_smoother(
            filtered_means, filtered_covs, Fs, cs, Qs
        )

        # Compare smoothed particles with Kalman smoother results
        particle_means = jnp.mean(smoothed_states.particles, axis=1)
        mse = jnp.mean(jnp.square(particle_means - smoothed_means))
        assert mse <= 0.1, "Mean squared error is too high"

    @chex.variants(with_jit=True, without_jit=True)
    @parameterized.product(method=["tracing", "exact", "mcmc"])
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

        n_filter_particles = 1000
        resampling_fn = systematic.resampling
        ess_threshold = 0.7
        filter_obj = build_filter(
            init_sample,
            propagate_sample,
            log_potential,
            n_filter_particles,
            resampling_fn,
            ess_threshold,
        )

        if method == "tracing":
            bs_fn = tracing
        elif method == "exact":
            bs_fn = exact
        elif method == "mcmc":
            bs_fn = partial(mcmc, n_steps=10)
        else:
            raise ValueError(f"{method} is not a valid backward sampling method.")

        key = random.key(0)
        filter_key, smoother_key = random.split(key)
        num_time_steps = 5
        model_inputs = jnp.empty(num_time_steps + 1)
        filtered_states = filter(filter_obj, model_inputs, False, filter_key)

        n_smoother_particles = 1000
        smoother_obj = build_smoother(
            log_potential, bs_fn, systematic.resampling, n_smoother_particles
        )
        smoothed_states = self.variant(
            smoother, static_argnames=("smoother_obj", "parallel")
        )(smoother_obj, filtered_states, model_inputs, False, smoother_key)

        # Verify that the pyre structure is preserved
        particles = smoothed_states.particles
        assert isinstance(particles, tuple) and len(particles) == 2
        expected_shape = (num_time_steps + 1, n_smoother_particles, 2)
        chex.assert_shape(particles, expected_shape)
