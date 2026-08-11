import chex
import jax
import jax.numpy as jnp
import pytest
from absl.testing import parameterized
from jax import random

from cuthbert import filter, smoother
from cuthbert.ensemble_kalman import ensemble_kalman_filter, ensemble_rts_smoother
from cuthbertlib import ensemble_kalman as enkf_lib
from cuthbertlib.kalman.generate import generate_lgssm
from tests.cuthbert.gaussian.test_kalman import std_kalman_filter
from tests.cuthbertlib.kalman.test_smoothing import std_kalman_smoother


@pytest.fixture(scope="module", autouse=True)
def config():
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", False)


def load_enrts_inference(m0, chol_P0, Fs, cs, chol_Qs, Hs, ds, chol_Rs, ys):
    n_particles = 100_000
    x_dim = m0.shape[0]

    def init_sample(key, model_inputs):
        return m0 + chol_P0 @ random.normal(key, m0.shape)

    def get_dynamics(model_inputs):
        idx = model_inputs - 1
        return lambda x, key: (
            Fs[idx] @ x + cs[idx] + chol_Qs[idx] @ random.normal(key, (x_dim,))
        )

    def get_observations(model_inputs):
        idx = model_inputs - 1
        return lambda x: Hs[idx] @ x + ds[idx], chol_Rs[idx], ys[idx]

    filter_obj = ensemble_kalman_filter.build_filter(
        init_sample=init_sample,
        get_dynamics=get_dynamics,
        get_observations=get_observations,
        n_particles=n_particles,
        store_predicted_ensemble=True,
    )
    smoother_obj = ensemble_rts_smoother.build_smoother()
    model_inputs = jnp.arange(len(ys) + 1)
    return filter_obj, smoother_obj, model_inputs


class Test(chex.TestCase):
    @chex.variants(with_jit=True, without_jit=True)
    @parameterized.product(
        seed=[0, 123, 456],
        x_dim=[3],
        y_dim=[2],
        num_time_steps=[20],
    )
    def test_smoother(self, seed, x_dim, y_dim, num_time_steps):
        m0, chol_P0, Fs, cs, chol_Qs, Hs, ds, chol_Rs, ys = generate_lgssm(
            seed, x_dim, y_dim, num_time_steps
        )
        filter_obj, smoother_obj, model_inputs = load_enrts_inference(
            m0, chol_P0, Fs, cs, chol_Qs, Hs, ds, chol_Rs, ys
        )

        init_key, filter_key = random.split(random.key(seed + 1))
        init_state = filter_obj.init_prepare(model_inputs[0], key=init_key)
        filtered_states = filter(
            filter_obj,
            model_inputs[1:],
            init_state,
            parallel=False,
            key=filter_key,
        )

        def first_dynamics(x, key):
            return Fs[0] @ x + cs[0] + chol_Qs[0] @ random.normal(key, (x_dim,))

        expected_first_prediction = enkf_lib.predict(
            random.split(filtered_states.key[0], 3)[0],
            filtered_states.ensemble[0],
            first_dynamics,
        )
        chex.assert_trees_all_close(
            filtered_states.predicted_ensemble[1],
            expected_first_prediction,
            rtol=1e-12,
            atol=1e-12,
        )

        smoothed_states = self.variant(
            smoother, static_argnames=("smoother_obj", "parallel")
        )(
            smoother_obj,
            filtered_states,
            model_inputs,
            parallel=False,
        )

        P0 = chol_P0 @ chol_P0.T
        Qs = chol_Qs @ chol_Qs.transpose(0, 2, 1)
        Rs = chol_Rs @ chol_Rs.transpose(0, 2, 1)
        filtered_means, filtered_covs, _ = std_kalman_filter(
            m0, P0, Fs, cs, Qs, Hs, ds, Rs, ys
        )
        (desired_means, desired_covs), desired_cross_covs = std_kalman_smoother(
            filtered_means, filtered_covs, Fs, cs, Qs
        )

        smoothed_means = smoothed_states.mean
        smoothed_chol_covs = smoothed_states.chol_cov
        smoothed_covs = smoothed_chol_covs @ smoothed_chol_covs.transpose(0, 2, 1)
        smoothed_deviations = smoothed_states.ensemble - smoothed_means[:, None, :]
        smoothed_cross_covs = jnp.einsum(
            "tni,tnj->tij",
            smoothed_deviations[:-1],
            smoothed_deviations[1:],
        ) / (smoothed_states.n_particles - 1)

        chex.assert_trees_all_close(
            (smoothed_means, smoothed_covs),
            (desired_means, desired_covs),
            rtol=2e-2,
            atol=2e-2,
        )
        chex.assert_trees_all_close(
            smoothed_cross_covs,
            desired_cross_covs,
            rtol=3e-2,
            atol=3e-2,
        )
        chex.assert_trees_all_close(
            smoothed_states.ensemble[-1],
            filtered_states.ensemble[-1],
            rtol=0.0,
            atol=0.0,
        )
