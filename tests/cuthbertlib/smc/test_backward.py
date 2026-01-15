from functools import partial

import chex
import jax
import jax.numpy as jnp
import pytest

from cuthbertlib.kalman.filtering import (
    predict as kalman_predict,
)
from cuthbertlib.kalman.filtering import (
    update as kalman_update,
)
from cuthbertlib.kalman.generate import generate_lgssm
from cuthbertlib.smc.smoothing.exact_sampling import simulate as exact
from cuthbertlib.smc.smoothing.mcmc import simulate as mcmc
from cuthbertlib.smc.smoothing.tracing import simulate as tracing
from tests.cuthbertlib.kalman.test_smoothing import std_kalman_smoother


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
@pytest.mark.parametrize("x_dim", [2])
@pytest.mark.parametrize("y_dim", [2])
@pytest.mark.parametrize("N", [5_000])
@pytest.mark.parametrize("method", ["mcmc", "exact", "tracing"])
def test_backward(seed, x_dim, y_dim, N, method):
    """Test SMC backward simulation methods on a two-step linear Gaussian system.

    Setup:
    - x0s ~ p(x0 | y0): filter particles at time 0
    - x1s ~ p(x1 | y0, y1): filter particles at time 1 after observing y1
    - Backward simulation should recover p(x0 | y0, y1): smoothed distribution

    The observation y1 provides information that makes smoothing meaningful.
    """
    # Generate LGSSM with observations
    m0, chol_P0, Fs, cs, chol_Qs, Hs, ds, chol_Rs, ys = generate_lgssm(
        seed, x_dim, y_dim, 1
    )

    F, c, chol_Q = Fs[0], cs[0], chol_Qs[0]
    H1, d1, chol_R1 = Hs[1], ds[1], chol_Rs[1]  # Observation model at time 1
    y1 = ys[1]  # Observation at time 1
    # We ignore/discard the observation at time 0 for this test

    # Sample filter particles at time 0 from p(x0 | y0)
    key = jax.random.key(seed)
    x0_key, x1_key, sim_key = jax.random.split(key, 3)
    x0s = jax.vmap(lambda z: m0 + chol_P0 @ z)(jax.random.normal(x0_key, (N, x_dim)))

    # Compute predicted distribution at time 1: p(x1 | y0)
    m1_pred, chol_P1_pred = kalman_predict(m0, chol_P0, F, c, chol_Q)

    # Update with observation y1 to get filtered distribution: p(x1 | y0, y1)
    (m1, chol_P1), _ = kalman_update(m1_pred, chol_P1_pred, H1, d1, chol_R1, y1)

    # Sample filter particles at time 1 from p(x1 | y0, y1)
    x1s = jax.vmap(lambda z: m1 + chol_P1 @ z)(jax.random.normal(x1_key, (N, x_dim)))

    # Run standard Kalman smoother for one step to get ground truth
    ms = jax.numpy.vstack([m0, m1])
    chol_Ps = jax.numpy.vstack([chol_P0[None], chol_P1[None]])
    Ps = chol_Ps @ chol_Ps.transpose(0, 2, 1)
    Qs = chol_Qs @ chol_Qs.transpose(0, 2, 1)
    (des_smooth_ms, des_smooth_Ps), des_cross_covs = std_kalman_smoother(
        ms, Ps, Fs, cs, Qs
    )
    des_m0, des_P0 = des_smooth_ms[0], des_smooth_Ps[0]

    # Run smoothing simulation
    prec_Q = jnp.linalg.inv(chol_Q @ chol_Q.T)

    def log_conditional_density(x0, x1):
        diff = x1 - F @ x0 - c
        return -0.5 * jnp.sum(diff @ prec_Q * diff)

    # We may want to make this configuration more professional in the future, when we add more methods.
    match method:
        case "tracing":
            backward_method = tracing
        case "exact":
            backward_method = exact
        case "mcmc":
            backward_method = partial(mcmc, n_steps=50)
        case _:
            raise ValueError(f"Unknown method: {method}")

    smoothed_x0s, smoothed_x0_indices = backward_method(
        sim_key, x0s, x1s, jnp.zeros(N), log_conditional_density, jnp.arange(N)
    )

    # Check indices are correct
    chex.assert_trees_all_equal(smoothed_x0s, x0s[smoothed_x0_indices])

    if method != "tracing":
        # tracing here doesn't actually modify the initial particles (we haven't done resampling)
        # so wouldn't test anything and is not expected to match statistics

        # Check marginal mean and covariance of samples are correct
        sample_x0_mean = jnp.mean(smoothed_x0s, axis=0)
        sample_x0_cov = jnp.cov(smoothed_x0s, rowvar=False)
        chex.assert_trees_all_close(
            (sample_x0_mean, sample_x0_cov), (des_m0, des_P0), atol=1e-1, rtol=1e-1
        )  # atol is quite large but it's Monte Carlo and N^2 cost for exact sampling, worth revisiting at some point

        # Check cross-covariance is correct
        sample_x0_x1_cov = jnp.cov(smoothed_x0s, x1s, rowvar=False)[:x_dim, x_dim:]
        chex.assert_trees_all_close(
            sample_x0_x1_cov, des_cross_covs[0], atol=1e-1, rtol=1e-1
        )  # Again atol is quite large
