import chex
import jax
import jax.numpy as jnp
import pytest

from cuthbertlib.smc import backward
from tests.cuthbertlib.kalman.test_smoothing import std_kalman_smoother
from tests.cuthbertlib.kalman.utils import generate_lgssm


@pytest.mark.parametrize("seed", [0, 42, 99, 123, 456])
@pytest.mark.parametrize("x_dim", [2])
@pytest.mark.parametrize("N", [10_000])
def test_backward_simulate(seed, x_dim, N):
    m0, chol_P0, Fs, cs, chol_Qs = generate_lgssm(seed, x_dim, 0, 1)[:5]
    m1, chol_P1 = generate_lgssm(seed + 1, x_dim, 0, 0)[:2]

    # Backward simulation fails unless there is overlap between
    # filter distribution p(x0, y0) and smoother distribution p(x0 | y0, y1).
    # To encourage this we increase the variance of the smoother distribution,
    # therefore decreasing influence of y1 on p(x0 | y0, y1).
    # Although you couldn't do this in practice.
    chol_Qs *= 5.0

    F, c, chol_Q = Fs[0], cs[0], chol_Qs[0]

    # Run standard Kalman smoother for one step
    ms = jax.numpy.vstack([m0, m1])
    chol_Ps = jax.numpy.vstack([chol_P0[None], chol_P1[None]])
    Ps = chol_Ps @ chol_Ps.transpose(0, 2, 1)
    Qs = chol_Qs @ chol_Qs.transpose(0, 2, 1)
    (des_smooth_ms, des_smooth_Ps), des_cross_covs = std_kalman_smoother(
        ms, Ps, Fs, cs, Qs
    )
    des_m0, des_P0 = des_smooth_ms[0], des_smooth_Ps[0]

    # Sample filter particles
    key = jax.random.PRNGKey(seed)
    x0_key, x1_key, sim_key = jax.random.split(key, 3)
    x0s = jax.vmap(lambda z: m0 + chol_P0 @ z)(jax.random.normal(x0_key, (N, x_dim)))
    x1s = jax.vmap(lambda z: m1 + chol_P1 @ z)(jax.random.normal(x1_key, (N, x_dim)))

    # Run backward simulation
    prec_Q = jnp.linalg.inv(chol_Q @ chol_Q.T)

    def log_conditional_density(x0, x1):
        diff = x1 - F @ x0 - c
        return -0.5 * jnp.sum(diff @ prec_Q * diff)

    smoothed_x0s, smoothed_x0_indices = backward.simulate(
        sim_key, x0s, x1s, jnp.zeros(N), log_conditional_density
    )

    # Check indices are correct
    chex.assert_trees_all_equal(smoothed_x0s, x0s[smoothed_x0_indices])

    # Check marginal mean and covariance of samples are correct
    sample_x0_mean = jnp.mean(smoothed_x0s, axis=0)
    sample_x0_cov = jnp.cov(smoothed_x0s, rowvar=False)
    chex.assert_trees_all_close(
        (sample_x0_mean, sample_x0_cov), (des_m0, des_P0), atol=1e-1
    )  # atol is quite large but it's Monte Carlo and N^2 cost, worth revisiting at some point

    # Check cross-covariance is correct
    sample_x0_x1_cov = jnp.cov(smoothed_x0s, x1s, rowvar=False)[:x_dim, x_dim:]
    chex.assert_trees_all_close(
        sample_x0_x1_cov, des_cross_covs[0], atol=1e-1
    )  # Again atol is quite large
