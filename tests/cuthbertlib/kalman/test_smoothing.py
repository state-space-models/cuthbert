import chex
import jax
import pytest

from cuthbertlib.kalman.filtering import filter
from cuthbertlib.kalman.smoothing import smoother, update
from cuthbertlib.kalman.utils import append_tree
from tests.cuthbertlib.kalman.utils import generate_lgssm


@pytest.fixture(scope="module", autouse=True)
def config():
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", False)


def std_kalman_smoother(ms, Ps, Fs, cs, Qs):
    def body(carry, inp):
        smooth_m, smooth_P = carry
        m, P, F, c, Q = inp

        mean_diff = smooth_m - (F @ m + c)
        S = F @ P @ F.T + Q
        cov_diff = smooth_P - S

        gain = P @ jax.scipy.linalg.solve(S, F, assume_a="pos").T
        cross_cov = gain @ smooth_P
        smooth_m = m + gain @ mean_diff
        smooth_P = P + gain @ cov_diff @ gain.T
        return (smooth_m, smooth_P), ((smooth_m, smooth_P), cross_cov)

    final_state = (ms[-1], Ps[-1])
    _, (smoothed_states, cross_covs) = jax.lax.scan(
        body, final_state, (ms[:-1], Ps[:-1], Fs, cs, Qs), reverse=True
    )
    smoothed_states = append_tree(smoothed_states, final_state)
    return smoothed_states, cross_covs


@pytest.mark.parametrize("seed", [0, 42, 99, 123, 456])
@pytest.mark.parametrize("x_dim", [3])
def test_smoother_update(seed, x_dim):
    m0, chol_P0, Fs, cs, chol_Qs = generate_lgssm(seed, x_dim, 0, 1)[:5]
    m1, chol_P1 = generate_lgssm(seed + 1, x_dim, 0, 0)[:2]

    # Run standard Kalman smoother for one step
    ms = jax.numpy.vstack([m0, m1])
    chol_Ps = jax.numpy.vstack([chol_P0[None], chol_P1[None]])
    Ps = chol_Ps @ chol_Ps.transpose(0, 2, 1)
    Qs = chol_Qs @ chol_Qs.transpose(0, 2, 1)
    (des_smooth_ms, des_smooth_Ps), des_cross_covs = std_kalman_smoother(
        ms, Ps, Fs, cs, Qs
    )
    des_m0, des_P0 = des_smooth_ms[0], des_smooth_Ps[0]

    # Run single square root smoother update
    (smooth_m0, smooth_chol_P0), (smooth_gain,) = update(
        m0, chol_P0, m1, chol_P1, Fs[0], cs[0], chol_Qs[0]
    )
    smooth_P0 = smooth_chol_P0 @ smooth_chol_P0.T
    cross_cov = smooth_gain @ Ps[1]
    chex.assert_trees_all_close((smooth_m0, smooth_P0), (des_m0, des_P0), rtol=1e-10)
    chex.assert_trees_all_close(cross_cov, des_cross_covs[0], rtol=1e-10)


@pytest.mark.parametrize("seed", [0, 42, 99, 123, 456])
@pytest.mark.parametrize("x_dim", [3])
@pytest.mark.parametrize("y_dim", [1, 2])
@pytest.mark.parametrize("num_time_steps", [1, 25])
def test_smoother(seed, x_dim, y_dim, num_time_steps):
    # Generate a linear-Gaussian state-space model.
    m0, chol_P0, Fs, cs, chol_Qs, Hs, ds, chol_Rs, ys = generate_lgssm(
        seed, x_dim, y_dim, num_time_steps
    )

    # Run the Kalman filter and the standard Kalman smoother.
    (filt_means, filt_chol_covs), _ = filter(
        m0, chol_P0, Fs, cs, chol_Qs, Hs, ds, chol_Rs, ys, parallel=False
    )
    filt_covs = filt_chol_covs @ filt_chol_covs.transpose(0, 2, 1)
    Qs = chol_Qs @ chol_Qs.transpose(0, 2, 1)
    (des_means, des_covs), des_cross_covs = std_kalman_smoother(
        filt_means, filt_covs, Fs, cs, Qs
    )

    # Run the sequential and parallel versions of the square root smoother.
    (seq_means, seq_chol_covs), _ = smoother(
        filt_means, filt_chol_covs, Fs, cs, chol_Qs, parallel=False
    )
    (par_means, par_chol_covs), (smoother_gains,) = smoother(
        filt_means, filt_chol_covs, Fs, cs, chol_Qs, parallel=True
    )

    seq_covs = seq_chol_covs @ seq_chol_covs.transpose(0, 2, 1)
    par_covs = par_chol_covs @ par_chol_covs.transpose(0, 2, 1)
    cross_covs = smoother_gains @ par_covs[1:]
    chex.assert_trees_all_close(
        (seq_means, seq_covs), (par_means, par_covs), (des_means, des_covs), rtol=1e-10
    )
    chex.assert_trees_all_close(cross_covs, des_cross_covs, rtol=1e-10)
