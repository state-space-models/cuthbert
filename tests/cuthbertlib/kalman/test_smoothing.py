import chex
import jax
import pytest

from cuthbertlib.kalman.smoothing import update
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
    (smooth_m0, smooth_chol_P0), smooth_gain = update(
        m0, chol_P0, m1, chol_P1, Fs[0], cs[0], chol_Qs[0]
    )
    smooth_P0 = smooth_chol_P0 @ smooth_chol_P0.T
    cross_cov = smooth_gain @ Ps[1]
    chex.assert_trees_all_close((smooth_m0, smooth_P0), (des_m0, des_P0), rtol=1e-10)
    chex.assert_trees_all_close(cross_cov, des_cross_covs[0], rtol=1e-10)
