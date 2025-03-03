import chex
import jax
import jax.numpy as jnp
import pytest

from kalman import filter, sampler, smoother
from tests.kalman.utils import generate_lgssm


@pytest.mark.parametrize("seed", [0, 42, 99, 123, 456])
@pytest.mark.parametrize("x_dim", [3])
@pytest.mark.parametrize("y_dim", [1, 2])
@pytest.mark.parametrize("num_time_steps", [25])
@pytest.mark.parametrize("parallel", [False, True])
def test_sampler(seed, x_dim, y_dim, num_time_steps, parallel):
    m0, chol_P0, Fs, cs, chol_Qs, Hs, ds, chol_Rs, ys = generate_lgssm(
        seed, x_dim, y_dim, num_time_steps
    )

    (filt_means, filt_chol_covs), _ = filter(
        m0, chol_P0, Fs, cs, chol_Qs, Hs, ds, chol_Rs, ys, parallel=False
    )
    (des_means, des_chol_covs), _ = smoother(
        filt_means, filt_chol_covs, Fs, cs, chol_Qs, parallel=False
    )
    des_covs = des_chol_covs @ des_chol_covs.transpose(0, 2, 1)

    key = jax.random.key(seed)
    num_samples = 50_000
    samples = sampler(
        key, filt_means, filt_chol_covs, Fs, cs, chol_Qs, num_samples, parallel=parallel
    )
    sample_means = jnp.mean(samples, 1)
    sample_covs = jax.vmap(lambda x: jnp.cov(x, rowvar=False))(samples)
    chex.assert_trees_all_close(
        (sample_means, sample_covs), (des_means, des_covs), atol=1e-2, rtol=1e-2
    )
