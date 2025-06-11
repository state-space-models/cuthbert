import itertools
import chex
import jax
import jax.numpy as jnp
import pytest

from cuthbert.generalised_kalman.kalman import filter, sampler, smoother
from tests.cuthbertlib.kalman.test_filtering import std_predict, std_update
from tests.cuthbertlib.kalman.test_smoothing import std_kalman_smoother
from tests.cuthbertlib.kalman.utils import generate_lgssm


@pytest.fixture(scope="module", autouse=True)
def config():
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", False)


def std_kalman_filter(m0, P0, Fs, cs, Qs, Hs, ds, Rs, ys):
    """The standard Kalman filter."""
    # Use for loop instead of jax.lax.scan because std_update supports smaller
    # dimensional updates via NaNs in y
    ms = [m0]
    Ps = [P0]
    ells_incrs = []

    for i in range(len(Fs)):
        F, c, Q, H, d, R, y = Fs[i], cs[i], Qs[i], Hs[i], ds[i], Rs[i], ys[i]
        pred_m, pred_P = std_predict(ms[-1], Ps[-1], F, c, Q)
        m, P, ell_incr = std_update(pred_m, pred_P, H, d, R, y)
        ms.append(m)
        Ps.append(P)
        ells_incrs.append(ell_incr)

    ms = jnp.stack(ms)
    Ps = jnp.stack(Ps)
    ells = jnp.cumsum(jnp.stack(ells_incrs))
    return ms, Ps, ells


seeds = [0, 42, 99, 123, 456]
x_dims = [3]
y_dims = [1, 2]
num_time_steps = [1, 25]

common_params = list(itertools.product(seeds, x_dims, y_dims, num_time_steps))


@pytest.mark.parametrize("seed,x_dim,y_dim,num_time_steps", common_params)
def test_offline_filter(seed, x_dim, y_dim, num_time_steps):
    # Generate a linear-Gaussian state-space model.
    m0, chol_P0, Fs, cs, chol_Qs, Hs, ds, chol_Rs, ys = generate_lgssm(
        seed, x_dim, y_dim, num_time_steps
    )

    ############################################################################
    seed = 0
    x_dim = 3
    y_dim = 2
    num_time_steps = 25

    m0, chol_P0, Fs, cs, chol_Qs, Hs, ds, chol_Rs, ys = generate_lgssm(
        seed, x_dim, y_dim, num_time_steps
    )
    Fs = jnp.eye(x_dim)[None]
    cs = jnp.zeros(x_dim)[None]
    chol_Qs = jnp.zeros((x_dim, x_dim))[None]
    Hs = Hs[1][None]
    ds = ds[1][None]
    chol_Rs = chol_Rs[1][None]
    ys = ys[1][None]
    ys[0][0] *= jnp.nan

    F = Fs[0]
    c = cs[0]
    chol_Q = chol_Qs[0]
    H = Hs[0]
    d = ds[0]
    chol_R = chol_Rs[0]
    y = ys[0]

    flag = jnp.isnan(y)
    H2 = H[~flag]
    d2 = d[~flag]
    chol_R2 = chol_R[~flag][:, ~flag]
    y2 = y[~flag]

    from cuthbertlib.kalman.filtering import sqrt_associative_params_single

    elem, chol = sqrt_associative_params_single(
        m0, chol_P0, F, c, chol_Q, H, d, chol_R, y
    )

    elem2, chol2 = sqrt_associative_params_single(
        m0, chol_P0, F, c, chol_Q, H2, d2, chol_R2, y2
    )

    print(chol @ chol.T)
    print(chol2 @ chol2.T)

    print(elem[0] - elem2[0])
    print(elem[1] - elem2[1])
    print(elem[2] @ elem[2].T - elem2[2] @ elem2[2].T)
    print(elem[3] - elem2[3])
    print(elem[4] @ elem[4].T - elem2[4] @ elem2[4].T)
    print(elem[5] - elem2[5])

    ############################################################################

    if num_time_steps > 1:
        # Set an observation to nan
        ys[1][0] *= jnp.nan

    # Run both sequential and parallel versions of the square root filter.
    (seq_means, seq_chol_covs), (seq_ells,) = filter(
        m0, chol_P0, Fs, cs, chol_Qs, Hs, ds, chol_Rs, ys, parallel=False
    )
    (par_means, par_chol_covs), (par_ells,) = filter(
        m0, chol_P0, Fs, cs, chol_Qs, Hs, ds, chol_Rs, ys, parallel=True
    )

    # Run the standard Kalman filter.
    P0 = chol_P0 @ chol_P0.T
    Qs = chol_Qs @ chol_Qs.transpose(0, 2, 1)
    Rs = chol_Rs @ chol_Rs.transpose(0, 2, 1)
    des_means, des_covs, des_ells = std_kalman_filter(
        m0, P0, Fs, cs, Qs, Hs, ds, Rs, ys
    )

    seq_covs = seq_chol_covs @ seq_chol_covs.transpose(0, 2, 1)
    par_covs = par_chol_covs @ par_chol_covs.transpose(0, 2, 1)
    chex.assert_trees_all_close(
        (seq_means, seq_covs, seq_ells),
        (par_means, par_covs, par_ells),
        (des_means, des_covs, des_ells),
        rtol=1e-10,
    )


@pytest.mark.parametrize("seed,x_dim,y_dim,num_time_steps", common_params)
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


@pytest.mark.parametrize("seed,x_dim,y_dim,num_time_steps", common_params)
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

    # Check default
    sample = sampler(
        key, filt_means, filt_chol_covs, Fs, cs, chol_Qs, parallel=parallel
    )
    assert sample.shape == (num_time_steps + 1, x_dim)

    # Check large number of samples
    shape = (50, 1000)
    samples = sampler(
        key, filt_means, filt_chol_covs, Fs, cs, chol_Qs, shape, parallel=parallel
    )
    assert samples.shape == (*shape, num_time_steps + 1, x_dim)
    samples_flat = samples.reshape(
        (-1, num_time_steps + 1, x_dim)
    )  # Flatten axis 0 and 1
    sample_means = jnp.mean(samples_flat, 0)
    sample_covs = jax.vmap(lambda x: jnp.cov(x, rowvar=False), in_axes=1)(samples_flat)
    chex.assert_trees_all_close(
        (sample_means, sample_covs), (des_means, des_covs), atol=1e-2, rtol=1e-2
    )
