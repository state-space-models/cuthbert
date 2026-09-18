import chex
import jax
import jax.numpy as jnp
import pytest
from jax import random
from jax.scipy.linalg import solve_triangular

from cuthbertlib.ensemble_kalman import (
    construct_tapered_chol_innovation_covariance,
)
from cuthbertlib.ensemble_kalman.filtering import (
    _quadratic_form_residual,
    predict,
    update,
)
from cuthbertlib.kalman.filtering import update as kalman_update
from cuthbertlib.kalman.generate import generate_lgssm
from cuthbertlib.linalg import tria


@pytest.fixture(scope="module", autouse=True)
def config():
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", False)


@pytest.mark.parametrize("seed", [0, 42, 99, 123, 456])
@pytest.mark.parametrize("x_dim", [3])
def test_predict_identity(seed, x_dim):
    """Identity dynamics with zero noise should preserve the ensemble."""
    key = random.key(seed)
    N = 100
    ensemble = random.normal(key, (N, x_dim))
    predicted = predict(key, ensemble, lambda x, key: x, inflation=0.0)
    chex.assert_trees_all_close(predicted, ensemble, atol=1e-12)


@pytest.mark.parametrize("seed", [0, 42, 99, 123, 456])
@pytest.mark.parametrize("x_dim", [3])
def test_predict_linear(seed, x_dim):
    """Linear dynamics should shift the ensemble mean correctly."""
    key = random.key(seed)
    N = 1_000_000

    lgssm = generate_lgssm(seed, x_dim, 1, 1)
    F, c, chol_Q = lgssm[2][0], lgssm[3][0], lgssm[4][0]

    # Generate ensemble from known distribution
    m0, chol_P0 = lgssm[0], lgssm[1]
    keys = random.split(key, N)
    ensemble = jax.vmap(lambda k: m0 + chol_P0 @ random.normal(k, (x_dim,)))(keys)

    predicted = predict(
        random.key(seed + 100),
        ensemble,
        lambda x, key: F @ x + c + chol_Q @ random.normal(key, (x_dim,)),
        inflation=0.0,
    )

    # Expected mean: F @ m0 + c (noise is zero-mean)
    expected_mean = F @ m0 + c
    pred_mean = jnp.mean(predicted, axis=0)
    chex.assert_trees_all_close(pred_mean, expected_mean, atol=1e-2)


@pytest.mark.parametrize("seed", [0, 42, 99, 123, 456])
@pytest.mark.parametrize("x_dim", [3])
def test_predict_inflation(seed, x_dim):
    """Inflation should scale deviations from the mean."""
    key = random.key(seed)
    N = 100
    ensemble = random.normal(key, (N, x_dim))
    delta = 0.05

    predicted = predict(key, ensemble, lambda x, key: x, inflation=delta)

    mean = jnp.mean(ensemble, axis=0)
    expected = mean + (1 + delta) * (ensemble - mean)
    chex.assert_trees_all_close(predicted, expected, atol=1e-12)


@pytest.mark.parametrize("seed", [0, 42, 99, 123, 456])
@pytest.mark.parametrize("x_dim", [3])
@pytest.mark.parametrize("y_dim", [1, 2])
def test_update_linear_gaussian(seed, x_dim, y_dim):
    """EnKF update should converge to Kalman update for large ensemble."""
    key = random.key(seed)
    N = 100_000

    lgssm = generate_lgssm(seed, x_dim, y_dim, 1)
    m0, chol_P0 = lgssm[0], lgssm[1]
    H, d, chol_R, y = lgssm[5][0], lgssm[6][0], lgssm[7][0], lgssm[8][0]

    # Generate large ensemble
    keys = random.split(key, N)
    ensemble = jax.vmap(lambda k: m0 + chol_P0 @ random.normal(k, (x_dim,)))(keys)

    # EnKF update
    updated, ll = update(
        random.key(seed + 200),
        ensemble,
        lambda x: H @ x + d,
        chol_R,
        y,
        perturbed_obs=True,
    )

    enkf_mean = jnp.mean(updated, axis=0)
    enkf_dev = updated - enkf_mean
    enkf_cov = enkf_dev.T @ enkf_dev / (N - 1)

    # Kalman update
    (kalman_mean, kalman_chol_cov), kalman_ll = kalman_update(
        m0, chol_P0, H, d, chol_R, y
    )
    kalman_cov = kalman_chol_cov @ kalman_chol_cov.T

    chex.assert_trees_all_close(enkf_mean, kalman_mean, atol=1e-2)
    chex.assert_trees_all_close(enkf_cov, kalman_cov, atol=1e-2)


@pytest.mark.parametrize("seed", [0, 42, 99, 123, 456])
@pytest.mark.parametrize("x_dim", [3])
@pytest.mark.parametrize("y_dim", [1, 2])
def test_update_perturbed_vs_unperturbed(seed, x_dim, y_dim):
    """Both perturbed and unperturbed modes should produce correct shapes."""
    key = random.key(seed)
    N = 100

    lgssm = generate_lgssm(seed, x_dim, y_dim, 1)
    m0, chol_P0 = lgssm[0], lgssm[1]
    H, d, chol_R, y = lgssm[5][0], lgssm[6][0], lgssm[7][0], lgssm[8][0]

    keys = random.split(key, N)
    ensemble = jax.vmap(lambda k: m0 + chol_P0 @ random.normal(k, (x_dim,)))(keys)

    for perturbed in [True, False]:
        updated, ll = update(
            random.key(seed + 300),
            ensemble,
            lambda x: H @ x + d,
            chol_R,
            y,
            perturbed_obs=perturbed,
        )
        chex.assert_shape(updated, (N, x_dim))
        chex.assert_shape(ll, ())
        assert jnp.isfinite(ll)


@pytest.mark.parametrize("seed", [0, 42, 99, 123, 456])
@pytest.mark.parametrize("x_dim", [3])
@pytest.mark.parametrize("y_dim", [1, 2])
def test_update_log_likelihood(seed, x_dim, y_dim):
    """Log-likelihood should match MVN logpdf evaluated at the ensemble mean prediction."""
    key = random.key(seed)
    N = 100_000

    lgssm = generate_lgssm(seed, x_dim, y_dim, 1)
    m0, chol_P0 = lgssm[0], lgssm[1]
    H, d, chol_R, y = lgssm[5][0], lgssm[6][0], lgssm[7][0], lgssm[8][0]

    keys = random.split(key, N)
    ensemble = jax.vmap(lambda k: m0 + chol_P0 @ random.normal(k, (x_dim,)))(keys)

    _, ll = update(
        random.key(seed + 400),
        ensemble,
        lambda x: H @ x + d,
        chol_R,
        y,
        perturbed_obs=True,
    )

    # Reference: Kalman filter log-likelihood
    _, kalman_ll = kalman_update(m0, chol_P0, H, d, chol_R, y)

    chex.assert_trees_all_close(ll, kalman_ll, atol=2e-2)


@pytest.mark.parametrize("seed", [0, 42, 99, 123, 456])
@pytest.mark.parametrize("x_dim", [3])
@pytest.mark.parametrize("y_dim", [1, 2])
def test_update_nan_observation(seed, x_dim, y_dim):
    """NaN observation should return ensemble unchanged with zero log-likelihood."""
    key = random.key(seed)
    N = 100

    lgssm = generate_lgssm(seed, x_dim, y_dim, 1)
    m0, chol_P0 = lgssm[0], lgssm[1]
    H, d, chol_R = lgssm[5][0], lgssm[6][0], lgssm[7][0]
    y_nan = jnp.full(y_dim, jnp.nan)

    keys = random.split(key, N)
    ensemble = jax.vmap(lambda k: m0 + chol_P0 @ random.normal(k, (x_dim,)))(keys)

    updated, ll = update(
        random.key(seed + 500),
        ensemble,
        lambda x: H @ x + d,
        chol_R,
        y_nan,
        perturbed_obs=True,
    )

    chex.assert_trees_all_close(updated, ensemble, atol=1e-12)
    chex.assert_trees_all_close(ll, jnp.array(0.0), atol=1e-12)


@pytest.mark.parametrize("seed", [0, 42, 99, 123, 456])
@pytest.mark.parametrize("x_dim", [3])
@pytest.mark.parametrize("y_dim", [2, 3])
def test_update_partial_nan_observation(seed, x_dim, y_dim):
    """Partially-NaN observations should match Kalman missing-dimension behavior."""
    key = random.key(seed)
    N = 100_000

    lgssm = generate_lgssm(seed, x_dim, y_dim, 1)
    m0, chol_P0 = lgssm[0], lgssm[1]
    H, d, chol_R, y = lgssm[5][0], lgssm[6][0], lgssm[7][0], lgssm[8][0]
    y = y.at[0].set(jnp.nan)

    keys = random.split(key, N)
    ensemble = jax.vmap(lambda k: m0 + chol_P0 @ random.normal(k, (x_dim,)))(keys)

    updated, ll = update(
        random.key(seed + 600),
        ensemble,
        lambda x: H @ x + d,
        chol_R,
        y,
        perturbed_obs=True,
    )
    assert jnp.isfinite(ll)
    assert jnp.all(jnp.isfinite(updated))

    enkf_mean = jnp.mean(updated, axis=0)
    enkf_dev = updated - enkf_mean
    enkf_cov = enkf_dev.T @ enkf_dev / (N - 1)

    (kalman_mean, kalman_chol_cov), kalman_ll = kalman_update(
        m0, chol_P0, H, d, chol_R, y
    )
    kalman_cov = kalman_chol_cov @ kalman_chol_cov.T

    chex.assert_trees_all_close(enkf_mean, kalman_mean, atol=2e-2)
    chex.assert_trees_all_close(enkf_cov, kalman_cov, atol=3e-2)
    chex.assert_trees_all_close(ll, kalman_ll, atol=3e-2)


@pytest.mark.parametrize("localize_marginal", [False, True])
def test_update_covariance_modifiers(localize_marginal):
    ensemble = jnp.array(
        [
            [-2.0, -1.0],
            [-1.0, 2.0],
            [1.0, -2.0],
            [2.0, 1.0],
        ]
    )
    H = jnp.array([[1.0, 0.5], [-0.25, 1.0]])
    chol_R = jnp.diag(jnp.array([0.4, 0.7]))
    y = jnp.array([0.3, -0.6])
    cross_taper = jnp.array([[1.0, 0.25], [0.5, 1.0]])
    marginal_taper = jnp.array([[1.0, 0.2], [0.2, 1.0]]) if localize_marginal else None

    def modify_cross_covariance(C_xy):
        return cross_taper * C_xy

    if marginal_taper is not None:
        chol_marginal_taper = jnp.linalg.cholesky(marginal_taper)

        def construct_chol_innovation_covariance(Y, chol_R):
            return construct_tapered_chol_innovation_covariance(
                Y, chol_marginal_taper, chol_R
            )

    updated, ll = update(
        random.key(0),
        ensemble,
        lambda x: H @ x,
        chol_R,
        y,
        perturbed_obs=False,
        cross_covariance_modifier=modify_cross_covariance,
        construct_chol_innovation_covariance=(
            construct_chol_innovation_covariance if localize_marginal else None
        ),
    )

    y_pred = ensemble @ H.T
    x_dev = ensemble - jnp.mean(ensemble, axis=0)
    y_mean = jnp.mean(y_pred, axis=0)
    y_dev = y_pred - y_mean
    C_xy = cross_taper * (x_dev.T @ y_dev / (ensemble.shape[0] - 1))
    C_yy = y_dev.T @ y_dev / (ensemble.shape[0] - 1)
    if marginal_taper is not None:
        C_yy = marginal_taper * C_yy
    S = C_yy + chol_R @ chol_R.T
    gain = jnp.linalg.solve(S, C_xy.T).T
    expected_updated = ensemble + (y - y_pred) @ gain.T
    innovation = y - y_mean
    _, logdet = jnp.linalg.slogdet(S)
    expected_ll = -0.5 * (
        innovation @ jnp.linalg.solve(S, innovation)
        + logdet
        + y.shape[0] * jnp.log(2 * jnp.pi)
    )

    chex.assert_trees_all_close(updated, expected_updated, rtol=1e-12, atol=1e-12)
    chex.assert_trees_all_close(ll, expected_ll, rtol=1e-12, atol=1e-12)
    if marginal_taper is None:
        _, untapered_ll = update(
            random.key(0),
            ensemble,
            lambda x: H @ x,
            chol_R,
            y,
            perturbed_obs=False,
        )
        chex.assert_trees_all_equal(ll, untapered_ll)


@pytest.mark.parametrize("localize_marginal", [False, True])
def test_update_covariance_modifiers_with_missing_observations(localize_marginal):
    ensemble = jnp.array(
        [
            [-2.0, -1.0],
            [-1.0, 2.0],
            [1.0, -2.0],
            [2.0, 1.0],
        ]
    )
    H = jnp.array(
        [
            [1.0, 0.2],
            [-0.5, 1.0],
            [0.3, -0.8],
            [1.2, 0.4],
        ]
    )
    chol_R = jnp.diag(jnp.array([0.3, 0.4, 0.5, 0.6]))
    y = jnp.array([0.1, jnp.nan, -0.7, jnp.nan])
    cross_taper = jnp.array([[1.0, 0.1, 0.4, 0.2], [0.3, 0.5, 0.8, 0.6]])
    marginal_taper = (
        jnp.array(
            [
                [1.0, 0.1, 0.2, 0.3],
                [0.1, 1.0, 0.4, 0.5],
                [0.2, 0.4, 1.0, 0.6],
                [0.3, 0.5, 0.6, 1.0],
            ]
        )
        if localize_marginal
        else None
    )

    def modify_cross_covariance(C_xy):
        return cross_taper * C_xy

    if marginal_taper is not None:
        chol_marginal_taper = jnp.linalg.cholesky(marginal_taper)

        def construct_chol_innovation_covariance(Y, chol_R):
            return construct_tapered_chol_innovation_covariance(
                Y, chol_marginal_taper, chol_R
            )

    actual = update(
        random.key(0),
        ensemble,
        lambda x: H @ x,
        chol_R,
        y,
        perturbed_obs=False,
        cross_covariance_modifier=modify_cross_covariance,
        construct_chol_innovation_covariance=(
            construct_chol_innovation_covariance if localize_marginal else None
        ),
    )

    observed = jnp.array([0, 2])
    expected = update(
        random.key(0),
        ensemble,
        lambda x: H[observed] @ x,
        chol_R[observed[:, None], observed],
        y[observed],
        perturbed_obs=False,
        cross_covariance_modifier=lambda C_xy: cross_taper[:, observed] * C_xy,
        construct_chol_innovation_covariance=(
            None
            if marginal_taper is None
            else lambda Y, chol_R: construct_tapered_chol_innovation_covariance(
                Y,
                jnp.linalg.cholesky(marginal_taper[observed[:, None], observed]),
                chol_R,
            )
        ),
    )

    chex.assert_trees_all_close(actual, expected, rtol=1e-12, atol=1e-12)


def _random_update_problem(x_dim, y_dim, n_particles):
    """Ensemble, linear observation function, dense non-diagonal chol_R and y."""
    keys = random.split(random.key(0), 4)
    ensemble = random.normal(keys[0], (n_particles, x_dim))
    H = random.normal(keys[1], (y_dim, x_dim))
    factor = random.normal(keys[2], (y_dim, y_dim))
    chol_R = jnp.linalg.cholesky(factor @ factor.T + jnp.eye(y_dim))
    y = random.normal(keys[3], (y_dim,))
    return ensemble, (lambda x: H @ x), chol_R, y


@pytest.mark.parametrize("perturbed_obs", [True, False])
@pytest.mark.parametrize(("x_dim", "y_dim", "n_particles"), [(6, 12, 4), (3, 4, 10)])
def test_update_ensemble_subspace_matches_dense(
    x_dim, y_dim, n_particles, perturbed_obs
):
    """The ensemble-subspace update is exact, so it must reproduce the dense update.

    (6, 12, 4) has n_particles < y_dim, the regime the path targets; (3, 4, 10) has
    y_dim < n_particles. Between them the dense path also takes both of its gain
    associations, n_particles < x_dim and n_particles >= x_dim.
    """
    ensemble, observation_fn, chol_R, y = _random_update_problem(
        x_dim, y_dim, n_particles
    )
    args = (random.key(1), ensemble, observation_fn, chol_R, y)

    dense = update(*args, perturbed_obs=perturbed_obs)
    subspace = update(*args, perturbed_obs=perturbed_obs, ensemble_subspace=True)

    chex.assert_trees_all_close(subspace, dense, rtol=1e-10, atol=1e-10)


def _quadratic_form_difference(A, z, chol_C):
    r"""Reference evaluation of $z^\top z - z^\top AC^{-1}A^\top z$ as a difference.

    A transparent statement of the identity, where $C = L_CL_C^\top$. It subtracts two
    nearly equal non-negative terms and so loses relative precision, which is why the
    library uses the least-squares residual instead.
    """
    g = A.T @ z
    return z @ z - jnp.sum(jnp.square(solve_triangular(chol_C, g, lower=True)))


@pytest.mark.parametrize(
    ("y_dim", "n_particles", "zeroed_rows"), [(12, 4, 0), (4, 10, 0), (12, 4, 5)]
)
def test_quadratic_form_residual_matches_difference(y_dim, n_particles, zeroed_rows):
    """The residual and difference forms both equal z' (I + A A')^{-1} z.

    Zeroed rows make A rank-deficient, as on the missing-data path.
    """
    keys = random.split(random.key(0), 2)
    A = random.normal(keys[0], (y_dim, n_particles)).at[:zeroed_rows].set(0.0)
    z = random.normal(keys[1], (y_dim,))
    chol_C = tria(jnp.concatenate([A.T, jnp.eye(n_particles)], axis=1))

    expected = z @ jnp.linalg.solve(jnp.eye(y_dim) + A @ A.T, z)

    chex.assert_trees_all_close(_quadratic_form_residual(A, z), expected, rtol=1e-10)
    chex.assert_trees_all_close(
        _quadratic_form_difference(A, z, chol_C), expected, rtol=1e-10
    )


def test_update_ensemble_subspace_missing_observations():
    """Partially and fully missing observations match the dense path.

    The 1D chol_R case uses perturbed_obs=False: collect_nans_chol refactors a 2D
    factor by QR, which can flip diagonal signs, so the 1D and 2D representations of
    one R give equally valid but different perturbation draws from the same key.
    """
    ensemble, observation_fn, chol_R, y = _random_update_problem(6, 6, 4)
    partial = y.at[jnp.array([1, 4])].set(jnp.nan)
    args = (random.key(1), ensemble, observation_fn)

    chex.assert_trees_all_close(
        update(*args, chol_R, partial, ensemble_subspace=True),
        update(*args, chol_R, partial),
        rtol=1e-10,
        atol=1e-10,
    )

    diag_R = 0.4 + 0.1 * jnp.arange(6)
    chex.assert_trees_all_close(
        update(*args, diag_R, partial, perturbed_obs=False, ensemble_subspace=True),
        update(*args, jnp.diag(diag_R), partial, perturbed_obs=False),
        rtol=1e-10,
        atol=1e-10,
    )

    all_missing = jnp.full_like(y, jnp.nan)
    updated, ll = update(*args, chol_R, all_missing, ensemble_subspace=True)
    chex.assert_trees_all_close(updated, ensemble, rtol=1e-12, atol=1e-12)
    chex.assert_trees_all_close(ll, jnp.array(0.0), atol=1e-12)


@pytest.mark.parametrize("perturbed_obs", [True, False])
@pytest.mark.parametrize("form", ["scalar", "diagonal", "dense"])
def test_update_ensemble_subspace_chol_R_forms(form, perturbed_obs):
    """Scalar, 1D and 2D factors of the same R reproduce the dense update.

    The diagonal and dense cases use unequal standard deviations, so that scaling the
    perturbations along the wrong axis would be detected. With nothing missing, no
    factor is refactored, so both paths draw identical perturbations from one key.
    """
    y_dim = 8
    ensemble, observation_fn, _, y = _random_update_problem(6, y_dim, 5)
    if form == "scalar":
        std = jnp.full((y_dim,), 0.7)
        chol_R = jnp.asarray(0.7)
    else:
        std = 0.4 + 0.1 * jnp.arange(y_dim)
        chol_R = std if form == "diagonal" else jnp.diag(std)
    args = (random.key(1), ensemble, observation_fn)

    expected = update(*args, jnp.diag(std), y, perturbed_obs=perturbed_obs)
    actual = update(
        *args, chol_R, y, perturbed_obs=perturbed_obs, ensemble_subspace=True
    )

    chex.assert_trees_all_close(actual, expected, rtol=1e-10, atol=1e-10)


def test_update_ensemble_subspace_rejects_invalid_arguments():
    """Localization hooks are rejected with ensemble_subspace; non-2D chol_R without."""
    ensemble, observation_fn, chol_R, y = _random_update_problem(6, 6, 4)
    args = (random.key(1), ensemble, observation_fn)

    with pytest.raises(ValueError, match="cross_covariance_modifier"):
        update(
            *args,
            chol_R,
            y,
            cross_covariance_modifier=lambda C_xy: C_xy,
            ensemble_subspace=True,
        )
    with pytest.raises(ValueError, match="construct_chol_innovation_covariance"):
        update(
            *args,
            chol_R,
            y,
            construct_chol_innovation_covariance=lambda Y, chol_R: chol_R,
            ensemble_subspace=True,
        )
    for structured_chol_R in [jnp.asarray(0.5), jnp.full((6,), 0.5)]:
        with pytest.raises(ValueError, match="chol_R must be 2D"):
            update(*args, structured_chol_R, y)
