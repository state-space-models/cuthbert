"""Implements the Ensemble Kalman Filter (EnKF) predict and update steps.

See Algorithm 10.2, [Sanz-Alonso et al., Inverse Problems and Data Assimilation](https://arxiv.org/abs/1810.06191).
Based in part on the [CD-Dynamax implementation](https://github.com/hd-UQ/cd_dynamax/blob/public/cd_dynamax/src/continuous_discrete_nonlinear_gaussian_ssm/inference_enkf.py).
"""

from typing import Callable

import jax
import jax.numpy as jnp
from jax import random
from jax.scipy.linalg import cho_solve, solve_triangular

from cuthbertlib.linalg import collect_nans_chol, tria
from cuthbertlib.stats import multivariate_normal
from cuthbertlib.types import Array, KeyArray, ScalarArray

ObservationFn = Callable[[Array], Array]
DynamicsFn = Callable[[Array, KeyArray], Array]
CrossCovarianceModifier = Callable[[Array], Array]
ConstructCholInnovationCovariance = Callable[[Array, Array], Array]


def no_covariance_modifier(covariance: Array) -> Array:
    """Return an empirical covariance unchanged.

    The identity covariance modifier, used as the default when no modification
    (e.g. localization) is requested.

    Args:
        covariance: Empirical covariance matrix.

    Returns:
        The covariance matrix, unchanged.
    """
    return covariance


def _whiten(chol_R: Array, V: Array) -> Array:
    """Apply the inverse observation-noise factor, ``chol_R^{-1} @ V``.

    Args:
        chol_R: Generalized Cholesky factor of the observation noise covariance.
            A 2D array is applied by triangular solve, at O(y_dim ** 2) per column
            of ``V``. A scalar or a 1D array is a multiple of the identity or a
            diagonal factor respectively, and is applied by division at O(y_dim).
        V: Array to whiten, shape (y_dim,) or (y_dim, m).

    Returns:
        Array with the shape of ``V``.

    Raises:
        ValueError: If ``chol_R`` has more than two dimensions.
    """
    if jnp.ndim(chol_R) == 2:
        return solve_triangular(chol_R, V, lower=True)

    if jnp.ndim(chol_R) <= 1:
        # Broadcast the diagonal down the rows of V, which is (y_dim,) or (y_dim, m).
        scale = chol_R if jnp.ndim(V) == 1 else jnp.reshape(chol_R, (-1, 1))
        return V / scale

    raise ValueError(
        "chol_R must be a scalar, a 1D diagonal factor, or a 2D factor, but has "
        f"{jnp.ndim(chol_R)} dimensions."
    )


def _apply_chol(chol_R: Array, V: Array) -> Array:
    """Apply the observation-noise factor, ``chol_R @ V``.

    Args:
        chol_R: Generalized Cholesky factor, as in [_whiten][cuthbertlib.ensemble_kalman.filtering._whiten].
        V: Array to scale, shape (y_dim,) or (y_dim, m).

    Returns:
        Array with the shape of ``V``.

    Raises:
        ValueError: If ``chol_R`` has more than two dimensions.
    """
    if jnp.ndim(chol_R) == 2:
        return chol_R @ V

    if jnp.ndim(chol_R) <= 1:
        # Broadcast the diagonal down the rows of V, which is (y_dim,) or (y_dim, m).
        scale = chol_R if jnp.ndim(V) == 1 else jnp.reshape(chol_R, (-1, 1))
        return scale * V

    raise ValueError(
        "chol_R must be a scalar, a 1D diagonal factor, or a 2D factor, but has "
        f"{jnp.ndim(chol_R)} dimensions."
    )


def _log_det_from_chol(chol: Array) -> ScalarArray:
    """Log-determinant of ``chol @ chol.T`` from a 1D diagonal or 2D factor.

    Uses the absolute diagonal, since a generalized Cholesky factor produced by
    [tria][cuthbertlib.linalg.tria] (a QR) may carry negative diagonal entries.

    A scalar factor is rejected rather than accepted: it carries no dimension, so
    the determinant of the covariance it stands for is undefined without ``y_dim``.
    Callers reach this only after
    [collect_nans_chol][cuthbertlib.linalg.collect_nans_chol], which expands a
    scalar factor to a 1D factor of length ``y_dim``.

    Args:
        chol: Generalized Cholesky factor, 1D (diagonal) or 2D.

    Returns:
        Scalar log-determinant.

    Raises:
        ValueError: If ``chol`` is not 1D or 2D.
    """
    if jnp.ndim(chol) == 2:
        diagonal = jnp.diag(chol)
    elif jnp.ndim(chol) == 1:
        diagonal = chol
    else:
        raise ValueError(
            "chol must be a 1D diagonal factor or a 2D factor, but has "
            f"{jnp.ndim(chol)} dimensions."
        )

    return 2 * jnp.sum(jnp.log(jnp.abs(diagonal)))


def _quadratic_form_residual(A: Array, z: Array) -> ScalarArray:
    r"""Evaluates a quadratic form as a least-squares residual.

    $$z^\top z - z^\top AC^{-1}A^\top z$$

    with $C = A^\top A + I_N$. Implements

    $$\min_{v}\ \left\| \begin{bmatrix} z \\ 0 \end{bmatrix}
        - \begin{bmatrix} A \\ I_N \end{bmatrix} v \right\|^2$$

    which equals the form above, since the objective is
    $z^\top z - 2v^\top g + v^\top Cv$ with $g = A^\top z$, minimized at $v = C^{-1}g$.
    It is evaluated as $\|b - QQ^\top b\|^2$ for $b = [z;\,0]$ and a thin QR
    factorization $[A;\,I_N] = QW$, the residual of the orthogonal projection of $b$
    onto the column space.

    Unlike evaluating the difference directly, nothing of comparable size is subtracted
    and the minimizer is never formed, so relative accuracy does not degrade. The
    identity block also makes $[A;\,I_N]$ full column rank whatever the rank of $A$.

    Args:
        A: matrix, shape (y_dim, N).
        z: vector, shape (y_dim,).

    Returns:
        Scalar quadratic form.
    """
    n_particles = A.shape[1]
    dtype = A.dtype

    stacked = jnp.concatenate([A, jnp.eye(n_particles, dtype=dtype)], axis=0)
    target = jnp.concatenate([z, jnp.zeros(n_particles, dtype=dtype)])

    basis, _ = jnp.linalg.qr(stacked)
    residual = target - basis @ (basis.T @ target)
    return residual @ residual


def predict(
    key: KeyArray,
    ensemble: Array,
    dynamics_fn: DynamicsFn,
    inflation: float = 0.0,
) -> Array:
    """Propagate ensemble members through an arbitrary simulator p(x_{t+1} | x_t).

    Args:
        key: JAX PRNG key.
        ensemble: Ensemble of state vectors, shape (N, x_dim).
        dynamics_fn: Dynamics function mapping (state, key) -> state.
        inflation: Multiplicative inflation factor applied to ensemble deviations.

    Returns:
        Predicted ensemble, shape (N, x_dim).
    """
    N, x_dim = ensemble.shape

    # Propagate each member through the dynamics
    keys = random.split(key, N)
    propagated = jax.vmap(dynamics_fn, (0, 0))(ensemble, keys)

    # Apply multiplicative inflation
    mean = jnp.mean(propagated, axis=0)
    propagated = mean + (1 + inflation) * (propagated - mean)

    return propagated


def update(
    key: KeyArray,
    predicted_ensemble: Array,
    observation_fn: ObservationFn,
    chol_R: Array,
    y: Array,
    perturbed_obs: bool = True,
    cross_covariance_modifier: CrossCovarianceModifier = no_covariance_modifier,
    construct_chol_innovation_covariance: ConstructCholInnovationCovariance
    | None = None,
    ensemble_subspace: bool = False,
) -> tuple[Array, ScalarArray]:
    """Update ensemble members with an observation using the EnKF update.

    NaNs in ``y`` are treated as missing dimensions and are excluded from the
    update. When ``y`` is entirely NaN, the update is a no-op: the predicted
    ensemble is returned unchanged with zero log-likelihood contribution.

    Args:
        key: JAX PRNG key.
        predicted_ensemble: Predicted ensemble, shape (N, x_dim).
        observation_fn: Observation function mapping state -> obs.
        chol_R: Generalized Cholesky factor of the observation noise covariance,
            shape (y_dim, y_dim). Square roots that are not generalized Cholesky
            factors, such as a symmetric R ** 0.5, are not supported.
            When ``ensemble_subspace`` is True this may instead be a scalar (a multiple
            of the identity) or a 1D array of shape (y_dim,) (a diagonal factor).
            Prefer those forms when the structure allows: a 2D factor must be applied
            by triangular solve, at O(N * y_dim ** 2) instead of O(N * y_dim), and
            stored densely in y_dim ** 2 entries. On either path, a 2D factor is also
            refactored in O(y_dim ** 3) at any step where ``y`` has missing values,
            whereas scalar and 1D factors handle missingness in O(y_dim). Steps with
            nothing missing skip the refactor.
        y: Observation vector, shape (y_dim,). NaNs indicate missing dimensions.
        perturbed_obs: If True, use perturbed observations (stochastic EnKF).
            If False, use deterministic update.
        cross_covariance_modifier: Function that modifies the empirical
            state-observation cross-covariance, shape (x_dim, y_dim), and returns
            an array with the same shape. Defaults to the identity.
        construct_chol_innovation_covariance: Optional function that
            receives normalized observation deviations with shape (y_dim, N) and
            ``chol_R`` with shape (y_dim, y_dim). It must return a generalized
            Cholesky factor of the complete innovation covariance with shape
            (y_dim, y_dim). The deviations have already been divided by
            ``sqrt(N - 1)``. Both inputs use the original observation order.
            ``None`` uses the standard, unlocalized square-root construction.
        ensemble_subspace: If True, perform the analysis in the N-dimensional
            ensemble subspace. This is algebraically exact and costs
            O(N ** 2 * x_dim) in the state dimension rather than
            O(N * x_dim * y_dim), so it is preferable when ``N << y_dim``. It is
            incompatible with both localization arguments above, which it rejects.
            Defaults to False; the choice is never made automatically.

    Returns:
        Tuple of (updated_ensemble, log_likelihood).

    Raises:
        ValueError: If ``ensemble_subspace`` is combined with either localization
            argument, or if a non-2D ``chol_R`` is given without it.
    """
    if ensemble_subspace:
        if cross_covariance_modifier is not no_covariance_modifier:
            raise ValueError(
                "ensemble_subspace=True is incompatible with cross_covariance_modifier: "
                "the ensemble-subspace update never forms the state-observation "
                "cross-covariance, so there is nothing to modify."
            )
        if construct_chol_innovation_covariance is not None:
            raise ValueError(
                "ensemble_subspace=True is incompatible with "
                "construct_chol_innovation_covariance: tapering the innovation "
                "covariance destroys the rank-N structure the update relies on."
            )
        return _update_ensemble_subspace(
            key,
            predicted_ensemble,
            observation_fn,
            chol_R,
            y,
            perturbed_obs,
        )

    if jnp.ndim(chol_R) != 2:
        raise ValueError(
            "chol_R must be 2D, of shape (y_dim, y_dim). Scalar and diagonal factors "
            "are only supported with ensemble_subspace=True, because this path "
            "factorizes the dense y_dim x y_dim innovation covariance."
        )

    N, x_dim = predicted_ensemble.shape

    # Map ensemble to observation space
    y_pred = jax.vmap(observation_fn, (0,))(predicted_ensemble)
    x_mean = jnp.mean(predicted_ensemble, axis=0)
    x_dev = predicted_ensemble - x_mean

    missing = jnp.isnan(y)

    # Modify or construct covariances before reordering due to NaNs.
    argsort = jnp.argsort(missing, stable=True)
    original_y_dev = y_pred - jnp.mean(y_pred, axis=0)
    normalized_original_y_dev = original_y_dev.T / jnp.sqrt(N - 1)

    C_xy = x_dev.T @ original_y_dev / (N - 1)
    C_xy = cross_covariance_modifier(C_xy)

    if construct_chol_innovation_covariance is not None:
        original_chol_S = construct_chol_innovation_covariance(
            normalized_original_y_dev, chol_R
        )

    # Handle partially-missing observations by reordering and zeroing missing dims.
    # Use y_pred.T because y_pred is (N, y_dim) and we want to reorder along axis 0.
    # Refactoring chol_R is O(y_dim ** 3); skip it when nothing is missing, in which
    # case the reordering is the identity and the inputs are returned unchanged.
    flag, chol_R, y, y_pred = jax.lax.cond(
        jnp.any(missing),
        lambda args: collect_nans_chol(missing, *args[1:]),
        lambda args: args,
        (missing, chol_R, y, y_pred.T),
    )
    y_pred = y_pred.T
    y_dim = y.shape[0]

    y_mean = jnp.mean(y_pred, axis=0)
    y_dev = y_pred - y_mean
    C_xy = C_xy[:, argsort]
    C_xy = jnp.where(flag[None, :], 0.0, C_xy)

    if construct_chol_innovation_covariance is None:
        chol_S = tria(jnp.concatenate([y_dev.T / jnp.sqrt(N - 1), chol_R], axis=1))
    else:
        # The constructor sees the original indexing. Only collect and refactor its
        # result when dimensions are missing; otherwise preserve its returned factor.
        chol_S = jax.lax.cond(
            jnp.any(missing),
            lambda chol: collect_nans_chol(missing, chol)[1],
            lambda chol: chol,
            original_chol_S,
        )

    # Innovation per member
    if perturbed_obs:
        y_n = y[None, :] + (chol_R @ random.normal(key, (y_dim, N))).T
    else:
        y_n = jnp.broadcast_to(y[None, :], (N, y_dim))

    innovations = y_n - y_pred

    # Doing K = C_xy @ S^{-1}\delta right to left has cost O(Nd_y^2 + Nd_yd_x), left to right O(d_xd_y^2 + Nd_yd_x).
    # If N < d_x, then right to left is cheaper; otherwise left to right is cheaper.
    if N < x_dim:
        increment = cho_solve((chol_S, True), innovations.T).T @ C_xy.T
    else:
        increment = innovations @ cho_solve((chol_S, True), C_xy.T)

    # Update ensemble
    updated = predicted_ensemble + increment

    # Log-likelihood
    ll = multivariate_normal.logpdf(y, y_mean, chol_S, nan_support=False)

    return updated, jnp.asarray(ll)


def _update_ensemble_subspace(
    key: KeyArray,
    predicted_ensemble: Array,
    observation_fn: ObservationFn,
    chol_R: Array,
    y: Array,
    perturbed_obs: bool,
) -> tuple[Array, ScalarArray]:
    r"""EnKF update carried out in the N-dimensional ensemble subspace.

    Algebraically identical to the dense update, more efficient when $N << d_y$. Writing
    $X$ and $Y$ for the state and observation deviations scaled by
    $1/\sqrt{N - 1}$, $\delta$ for the per-member innovations and
    $C = I_N + Y^\top R^{-1} Y$, the update is

    $$X_\text{new} = X_\text{old} + X\,C^{-1}Y^\top R^{-1}\delta$$

    Working in whitened coordinates $A = \mathrm{chol}_R^{-1}Y$ makes
    $C = I_N + A^\top A$, whose factor is obtained by
    [tria][cuthbertlib.linalg.tria] without forming $C$. The log-likelihood follows
    from $\log\det S = \log\det R + \log\det C$, with the quadratic form evaluated by
    [_quadratic_form_residual][cuthbertlib.ensemble_kalman.filtering._quadratic_form_residual].

    Args:
        key: JAX PRNG key.
        predicted_ensemble: Predicted ensemble, shape (N, x_dim).
        observation_fn: Observation function mapping state -> obs.
        chol_R: Generalized Cholesky factor of the observation noise covariance,
            as a scalar, a 1D array of shape (y_dim,), or a 2D array.
        y: Observation vector, shape (y_dim,). NaNs indicate missing dimensions.
        perturbed_obs: If True, use perturbed observations (stochastic EnKF).

    Returns:
        Tuple of (updated_ensemble, log_likelihood).
    """
    N = predicted_ensemble.shape[0]

    # Map ensemble to observation space
    y_pred = jax.vmap(observation_fn, (0,))(predicted_ensemble)
    x_dev = predicted_ensemble - jnp.mean(predicted_ensemble, axis=0)

    # Handle partially-missing observations by reordering and zeroing missing dims.
    # Use y_pred.T because y_pred is (N, y_dim) and we want to reorder along axis 0.
    missing = jnp.isnan(y)
    if jnp.ndim(chol_R) == 2:
        # Refactoring a dense factor is O(y_dim ** 3); skip it when nothing is missing.
        # Scalar and 1D factors are handled in O(y_dim), and a scalar is promoted to 1D,
        # so an identity branch would not match shapes there.
        chol_R, y, y_pred = jax.lax.cond(
            jnp.any(missing),
            lambda args: collect_nans_chol(missing, *args)[1:],
            lambda args: args,
            (chol_R, y, y_pred.T),
        )
    else:
        _, chol_R, y, y_pred = collect_nans_chol(missing, chol_R, y, y_pred.T)
    y_pred = y_pred.T
    y_dim = y.shape[0]

    y_mean = jnp.mean(y_pred, axis=0)
    y_dev = y_pred - y_mean

    scale = jnp.sqrt(jnp.asarray(N - 1, dtype=predicted_ensemble.dtype))

    # Whitened observation deviations, chol_R^{-1} @ Y, shape (y_dim, N)
    whitened_anomalies = _whiten(chol_R, y_dev.T / scale)

    # chol_C @ chol_C.T = I_N + whitened_anomalies.T @ whitened_anomalies, without forming it
    chol_C = tria(
        jnp.concatenate(
            [whitened_anomalies.T, jnp.eye(N, dtype=whitened_anomalies.dtype)],
            axis=1,
        )
    )

    # Innovation per member
    if perturbed_obs:
        noise = random.normal(key, (y_dim, N))
        y_n = y[None, :] + _apply_chol(chol_R, noise).T
    else:
        y_n = jnp.broadcast_to(y[None, :], (N, y_dim))

    whitened_member_innovations = _whiten(chol_R, (y_n - y_pred).T)

    # Ensemble-subspace coefficients C^{-1} Y.T R^{-1} delta, shape (N, N)
    coefficients = cho_solve(
        (chol_C, True), whitened_anomalies.T @ whitened_member_innovations
    )
    updated = predicted_ensemble + coefficients.T @ (x_dev / scale)

    # The log-likelihood uses the unperturbed innovation about the ensemble mean, which
    # is a different object from the per-member innovations driving the update above.
    whitened_mean_innovation = _whiten(chol_R, y - y_mean)
    quadratic_form = _quadratic_form_residual(
        whitened_anomalies, whitened_mean_innovation
    )

    log_det = _log_det_from_chol(chol_R) + _log_det_from_chol(chol_C)
    ll = -0.5 * (y_dim * jnp.log(2 * jnp.pi) + log_det + quadratic_form)

    return updated, jnp.asarray(ll)
