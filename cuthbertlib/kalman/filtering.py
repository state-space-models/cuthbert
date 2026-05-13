"""Implements the square root parallel Kalman filter and associative variant."""

from typing import NamedTuple

import jax.numpy as jnp
from jax.scipy.linalg import cho_solve, solve_triangular

from cuthbertlib.linalg import tria
from cuthbertlib.stats import multivariate_normal
from cuthbertlib.stats.multivariate_normal import collect_nans_chol
from cuthbertlib.types import Array, ArrayLike, ScalarArray


class FilterScanElement(NamedTuple):
    """Arrays carried through the Kalman filter scan."""

    A: Array
    b: Array
    U: Array
    eta: Array
    Z: Array
    ell: ScalarArray


def predict(
    m: ArrayLike,
    chol_P: ArrayLike,
    F: ArrayLike,
    c: ArrayLike,
    chol_Q: ArrayLike,
) -> tuple[Array, Array]:
    """Propagate the mean and square root covariance through linear Gaussian dynamics.

    Args:
        m: Mean of the state.
        chol_P: Generalized Cholesky factor of the covariance of the state.
        F: Transition matrix.
        c: Transition shift.
        chol_Q: Generalized Cholesky factor of the transition noise covariance.

    Returns:
        Propagated mean and square root covariance.

    References:
        Paper: G. J. Bierman, Factorization Methods for Discrete Sequential Estimation,
        Code: https://github.com/EEA-sensors/sqrt-parallel-smoothers/tree/main/parsmooth/sequential
    """
    m, chol_P = jnp.asarray(m), jnp.asarray(chol_P)
    F, c, chol_Q = jnp.asarray(F), jnp.asarray(c), jnp.asarray(chol_Q)
    m1 = F @ m + c
    A = jnp.concatenate([F @ chol_P, chol_Q], axis=1)
    chol_P1 = tria(A)
    return m1, chol_P1


# Note that `update` is aliased as `kalman.filter_update` in `kalman.__init__.py`
def update(
    m: ArrayLike,
    chol_P: ArrayLike,
    H: ArrayLike,
    d: ArrayLike,
    chol_R: ArrayLike,
    y: ArrayLike,
    log_normalizing_constant: ArrayLike = 0.0,
    steady_state_params: "SteadyStateFilterParams | None" = None,
) -> tuple[tuple[Array, Array], Array]:
    """Update the mean and square root covariance with a linear Gaussian observation.

    Args:
        m: Mean of the state.
        chol_P: Generalized Cholesky factor of the covariance of the state.
        H: Observation matrix.
        d: Observation shift.
        chol_R: Generalized Cholesky factor of the observation noise covariance.
        y: Observation.
        log_normalizing_constant: Optional input of log normalizing constant to be added to
            log normalizing constant of the Bayesian update.
        steady_state_params: Optional precomputed steady-state parameters
            (see :class:`SteadyStateFilterParams`).  When provided the QR
            decomposition is skipped; ``elem.U`` is used as the posterior
            Cholesky covariance and ``K`` as the Kalman gain.  For the
            sequential filter the caller is responsible for supplying a
            ``SteadyStateFilterParams`` whose ``K`` and ``elem.U`` reflect the
            Riccati steady state, not the parallel-scan values produced by
            :func:`~cuthbert.gaussian.kalman.compute_steady_state_filter_params`.

    Returns:
        Updated mean and square root covariance as well as the log marginal likelihood.

    References:
        Paper: G. J. Bierman, Factorization Methods for Discrete Sequential Estimation,
        Code: https://github.com/EEA-sensors/sqrt-parallel-smoothers/tree/main/parsmooth/sequential
    """
    # Handle case where there is no observation
    flag = jnp.isnan(y)
    flag, chol_R, H, d, y = collect_nans_chol(flag, chol_R, H, d, y)

    m, chol_P = jnp.asarray(m), jnp.asarray(chol_P)
    H, d, chol_R = jnp.asarray(H), jnp.asarray(d), jnp.asarray(chol_R)
    y = jnp.asarray(y)

    n_y, n_x = H.shape

    y_hat = H @ m + d
    y_diff = y - y_hat

    if steady_state_params is None:
        M = jnp.block(
            [
                [H @ chol_P, chol_R],
                [chol_P, jnp.zeros((n_x, n_y), dtype=chol_P.dtype)],
            ]
        )
        chol_S = tria(M)
        chol_Py = chol_S[n_y:, n_y:]
        Gmat = chol_S[n_y:, :n_y]
        Imat = chol_S[:n_y, :n_y]
        my = m + Gmat @ solve_triangular(Imat, y_diff, lower=True)
    else:
        Imat = steady_state_params.chol_S
        chol_Py = steady_state_params.elem.U
        my = m + steady_state_params.K @ y_diff

    ell = multivariate_normal.logpdf(y, y_hat, Imat, nan_support=False)
    return (my, chol_Py), jnp.asarray(ell + log_normalizing_constant)


def associative_params_single(
    F: Array,
    c: Array,
    chol_Q: Array,
    H: Array,
    d: Array,
    chol_R: Array,
    y: Array,
    steady_state_params: "SteadyStateFilterParams | None" = None,
) -> FilterScanElement:
    """Single time step for scan element for square root parallel Kalman filter.

    Args:
        F: State transition matrix.
        c: State transition shift vector.
        chol_Q: Generalized Cholesky factor of the state transition noise covariance.
        H: Observation matrix.
        d: Observation shift.
        chol_R: Generalized Cholesky factor of the observation noise covariance.
        y: Observation.
        steady_state_params: Optional precomputed steady-state parameters
            (see :class:`SteadyStateFilterParams`).  When provided the QR
            decomposition is skipped; ``A``, ``U``, and ``Z`` are reused from
            the stored element and only the observation-dependent ``b``,
            ``eta``, and ``ell`` are evaluated, replacing the expensive
            per-step QR with cheap matrix–vector products.

    Returns:
        Prepared scan element for the square root parallel Kalman filter.
    """
    # Handle case where there is no observation
    flag = jnp.isnan(y)
    flag, chol_R, H, d, y = collect_nans_chol(flag, chol_R, H, d, y)

    ny, nx = H.shape

    inn = y - H @ c - d  # innovation relative to the prior prediction

    if steady_state_params is None:
        # joint over the predictive and the observation
        Psi_ = jnp.block([[H @ chol_Q, chol_R], [chol_Q, jnp.zeros((nx, ny))]])
        Tria_Psi_ = tria(Psi_)

        Psi11 = Tria_Psi_[:ny, :ny]
        Psi21 = Tria_Psi_[ny : ny + nx, :ny]
        U = Tria_Psi_[ny : ny + nx, ny:]

        # pre-compute inverse of Psi11: we apply it to matrices and vectors alike.
        Psi11_inv = solve_triangular(Psi11, jnp.eye(ny), lower=True)

        K = Psi21 @ Psi11_inv  # local Kalman gain
        HF = H @ F
        A = F - K @ HF  # corrected transition matrix

        # information filter (Z_filter is the pre-padded (nx, ny) version)
        Z_filter = HF.T @ Psi11_inv.T
        if nx > ny:
            Z = jnp.concatenate([Z_filter, jnp.zeros((nx, nx - ny))], axis=1)
        else:
            Z = tria(Z_filter)
    else:
        A, _, U, _, Z, _ = steady_state_params.elem
        K = steady_state_params.K
        Psi11 = steady_state_params.chol_S
        Psi11_inv = steady_state_params.Psi11_inv
        Z_filter = steady_state_params.Z_filter

    b = c + K @ inn  # corrected transition offset
    eta = Z_filter @ (Psi11_inv @ inn)
    ell = jnp.asarray(
        multivariate_normal.logpdf(y, H @ c + d, Psi11, nan_support=False)
    )

    return FilterScanElement(A, b, U, eta, Z, ell)


class SteadyStateFilterParams(NamedTuple):
    """Precomputed, time-invariant quantities for a steady-state Kalman filter.

    In a time-invariant linear Gaussian SSM the filter gain and posterior
    covariance converge to constants.  Passing an instance of this class as the
    ``steady_state_params`` argument to :func:`associative_params_single` or
    :func:`update` skips the expensive per-step QR decomposition and reuses the
    constant ``A``, ``U``, and ``Z`` blocks.

    Fields:
        elem: A ``FilterScanElement`` whose ``A``, ``U``, and ``Z`` fields hold
            the steady-state values.  The ``b``, ``eta``, and ``ell`` fields are
            unused (they depend on the observation).
        K: Steady-state Kalman gain matrix, shape ``(nx, ny)``.
        chol_S: Lower-triangular Cholesky factor of the steady-state innovation
            covariance ``S = H P_pred H^T + R``, shape ``(ny, ny)``.
        Psi11_inv: Inverse of ``chol_S``, shape ``(ny, ny)``.  Precomputed so
            that the per-step cost is pure matrix–vector products with no
            triangular solve.
        Z_filter: Pre-padded information gain matrix ``(H F)^T S^{-T}``,
            shape ``(nx, ny)``.  Used to form ``eta`` without the QR
            decomposition; distinct from the square ``Z`` in ``elem`` which is
            used by the associative combine operator.
    """

    elem: FilterScanElement
    K: Array
    chol_S: Array
    Psi11_inv: Array
    Z_filter: Array


def compute_steady_state_filter_params(
    F: Array,
    chol_Q: Array,
    H: Array,
    chol_R: Array,
) -> SteadyStateFilterParams:
    """Compute steady-state filter parameters for a time-invariant linear Gaussian SSM.

    For a time-invariant model the ``A``, ``U``, and ``Z`` fields of each
    associative scan element are identical at every time step — they depend
    only on ``F``, ``chol_Q``, ``H``, and ``chol_R``, not on the observation.
    This function extracts those constant fields once (along with the Kalman
    gain ``K`` and the innovation Cholesky ``chol_S``) by performing the same
    block-triangularization as :func:`associative_params_single` but without
    evaluating the observation-dependent terms.

    The returned :class:`SteadyStateFilterParams` can be passed to
    :func:`associative_params_single` or :func:`update` to skip the per-step
    QR decomposition and only recompute the cheap observation-dependent
    quantities ``b``, ``eta``, and ``ell`` at every step.

    This function is intended to be called **outside of JIT** as a one-off
    pre-computation.  The returned params can then be passed into a JIT-compiled
    filter call for all subsequent runs.

    Note:
        The ``K`` stored here is the *parallel-scan* gain, derived from
        ``chol_Q`` alone (see :func:`associative_params_single`).  When using
        :func:`update` in a sequential filter, supply a
        :class:`SteadyStateFilterParams` whose ``K`` and ``elem.U`` reflect the
        Riccati steady state instead.

    Args:
        F: State transition matrix, shape ``(nx, nx)``.
        chol_Q: Generalized Cholesky factor of the transition noise covariance,
            shape ``(nx, nx)``.
        H: Observation matrix, shape ``(ny, nx)``.
        chol_R: Generalized Cholesky factor of the observation noise covariance,
            shape ``(ny, ny)``.

    Returns:
        :class:`SteadyStateFilterParams` containing the constant ``A``, ``U``,
        ``Z``, gain ``K``, innovation Cholesky ``chol_S``, precomputed
        ``Psi11_inv``, and pre-padded information gain ``Z_filter``.
    """
    F = jnp.asarray(F)
    chol_Q = jnp.asarray(chol_Q)
    H = jnp.asarray(H)
    chol_R = jnp.asarray(chol_R)

    ny, nx = H.shape

    # Mirrors the block-triangularization in associative_params_single,
    # stopping before the observation-dependent b, eta, ell.
    Psi_ = jnp.block([[H @ chol_Q, chol_R], [chol_Q, jnp.zeros((nx, ny))]])
    Tria_Psi_ = tria(Psi_)

    chol_S = Tria_Psi_[:ny, :ny]  # innovation Cholesky
    Psi21 = Tria_Psi_[ny : ny + nx, :ny]
    U = Tria_Psi_[ny : ny + nx, ny:]  # posterior sqrt covariance

    Psi11_inv = solve_triangular(chol_S, jnp.eye(ny), lower=True)
    K = Psi21 @ Psi11_inv  # Kalman gain

    HF = H @ F
    A = F - K @ HF

    Z_filter = HF.T @ Psi11_inv.T  # (nx, ny); used for eta, before padding
    Z = Z_filter
    if nx > ny:
        Z = jnp.concatenate([Z, jnp.zeros((nx, nx - ny))], axis=1)
    else:
        Z = tria(Z)

    dummy_b = jnp.zeros(nx)
    dummy_eta = jnp.zeros(nx)
    dummy_ell = jnp.array(0.0)

    elem = FilterScanElement(A=A, b=dummy_b, U=U, eta=dummy_eta, Z=Z, ell=dummy_ell)
    return SteadyStateFilterParams(
        elem=elem, K=K, chol_S=chol_S, Psi11_inv=Psi11_inv, Z_filter=Z_filter
    )


def filtering_operator(
    elem_i: FilterScanElement, elem_j: FilterScanElement
) -> FilterScanElement:
    """Binary associative operator for the square root Kalman filter.

    Args:
        elem_i: Filter scan element for the previous time step.
        elem_j: Filter scan element for the current time step.

    Returns:
        FilterScanElement: The output of the associative operator applied to the input elements.
    """
    A1, b1, U1, eta1, Z1, ell1 = elem_i
    A2, b2, U2, eta2, Z2, ell2 = elem_j

    nx = Z2.shape[0]

    Xi = jnp.block([[U1.T @ Z2, jnp.eye(nx)], [Z2, jnp.zeros_like(A1)]])
    tria_xi = tria(Xi)
    Xi11 = tria_xi[:nx, :nx]
    Xi21 = tria_xi[nx : nx + nx, :nx]
    Xi22 = tria_xi[nx : nx + nx, nx:]

    tmp_1 = solve_triangular(Xi11, U1.T, lower=True).T
    D_inv = jnp.eye(nx) - tmp_1 @ Xi21.T
    tmp_2 = D_inv @ (b1 + U1 @ (U1.T @ eta2))

    A = A2 @ D_inv @ A1
    b = A2 @ tmp_2 + b2
    U = tria(jnp.concatenate([A2 @ tmp_1, U2], axis=1))
    eta = A1.T @ (D_inv.T @ (eta2 - Z2 @ (Z2.T @ b1))) + eta1
    Z = tria(jnp.concatenate([A1.T @ Xi22, Z1], axis=1))

    mu = cho_solve((U1, True), b1)
    t1 = b1 @ mu - (eta2 + mu) @ tmp_2

    # Derivation for O(nx) log-determinant computation of D_inv:
    # This is a long comment but I wanted to include the full derivation for clarity and future reference.
    # The key idea is to express D_inv in terms of the blocks of Xi and then apply Sylvester's determinant theorem
    # to compute its determinant efficiently.
    #
    # 1. Expand the blocks of Xi @ Xi.T:
    #    (Xi @ Xi.T)[1,1] = I + U1.T @ Z2 @ Z2.T @ U1
    #    (Xi @ Xi.T)[2,1] = Z2 @ Z2.T @ U1
    #
    # 2. Equate to the corresponding blocks of L @ L.T:
    #    Xi11 @ Xi11.T = I + U1.T @ Z2 @ Z2.T @ U1
    #    Xi21 @ Xi11.T = Z2 @ Z2.T @ U1
    #
    # 3. Expand D_inv using tmp_1 = Xi11^{-1} @ U1.T:
    #    D_inv = I - tmp_1.T @ Xi21.T
    #          = I - U1 @ Xi11^{-T} @ Xi21.T
    #
    # 4. Apply Sylvester's determinant theorem:
    #    det(D_inv) = det(I - Xi11^{-T} @ Xi21.T @ U1)
    #
    # 5. Multiply interior by Xi11^{-1} @ Xi11 and substitute block identities:
    #    Let P = U1.T @ Z2 @ Z2.T @ U1
    #    det(D_inv) = det(I - (Xi11 @ Xi11.T)^{-1} @ (Xi21 @ Xi11.T).T @ U1)
    #               = det(I - (I + P)^{-1} @ P)
    #               = det((I + P)^{-1})
    #               = 1 / det(Xi11 @ Xi11.T)
    #               = det(Xi11)^{-2}
    #
    # 6. Simplify the log-determinant term in the log-likelihood:
    #    0.5 * log(det(D_inv)) = -log(|det(Xi11)|)
    #
    # Since Xi11 is lower triangular, the log-determinant is the sum of the logs of its diagonal.
    # Replace `0.5 * jnp.linalg.slogdet(D_inv)[1]` with:
    # -jnp.sum(jnp.log(jnp.abs(jnp.diag(Xi11))))
    # ell = ell1 + ell2 - 0.5 * t1 + 0.5 * jnp.linalg.slogdet(D_inv)[1]
    ell = ell1 + ell2 - 0.5 * t1 - jnp.sum(jnp.log(jnp.abs(jnp.diag(Xi11))))

    return FilterScanElement(A, b, U, eta, Z, ell)
