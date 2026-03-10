"""Implements triangularization operator a matrix via QR decomposition."""

import jax
import jax.numpy as jnp

from cuthbertlib.types import Array


def _adj(x: Array) -> Array:
    """Conjugate transpose for batched matrices."""
    return jnp.swapaxes(x.conj(), -1, -2)


@jax.custom_jvp
def tria(A: Array) -> Array:
    """A triangularization operator using QR decomposition.

    Args:
        A: The matrix to triangularize.

    Returns:
        A lower triangular matrix R such that R @ R.T = A @ A.T.

    References:
        Paper: Arasaratnam and Haykin (2008): Square-Root Quadrature Kalman Filtering
            https://ieeexplore.ieee.org/document/4524036
    """
    _, R_qr = jnp.linalg.qr(_adj(A), mode="reduced")
    return _adj(R_qr)


@tria.defjvp
def _tria_jvp(primals, tangents):
    # Derivation of the exact analytical JVP for the lower-triangularization operator:
    #
    # 1. The operation computes a lower triangular R such that:
    #    R @ R.T = A @ A.T
    #
    # 2. Taking the differential of both sides yields the Gramian differential identity:
    #    dR @ R.T + R @ dR.T = dA @ A.T + A @ dA.T
    #
    # 3. For rank-deficient A, the exact lower-triangular tangent dR decomposes into
    #    column-space and null-space components: dR = dR_col + dR_null.
    #
    # 4. To find dR_col, multiply the identity from the left by R^\dagger (pseudoinverse)
    #    and from the right by R^{\dagger T}:
    #    R^\dagger @ dR_col + dR_col.T @ R^{\dagger T} = R^\dagger @ dA @ A.T @ R^{\dagger T} + R^\dagger @ A @ dA.T @ R^{\dagger T}
    #
    # 5. Let Q be the active subspace orthogonal factor such that A = R @ Q.T.
    #    Substituting A.T @ R^{\dagger T} = Q and R^\dagger @ A = Q.T:
    #    R^\dagger @ dR_col + dR_col.T @ R^{\dagger T} = R^\dagger @ dA @ Q + Q.T @ dA.T @ R^{\dagger T}
    #
    # 6. Define K = R^\dagger @ dA @ Q. The right hand side becomes K + K.T:
    #    R^\dagger @ dR_col + (R^\dagger @ dR_col).T = K + K.T
    #
    # 7. Define dM_col = R^\dagger @ dR_col. Solving for the lower-triangular dM_col:
    #    dM_col = tril(K + K.T) - diag(K)
    #
    # 8. Recover the column-space differential dR_col by left-multiplying by R:
    #    dR_col = R @ dM_col
    #
    # 9. To satisfy the orthogonal cross-terms for arbitrary perturbations outside
    #    the column space of A, add the null-space component projected via (I - R @ R^\dagger):
    #    dR_null = (I - R @ R^\dagger) @ dA @ Q
    #
    # 10. The complete, exact JVP is the sum of both components:
    #     dR = dR_col + dR_null

    (A,) = primals
    (dA,) = tangents

    A_T = jnp.swapaxes(A, -1, -2)
    Q, R_qr = jnp.linalg.qr(A_T, mode="reduced")

    R = jnp.swapaxes(R_qr, -1, -2)

    # Q has shape (..., M, N). A is (..., N, M).
    # A^T = Q R_qr => A = R_qr^T Q^T = R Q^T

    R_pinv = jnp.linalg.pinv(R)

    # K = R^{-1} dA Q
    # R can be degenerate so we use the pseudoinverse to ensure the JVP is well-defined everywhere,
    K = R_pinv @ dA @ Q
    K_T = jnp.swapaxes(K, -1, -2)

    # Solve for lower triangular perturbation dM + dM^T = K + K^T
    I = jnp.eye(K.shape[-1], dtype=K.dtype)
    dM = jnp.tril(K + K_T) - K * I

    # Compute the null-space part
    dR_null = (jnp.eye(R.shape[-2], dtype=R.dtype) - R @ R_pinv) @ dA @ Q

    # Apply to get the tangent at R
    dR = R @ dM + dR_null

    return R, dR
