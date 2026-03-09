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
    # Derivation of the analytical JVP for the lower-triangularization operator:
    #
    # 1. The operation computes a lower triangular R such that:
    #    R @ R.T = A @ A.T
    #
    # 2. Taking the differential of both sides yields:
    #    dR @ R.T + R @ dR.T = dA @ A.T + A @ dA.T
    #
    # 3. Multiply from the left by R^{-1} and from the right by R^{-T}:
    #    R^{-1} @ dR + dR.T @ R^{-T} = R^{-1} @ dA @ A.T @ R^{-T} + R^{-1} @ A @ dA.T @ R^{-T}
    #
    # 4. Let Q be the orthogonal factor such that A = R @ Q.T.
    #    Substituting A @ R^{-T} = R @ Q.T @ R^{-T} = R @ (R^{-1} @ Q).T = R @ Q.T @ R^{-T} => A.T @ R^{-T} = Q
    #    and R^{-1} @ A = R^{-1} @ R @ Q.T = Q.T:
    #    R^{-1} @ dR + dR.T @ R^{-T} = R^{-1} @ dA @ Q + Q.T @ dA.T @ R^{-T}
    #
    # 5. Define K = R^{-1} @ dA @ Q. The right hand side becomes K + K.T:
    #    R^{-1} @ dR + (R^{-1} @ dR).T = K + K.T
    #
    # 6. Define dM = R^{-1} @ dR. Since R is lower triangular, its inverse and dR
    #    are lower triangular, meaning dM must also be lower triangular.
    #    dM + dM.T = K + K.T
    #
    # 7. Because dM is lower triangular, we can solve for it by taking the lower
    #    triangular part of K + K.T and subtracting the diagonal once to prevent
    #    double counting:
    #    dM = tril(K + K.T) - diag(K)
    #
    # 8. Recover the differential dR by left-multiplying by R:
    #    dR = R @ dM

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

    # Apply to get the tangent at R
    dR = R @ dM

    return R, dR
