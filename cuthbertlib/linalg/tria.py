"""Implements triangularization operator a matrix via QR decomposition."""

from functools import partial

import jax
import jax.numpy as jnp
from jax.scipy.linalg import solve_triangular

from cuthbertlib.types import Array


def _adj(x: Array) -> Array:
    """Conjugate transpose for batched matrices."""
    return jnp.swapaxes(x.conj(), -1, -2)


def _t(x: Array) -> Array:
    """Transpose for batched matrices."""
    return jnp.swapaxes(x, -1, -2)


@partial(jax.custom_jvp, nondiff_argnums=(1,))
def tria(A: Array, n_lower: int = 0) -> Array:
    r"""A triangularization operator using QR decomposition.

    The primal output does not depend on `n_lower`. The argument only selects the
    tangent returned in forward mode (and hence the cotangent in reverse mode),
    which must match how the output is used downstream:

    - `n_lower = 0`: the output $L$ is only used through products such as
      $F L$ or $L L^\top$, as for the predicted covariance factor.
    - `n_lower = k > 0`: the leading $k \times k$ block $L_{11}$ is used as a
      triangular factor (triangular solves, log-determinants from its diagonal)
      and the upper-right block is assumed to be zero, as for the innovation
      factor in the Kalman update. $L_{11} L_{11}^\top$ must be positive definite.

    In both cases, the tangent $dL$ satisfies
    $dL L^\top + L dL^\top = dA A^\top + A dA^\top$ at any rank of $A$, so that
    gradients of quantities that depend on $L$ only through $L L^\top$ are exact.

    Args:
        A: The matrix to triangularize, of shape (..., n, m).
        n_lower: Number of leading rows and columns of the output used as a
            lower-triangular factor downstream.

    Returns:
        A lower triangular matrix L such that L @ L.T = A @ A.T.

    References:
        Paper: Arasaratnam and Haykin (2008): Square-Root Quadrature Kalman Filtering
            https://ieeexplore.ieee.org/document/4524036
    """
    _, R_qr = jnp.linalg.qr(_adj(A), mode="reduced")
    return _adj(R_qr)


def _phi(X: Array) -> Array:
    """Lower triangle of a symmetric matrix with its diagonal halved."""
    Id = jnp.eye(X.shape[-1], dtype=X.dtype)
    return jnp.tril(X) - 0.5 * X * Id


def _factor_tangent(L: Array, G: Array) -> Array:
    """Tangent D with D L^T + L D^T = G L^T + L G^T, valid at any rank of L.

    The column-space part uses the pseudoinverse of L and the null-space part
    projects G onto the orthogonal complement of the column space of L.
    """
    L_pinv = jnp.linalg.pinv(L)
    K = L_pinv @ G
    Id = jnp.eye(L.shape[-2], dtype=L.dtype)
    return L @ _phi(K + _t(K)) + (Id - L @ L_pinv) @ G


@tria.defjvp
def _tria_jvp(n_lower, primals, tangents):
    # With A = L Q^T (thin QR of A^T) and G = dA Q, the Gramian differential is
    # dA A^T + A dA^T = G L^T + L G^T. Partition L, G and dL conformably with
    # n_lower = k. The tangent below has dL12 = 0 and dL11 lower triangular:
    #   K11 = L11^{-1} G11,  K12 = L11^{-1} G12,
    #   dL11 = L11 phi(K11 + K11^T)              (classical QR/Cholesky rule),
    #   W = K11 - phi(K11 + K11^T)              (skew-symmetric),
    #   dL21 = G21 - L21 W + L22 K12^T,
    #   dL22 = _factor_tangent(L22, G22 - L21 K12).
    # Each block of dL L^T + L dL^T then equals the corresponding block of
    # G L^T + L G^T. For k = 0 this reduces to _factor_tangent(L, G), and for
    # k = n with L invertible to the classical rule.
    (A,) = primals
    (dA,) = tangents

    Q, R_qr = jnp.linalg.qr(_t(A), mode="reduced")
    L = _t(R_qr)
    G = dA @ Q

    n = L.shape[-1]
    k = n_lower
    if k == 0:
        return L, _factor_tangent(L, G)

    L11 = L[..., :k, :k]
    K11 = solve_triangular(L11, G[..., :k, :k], lower=True)
    Phi11 = _phi(K11 + _t(K11))
    dL11 = L11 @ Phi11
    if k >= n:
        return L, dL11

    L21, L22 = L[..., k:, :k], L[..., k:, k:]
    G12, G21, G22 = G[..., :k, k:], G[..., k:, :k], G[..., k:, k:]
    K12 = solve_triangular(L11, G12, lower=True)
    W = K11 - Phi11
    dL21 = G21 - L21 @ W + L22 @ _t(K12)
    dL22 = _factor_tangent(L22, G22 - L21 @ K12)

    top = jnp.concatenate([dL11, jnp.zeros_like(G12)], axis=-1)
    bottom = jnp.concatenate([dL21, dL22], axis=-1)
    return L, jnp.concatenate([top, bottom], axis=-2)
