"""Utility functions (Cholesky updating) for quadrature."""

import jax
import jax.numpy as jnp

from cuthbertlib.types import Array, ArrayLike

__all__ = ["cholesky_update_many"]


def cholesky_update_many(
    chol_init: ArrayLike, update_vectors: ArrayLike, multiplier: float
) -> Array:
    r"""Update the Cholesky decomposition of a matrix with multiple update vectors.

    In mathematical terms, we compute :math:`A + \sum_{i=1}^{n} \alpha v_i v_i^T`
    where :math:`A` is the original matrix, :math:`v_i` are the update vectors and
    :math:`\alpha` is the multiplier.

    Args:
        chol_init: Initial Cholesky decomposition of the matrix, :math:`A`.
        update_vectors: Update vectors, :math:`v_i`.
        multiplier: The multiplier, :math:`\alpha`.

    Returns:
        The updated Cholesky decomposition of the matrix.

    Notes:
        If the updated matrix does not correspond to a positive definite matrix, the
        function has undefined behaviour. It is the responsibility of the caller to
        ensure that the updated matrix is positive definite as we cannot check this at runtime.
    """

    def body(chol, update_vector):
        res = _cholesky_update(chol, update_vector, multiplier=multiplier)
        return res, None

    final_chol, _ = jax.lax.scan(body, jnp.asarray(chol_init), update_vectors)
    return final_chol


def _set_diagonal(x: Array, y: Array) -> Array:
    N, _ = x.shape
    i, j = jnp.diag_indices(N)
    return x.at[i, j].set(y)


def _set_triu(x: Array, val: ArrayLike) -> Array:
    N, _ = x.shape
    i = jnp.triu_indices(N, 1)
    return x.at[i].set(val)


def _cholesky_update(
    chol: ArrayLike, update_vector: ArrayLike, multiplier: float = 1.0
) -> Array:
    chol = jnp.asarray(chol)
    chol_diag = jnp.diag(chol)

    # The algorithm in [1] is implemented as a double for loop. We can treat
    # the inner loop in Algorithm 3.1 as a vector operation, and thus the
    # whole algorithm as a single for loop, and hence can use a `tf.scan`
    # on it.

    # We use for accumulation omega and b as defined in Algorithm 3.1, since
    # these are updated per iteration.

    def scan_body(carry, inp):
        _, _, omega, b = carry
        index, diagonal_member, col = inp
        omega_at_index = omega[..., index]

        # Line 4
        new_diagonal_member = jnp.sqrt(
            jnp.square(diagonal_member) + multiplier / b * jnp.square(omega_at_index)
        )
        # `scaling_factor` is the same as `gamma` on Line 5.
        scaling_factor = jnp.square(diagonal_member) * b + multiplier * jnp.square(
            omega_at_index
        )

        # The following updates are the same as the for loop in lines 6-8.
        omega = omega - (omega_at_index / diagonal_member)[..., None] * col
        new_col = new_diagonal_member[..., None] * (
            col / diagonal_member[..., None]
            + (multiplier * omega_at_index / scaling_factor)[..., None] * omega
        )
        b = b + multiplier * jnp.square(omega_at_index / diagonal_member)
        return (new_diagonal_member, new_col, omega, b), (
            new_diagonal_member,
            new_col,
            omega,
            b,
        )

    # We will scan over the columns.
    chol = chol.T

    _, (new_diag, new_chol, _, _) = jax.lax.scan(
        scan_body,
        (0.0, jnp.zeros_like(chol[0]), update_vector, 1.0),
        (jnp.arange(0, chol.shape[0]), chol_diag, chol),
    )

    new_chol = new_chol.T
    new_chol = _set_diagonal(new_chol, new_diag)
    new_chol = _set_triu(new_chol, 0.0)
    new_chol = jnp.where(jnp.isfinite(new_chol), new_chol, 0.0)
    return new_chol
