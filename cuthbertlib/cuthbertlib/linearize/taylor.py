"""Implements Taylor-like linearization."""

from typing import Callable, overload

import jax
from jax import numpy as jnp
from jax.typing import ArrayLike

from cuthbertlib.linalg import symmetric_inv_sqrt
from cuthbertlib.types import Array, ArrayTree


@overload
def linearize_taylor(
    log_potential: Callable[[ArrayLike], Array],
    x: ArrayLike,
    has_aux: bool = False,
    rtol: float | None = None,
    ignore_nan_dims: bool = False,
) -> tuple[Array, Array]: ...
@overload
def linearize_taylor(
    log_potential: Callable[[ArrayLike], tuple[Array, ArrayTree]],
    x: ArrayLike,
    has_aux: bool = True,
    rtol: float | None = None,
    ignore_nan_dims: bool = False,
) -> tuple[Array, Array, ArrayTree]: ...


def linearize_taylor(
    log_potential: Callable[[ArrayLike], Array]
    | Callable[[ArrayLike], tuple[Array, ArrayTree]],
    x: ArrayLike,
    has_aux: bool = False,
    rtol: float | None = None,
    ignore_nan_dims: bool = False,
) -> tuple[Array, Array] | tuple[Array, Array, ArrayTree]:
    r"""Linearizes a log potential function around a given point using Taylor expansion.

    Unlike the other linearization methods, this applies to a potential function
    with no required notion of observation $y$ or conditional dependence.

    Instead we have the linearization

    $$
    \log G(x) = -\frac{1}{2} (x - m)^\top (L L^\top)^{-1} (x - m).
    $$

    Args:
        log_potential: A callable that returns a non-negative scalar. Does not need
            to be a normalized probability density in its input.
        x: The point to linearize around.
        has_aux: Whether `log_potential` returns an auxiliary value.
        rtol: The relative tolerance for the singular values of the precision matrix
            when passed to `symmetric_inv_sqrt`.
            Cutoff for small singular values; singular values smaller than
            `rtol * largest_singular_value` are treated as zero.
            The default is determined based on the floating point precision of the dtype.
            See https://docs.jax.dev/en/latest/_autosummary/jax.numpy.linalg.pinv.html.
        ignore_nan_dims: Whether to treat dimensions with NaN on the diagonal of the
            precision matrix as missing and ignore all rows and columns associated with
            them.

    Returns:
        Linearized mean and Cholesky factor of the covariance matrix.
            The auxiliary value is also returned if `has_aux` is `True`.
    """
    g_and_maybe_aux = jax.grad(log_potential, has_aux=has_aux)(x)
    prec_and_maybe_aux = jax.hessian(log_potential, has_aux=has_aux)(x)

    g, aux = g_and_maybe_aux if has_aux else (g_and_maybe_aux, None)
    prec = -prec_and_maybe_aux[0] if has_aux else -prec_and_maybe_aux

    L = symmetric_inv_sqrt(prec, rtol=rtol, ignore_nan_dims=ignore_nan_dims)

    # Change nans on diag to zeros for L @ L.T @ g, still retain nans on diag for L for bookkeeping
    # If ignore_nan_dims, change all rows and columns with nans on the diagonal to 0
    L_diag = jnp.diag(L)
    nan_mask = jnp.isnan(L_diag) * ignore_nan_dims
    L_temp = jnp.where(nan_mask[:, None] | nan_mask[None, :], 0.0, L)
    m = x + L_temp @ L_temp.T @ g
    return (m, L, aux) if has_aux else (m, L)
