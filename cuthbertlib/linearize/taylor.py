from typing import Callable, overload

import jax
from jax.typing import ArrayLike

from cuthbertlib.linearize.utils import symmetric_inv_sqrt
from cuthbertlib.types import Array, ArrayTree


@overload
def linearize_taylor(
    log_potential: Callable[[ArrayLike], Array],
    x: ArrayLike,
    has_aux: bool = False,
) -> tuple[Array, Array]: ...
@overload
def linearize_taylor(
    log_potential: Callable[[ArrayLike], tuple[Array, ArrayTree]],
    x: ArrayLike,
    has_aux: bool = True,
) -> tuple[Array, Array, ArrayTree]: ...


def linearize_taylor(
    log_potential: Callable[[ArrayLike], Array]
    | Callable[[ArrayLike], tuple[Array, ArrayTree]],
    x: ArrayLike,
    has_aux: bool = False,
) -> tuple[Array, Array] | tuple[Array, Array, ArrayTree]:
    """Linearize a log potential function around a given point using Taylor expansion.

    Unlike the other linearisation methods, this applies to a potential function
    with no required notion of observation $y$ or conditional dependence.

    Instead we have the linearisation

    log G(x) = -0.5 (x - m)^T (L L^T)^{-1} (x - m)

    Args:
        log_likelihood: A callable that returns a non-negative scalar. Does not need
            to be a normalized probability density in its input.
        x: The point to linearize around.
        has_aux: Whether the log_potential function returns an auxiliary value.

    Returns:
        Linearized mean and cholesky factor of the covariance matrix.
            As well as the auxiliary value if `has_aux` is True.
    """

    g_and_maybe_aux = jax.grad(log_potential, has_aux=has_aux)(x)
    prec_and_maybe_aux = jax.hessian(log_potential, has_aux=has_aux)(x)

    g, aux = g_and_maybe_aux if has_aux else (g_and_maybe_aux, None)
    prec = -prec_and_maybe_aux[0] if has_aux else -prec_and_maybe_aux

    L = symmetric_inv_sqrt(prec)
    m = x + L @ L.T @ g
    return (m, L, aux) if has_aux else (m, L)
