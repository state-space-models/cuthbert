"""Covariance localization utilities for ensemble Kalman methods."""

from typing import NamedTuple

import jax.numpy as jnp

from cuthbertlib.types import Array, ArrayLike, ScalarArrayLike


class CovarianceTapers(NamedTuple):
    """Tapers for cross-covariance and marginal covariance matrices.

    Attributes:
        cross: Taper applied elementwise to the cross-covariance matrix ``C_xy``,
            with shape ``(x_dim, y_dim)``.
        marginal: Optional symmetric PSD taper applied elementwise to the marginal
            covariance matrix ``C_yy``, with shape ``(y_dim, y_dim)``. If omitted,
            only the cross-covariance is localized.
    """

    cross: Array
    marginal: Array | None = None


def gaspari_cohn(
    distances: ArrayLike,
    support_radius: ScalarArrayLike,
) -> Array:
    """Evaluates the compactly supported fifth-order Gaspari-Cohn taper.

    This implements Eq. (4.10) of Gaspari and Cohn (1999),
    https://doi.org/10.1002/qj.49712555417.

    ``support_radius`` is the full support radius: taper values are exactly zero
    where the absolute distance is greater than or equal to it.

    Args:
        distances: Distances at which to evaluate the taper.
        support_radius: Positive distance at which the taper reaches zero.

    Returns:
        Taper values with the broadcast shape of the inputs.
    """
    distances = jnp.abs(distances)
    q = 2 * distances / support_radius

    inner = 1 - 5 / 3 * q**2 + 5 / 8 * q**3 + 1 / 2 * q**4 - 1 / 4 * q**5

    # Avoid NaNs in computing the outer branch of the polynomial
    safe_q = jnp.where(q > 0, q, 1)
    # Factored version of Eq. (4.10) for numerical stability
    outer = (2 - q) ** 4 * (2 * q**2 + 4 * q - 1) / (24 * safe_q)

    within_support = jnp.where(q <= 1, inner, outer)
    return jnp.where(distances < support_radius, within_support, 0)
