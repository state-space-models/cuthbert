"""Covariance localization utilities for ensemble Kalman methods."""

import jax.numpy as jnp

from cuthbertlib.linalg import tria
from cuthbertlib.types import Array, ArrayLike, ScalarArrayLike


def construct_tapered_chol_innovation_covariance(
    Y: Array,
    chol_taper: Array,
    chol_R: Array,
) -> Array:
    """Construct a tapered innovation covariance factor without forming it densely.

    If ``taper = chol_taper @ chol_taper.T`` and ``Y`` denotes the normalized
    observation deviations, the returned generalized Cholesky factor ``chol_S``
    satisfies ``chol_S @ chol_S.T = taper * (Y @ Y.T) + R``.

    Args:
        Y: Observation deviations transposed and divided by the square root of one
            less than the ensemble size, shape (y_dim, n_particles).
        chol_taper: Factor of a positive-semidefinite observation-space taper,
            shape (y_dim, y_dim).
        chol_R: Cholesky factor of the observation noise covariance, shape
            (y_dim, y_dim).

    Returns:
        Generalized Cholesky factor of the complete tapered innovation covariance,
        shape (y_dim, y_dim).
    """
    y_dim = Y.shape[0]
    Y_tilde = (chol_taper[:, :, None] * Y[:, None, :]).reshape(y_dim, -1)
    return tria(jnp.concatenate([Y_tilde, chol_R], axis=1))


def gaussian(
    distances: ArrayLike,
    length_scale: ScalarArrayLike,
) -> Array:
    r"""Evaluates a Gaussian covariance taper.

    The taper is the squared-exponential correlation function
    $\rho(d; \ell) = \exp(-\frac{1}{2}(d / \ell)^2)$. This taper
    has infinite support and is differentiable with respect
    to the length scale.

    Args:
        distances: Distances at which to evaluate the taper.
        length_scale: Positive characteristic distance of the taper.

    Returns:
        Taper values with the broadcast shape of the inputs.
    """
    scaled_distances = jnp.asarray(distances) / length_scale
    return jnp.exp(-0.5 * jnp.square(scaled_distances))


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
