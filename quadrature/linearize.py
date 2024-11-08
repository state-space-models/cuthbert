import jax
import jax.numpy as jnp
from jax.scipy.linalg import cho_solve

from quadrature.common import SigmaPoints, Quadrature
from quadrature.utils import tria, cholesky_update_many


def conditional_moments(mean_fn, cov_fn, m, cov, quadrature, mode="covariance"):
    """
    Linearize the conditional mean and covariance of a Gaussian distribution.

    For a given Gaussian distribution :math:`p(x) = N(x | m, P)`,
    and conditional mean `E[Y!x] and covariance `C[Y|x]1 functions for a random variable
    `Y`,  this function computes the linearization of the conditional mean and covariance of `Y` given `x`.
    i.e., it computes approximations to the following quantities:
    Y = A X + b + \epsilon using the sigma points method given by get_sigma_points.

    Parameters
    ----------
    mean_fn: callable
        The mean function E[Y|x] = mean_fn(x)
    cov_fn: callable
        The covariance function C[Y|x] = cov_fn(x)
    m: array_like
        The mean of the Gaussian distribution
    cov: array_like
        The covariance of the Gaussian distribution
    quadrature: Quadrature
        The quadrature object with the weights and sigma-points
    mode: str, optional
        The mode of the covariance. Default is 'covariance', which means that cov
        and cov_fn are given as covariance matrices. Otherwise, then
        the Cholesky factor of the covariances are given.

    Returns
    -------
    A, b, Q: array_like
        The linearized model parameters Y = A X + b + N(0, Q)
        Q is either given as a full covariance matrix or as a square root factor depending on the `mode`.

    """
    if mode == "covariance":
        chol = jnp.linalg.cholesky(cov)
    else:
        chol = cov
    x_pts: SigmaPoints = quadrature.get_sigma_points(m, chol)

    f_pts = SigmaPoints(jax.vmap(mean_fn)(x_pts.points), x_pts.wm, x_pts.wc)
    Psi_x = x_pts.covariance(f_pts)

    A = cho_solve((chol, True), Psi_x).T
    b = f_pts.mean - A @ m
    if mode != "covariance":
        # This can probably be abstracted better.
        sqrt_Phi = f_pts.sqrt

        chol_pts = jax.vmap(cov_fn)(x_pts.points)
        temp = jnp.sqrt(x_pts.wc[:, None, None]) * chol_pts

        # concatenate the blocks properly, it's a bit urk, but what can you do...
        temp = jnp.transpose(temp, [1, 0, 2]).reshape(temp.shape[1], -1)
        chol_Q = tria(jnp.concatenate([sqrt_Phi, temp], axis=1))
        chol_Q = cholesky_update_many(chol_Q, (A @ chol).T, -1.0)
        return A, b, chol_Q

    V_pts = jax.vmap(cov_fn)(x_pts.points)
    v_f = jnp.sum(x_pts.wc[:, None, None] * V_pts, 0)

    Phi = f_pts.covariance()
    Q = Phi + v_f - A @ cov @ A.T

    return A, b, 0.5 * (Q + Q.T)


def functional(fn, S, m, cov, quadrature: Quadrature, mode="covariance"):
    """
    Linearize a non-linear function of a Gaussian distribution.

    For a given Gaussian distribution :math:`p(x) = N(x | m, P)`,
    and Y = f(X) + epsilon, where epsilon is a zero-mean Gaussian noise with covariance S,
    this function computes approximations to the following quantities:
    Y = A X + b + \epsilon using the sigma points method given by get_sigma_points.

    Parameters
    ----------
    fn: callable
        The function Y = f(X) + N(0, S).
        Because the function is linearized, the function should be vectorized.
    S: array_like
        The covariance of the noise
    m: array_like
        The mean of the Gaussian distribution
    cov: array_like
        The covariance of the Gaussian distribution
    quadrature: Quadrature
        The quadrature object with the weights and sigma-points
    mode: str, optional
        The mode of the covariance. Default is 'covariance', which means that cov
        and cov_fn are given as covariance matrices. Otherwise, then
        the Cholesky factor of the covariances are given.

    Returns
    -------
    A, b, Q: array_like
        The linearized model parameters Y = A X + b + N(0, Q)
        Q is either given as a full covariance matrix or as a square root factor depending on the `mode`.

    Notes
    ------
    We do not support non-additive noise in this method.
    If you have a non-additive noise, you should use the `conditional_moments` or
    the taylor linearization method.
    Another solution is to form the covariance function using the quadrature method itself:
    For example, if you have a function `f(x, q)`, where `q` is a 0 mean random variable with covariance `S`,
    you can form the mean and covariance function as follows:
    ```
    def linearize_q_part(x):
        n_dim = S.shape[0]
        m_q = jnp.zeros(n_dim)
        A, b, Q = functional(lambda x: f(x, q_sigma_points.points), 0. * S, m_q, S, quadrature, mode)
        return A, b, Q

    def cov_fn(x):
        A, b, Q = linearize_q_part(x)
        return Q + A @ S @ A.T

    def mean_fn(x):
        A, b, Q = linearize_q_part(x)
        m_q = jnp.zeros(n_dim)
        return b + f(x, m_q)
        ```

        This technique is a bit wasteful due to our current separation of duties between the mean and covariance functions,
        but as we develop the library further, we will provide a more elegant solution.
    """

    # make the equivalent conditional_moments model
    def mean_fn(x):
        return fn(x)

    def cov_fn(x):
        return S

    return conditional_moments(mean_fn, cov_fn, m, cov, quadrature, mode)
