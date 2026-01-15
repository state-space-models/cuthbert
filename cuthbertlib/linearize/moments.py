"""Implements moment-based linearization."""

from typing import Callable, cast, overload

import jax
from jax.typing import ArrayLike

from cuthbertlib.types import Array, ArrayTree

MeanAndCholCovFunc = Callable[[ArrayLike], tuple[Array, Array]]
MeanAndCholCovFuncAux = Callable[[ArrayLike], tuple[Array, Array, ArrayTree]]


@overload
def linearize_moments(
    mean_and_chol_cov_function: MeanAndCholCovFunc,
    x: ArrayLike,
    has_aux: bool = False,
) -> tuple[Array, Array, Array]: ...
@overload
def linearize_moments(
    mean_and_chol_cov_function: MeanAndCholCovFuncAux,
    x: ArrayLike,
    has_aux: bool = True,
) -> tuple[Array, Array, Array, ArrayTree]: ...


def linearize_moments(
    mean_and_chol_cov_function: MeanAndCholCovFunc | MeanAndCholCovFuncAux,
    x: ArrayLike,
    has_aux: bool = False,
) -> tuple[Array, Array, Array] | tuple[Array, Array, Array, ArrayTree]:
    r"""Linearizes conditional mean and chol_cov functions into a linear-Gaussian form.

    Takes a function `mean_and_chol_cov_function(x)` that returns the
    conditional mean and Cholesky factor of the covariance matrix of the distribution
    $p(y \mid x)$ for a given input `x`.

    Returns $(H, d, L)$ defining a linear-Gaussian approximation to the conditional
    distribution $p(y \mid x) \approx N(y \mid H x + d, L L^\top)$.

    `mean_and_chol_cov_function` has the following signature with `has_aux` = False:
    ```
    m, chol = mean_and_chol_cov_function(x)
    ```
    or with `has_aux` = True:
    ```
    m, chol, aux = mean_and_chol_cov_function(x)
    ```

    Args:
        mean_and_chol_cov_function: A callable that returns the conditional mean and
            Cholesky factor of the covariance matrix of the distribution for a given
            input.
        x: The point to linearize around.
        has_aux: Whether `mean_and_chol_cov_function` returns an auxiliary value.

    Returns:
        Linearized matrix, shift, and Cholesky factor of the covariance matrix.
            The auxiliary value is also returned if `has_aux` is `True`.

    References:
        - [sqrt-parallel-smoothers](https://github.com/EEA-sensors/sqrt-parallel-smoothers/blob/main/parsmooth/linearization/_extended.py)
    """
    if has_aux:
        mean_and_chol_cov_function = cast(
            MeanAndCholCovFuncAux, mean_and_chol_cov_function
        )

        def mean_and_chol_cov_function_wrapper_aux(
            x: ArrayLike,
        ) -> tuple[Array, tuple[Array, Array, ArrayTree]]:
            mean, chol_cov, aux = mean_and_chol_cov_function(x)
            return mean, (mean, chol_cov, aux)

        F, (m, *extra) = jax.jacfwd(
            mean_and_chol_cov_function_wrapper_aux, has_aux=True
        )(x)

    else:
        mean_and_chol_cov_function = cast(
            MeanAndCholCovFunc, mean_and_chol_cov_function
        )

        def mean_and_chol_cov_function_wrapper(
            x: ArrayLike,
        ) -> tuple[Array, tuple[Array, Array]]:
            mean, chol_cov = mean_and_chol_cov_function(x)
            return mean, (mean, chol_cov)

        F, (m, *extra) = jax.jacfwd(mean_and_chol_cov_function_wrapper, has_aux=True)(x)

    b = m - F @ x
    return F, b, *extra
