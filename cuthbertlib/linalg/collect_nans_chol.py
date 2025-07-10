from typing import Any

from jax import numpy as jnp
from jax import tree

from cuthbertlib.linalg.tria import tria
from cuthbertlib.types import Array, ArrayLike


def set_to_zero(flag: ArrayLike, x: ArrayLike) -> Array:
    x = jnp.asarray(x)
    broadcast_flag = jnp.expand_dims(flag, list(range(1, x.ndim)))
    return jnp.where(broadcast_flag, 0.0, x)


def collect_nans_chol(flag: ArrayLike, chol: ArrayLike, *rest: Any) -> Any:
    """
    Converts a generalized Cholesky factor of a covariance matrix with NaNs
    into an ordered generalized Cholesky factor with NaNs rows and columns
    moved to the end with diagonal elements set to 1.

    Also reorders the rest of the arguments in the same way along the first axis
    and sets to 0 for dimensions where flag is True.

    Example behavior:
    ```
    flag = jnp.array([False, True, False, True])
    new_flag, new_chol, new_mean = collect_nans_chol(flag, chol, mean)
    ```

    Args:
        flag: Array, boolean array indicating which entries are NaN
            True for NaN entries, False for valid
        chol: Array, Cholesky factor of the covariance matrix
        rest: Any, rest of the arguments to be reordered in the same way
            along the first axis

    Returns:
        flag, chol and rest reordered so that valid entries are first and NaNs are last.
            Diagonal elements of chol are set to 1/√2π so that normalization is correct
    """

    flag = jnp.asarray(flag)
    chol = jnp.asarray(chol)

    # TODO: Can we support batching? I.e. when `chol` is a batch of Cholesky factors,
    # possibly with multiple leading dimensions

    if flag.ndim > 1 or chol.ndim > 2:
        raise ValueError("Batched flag or chol not supported yet")

    if not flag.shape:
        return (
            flag,
            set_to_zero(flag, chol),
            *tree.map(lambda x: set_to_zero(flag, x), rest),
        )

    if chol.size == 1:
        chol *= jnp.ones_like(flag, dtype=chol.dtype)

    # group the NaN entries together
    argsort = jnp.argsort(flag, stable=True)

    if chol.ndim == 1:
        chol = chol[argsort]
        flag = flag[argsort]
        chol = jnp.where(flag, 1 / jnp.sqrt(2 * jnp.pi), chol)

    else:
        chol = jnp.where(flag[:, None], 0.0, chol)
        chol = chol[argsort]
        # compute the tria of the covariance matrix with NaNs set to 0
        chol = tria(chol)

        flag = flag[argsort]

        # set the diagonal of chol_cov to 1/√2π where nans were present so that normalization is correct
        diag_chol = jnp.diag(chol)
        diag_chol = jnp.where(flag, 1 / jnp.sqrt(2 * jnp.pi), diag_chol)
        diag_indices = jnp.diag_indices_from(chol)
        chol = chol.at[diag_indices].set(diag_chol)

    rest = tree.map(lambda x: x[argsort], rest)
    rest = tree.map(lambda x: set_to_zero(flag, x), rest)

    return flag, chol, *rest
