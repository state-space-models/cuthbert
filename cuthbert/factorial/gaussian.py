"""Factorial utilities for Kalman states."""

from typing import TypeVar

from jax import numpy as jnp
from jax import tree
from jax.scipy.linalg import block_diag

from cuthbert.factorial.types import Factorializer, GetFactorialIndices
from cuthbert.gaussian.kalman import KalmanFilterState
from cuthbert.gaussian.types import LinearizedKalmanFilterState
from cuthbertlib.linalg import block_marginal_sqrt_cov
from cuthbertlib.types import Array, ArrayLike

KalmanState = TypeVar("KalmanState", KalmanFilterState, LinearizedKalmanFilterState)


def build_factorializer(
    get_factorial_indices: GetFactorialIndices,
) -> Factorializer:
    """Build a factorializer for Kalman states.

    Args:
        get_factorial_indices: Function to extract the factorial indices
            from model inputs.

    Returns:
        Factorializer object for Kalman states with functions to extract and join
        the relevant factors and marginalize and insert the updated factors.
    """
    return Factorializer(
        get_factorial_indices=get_factorial_indices,
        extract=extract,
        join=join,
        marginalize=marginalize,
        insert=insert,
    )


def extract(factorial_state: KalmanState, factorial_inds: ArrayLike) -> KalmanState:
    """Extract the relevant factors from a factorial Kalman state.

    Single dimensional arrays will be treated as scalars e.g. log normalizing constants.
        This means univariate problems still need to be stored with a dimension array
        (e.g. means with shape (F, 1) and chol_covs with shape (F, 1, 1)).
    Multidimensional arrays will be treated as arrays with shape (F, *).
        In this case the factorial_inds indices will be extracted from the first
        dimension and then the remaining dimensions will be preserved.

    Here F is the number of factors and d is the dimension of the state.

    Args:
        factorial_state: Factorial Kalman state storing means and chol_covs
            with shape (F, d) and (F, d, d) respectively.
        factorial_inds: Indices of the factors to extract. Integer array.

    Returns:
        Factorial Kalman state storing means and chol_covs
            with shape (len(factorial_inds), d) and (len(factorial_inds), d, d).
    """
    factorial_inds = jnp.asarray(factorial_inds)
    new_elem = tree.map(lambda x: _extract_arr(x, factorial_inds), factorial_state.elem)
    new_state = factorial_state._replace(elem=new_elem)

    if isinstance(factorial_state, LinearizedKalmanFilterState):
        new_mean_prev = _extract_arr(factorial_state.mean_prev, factorial_inds)
        new_state = new_state._replace(mean_prev=new_mean_prev)

    return new_state


def _extract_arr(arr: Array, factorial_inds: Array) -> Array:
    if arr.ndim == 0 or arr.ndim == 1:
        return arr
    else:
        return arr[factorial_inds]


def join(local_factorial_state: KalmanState) -> KalmanState:
    """Convert a factorial Kalman state into a joint local Kalman state.

    Single dimensional arrays will be treated as scalars e.g. log normalizing constants.
        This means univariate problems still need to be stored with a dimension array
        (e.g. means with shape (F, 1) and chol_covs with shape (F, 1, 1)).
    Two dimensional arrays will be treated as means with shape (F, d).
        In this case the factorial_inds indices will be extracted from the first
        dimension and then stacked into a single array.
    Three dimensional arrays will be treated as chol_covs with shape (F, d, d).
        In this case the factorial_inds indices will be extracted from the first
        dimension and then stacked into a block diagonal array.

    Here F is the number of factors and d is the dimension of the state.

    Args:
        local_factorial_state: Factorial Kalman state storing means and chol_covs
            with shape (F, d) and (F, d, d) respectively.
        factorial_inds: Indices of the factors to extract. Integer array.

    Returns:
        Joint local Kalman state with no factorial index dimension.
    """
    new_elem = tree.map(_join_arr, local_factorial_state.elem)
    new_state = local_factorial_state._replace(elem=new_elem)

    if isinstance(local_factorial_state, LinearizedKalmanFilterState):
        new_mean_prev = _join_arr(local_factorial_state.mean_prev)
        new_state = new_state._replace(mean_prev=new_mean_prev)

    return new_state


def _join_arr(arr: Array) -> Array:
    if arr.ndim == 0 or arr.ndim == 1:
        return arr
    elif arr.ndim == 2:  # means
        return arr.reshape(-1)
    elif arr.ndim == 3:  # chol_covs
        return block_diag(*arr)
    else:
        raise ValueError(f"Array must be 3D or lower, got {arr.ndim}D")


def marginalize(
    local_state: KalmanState,
    num_factors: int,
) -> KalmanState:
    """Marginalize a joint local Kalman state into a factorial Kalman state.

    Args:
        local_state: Joint local Kalman state to marginalize and insert.
            With means and chol_covs with shape (d * len(factorial_inds),)
            and (d * len(factorial_inds), d * len(factorial_inds)) respectively.
        num_factors: Number of factors to marginalize out. Integer.

    Returns:
        Joint local Kalman state with no factorial index dimension.
    """
    new_elem = tree.map(
        lambda loc: _marginalize_arr(loc, num_factors),
        local_state.elem,
    )
    new_state = local_state._replace(elem=new_elem)
    if isinstance(local_state, LinearizedKalmanFilterState):
        new_mean_prev = _marginalize_arr(local_state.mean_prev, num_factors)
        new_state = new_state._replace(mean_prev=new_mean_prev)

    return new_state


def _marginalize_arr(arr: Array, num_factors: int) -> Array:
    if arr.ndim == 0:
        return arr
    elif arr.ndim == 1:  # means
        return arr.reshape(num_factors, -1)
    elif arr.ndim == 2:  # chol_covs
        local_dim = arr.shape[-1] // num_factors
        return block_marginal_sqrt_cov(arr, local_dim)
    else:
        raise ValueError(f"Array must be 1D (means) or 2D (chol_covs), got {arr.ndim}D")


def insert(
    local_factorial_state: KalmanState,
    factorial_state: KalmanState,
    factorial_inds: ArrayLike,
) -> KalmanState:
    """Insert a local factorial Kalman state into a factorial Kalman state.

    Single dimensional arrays will be treated as scalars e.g. log normalizing constants.
        This means univariate problems still need to be stored with a dimension array
        (e.g. means with shape (F, 1) and chol_covs with shape (F, 1, 1)).
    Multidimensional arrays will be treated as arrays with shape (F, *).
        In this case the factorial_inds indices will be inserted into the first
        dimension and then the remaining dimensions will be preserved.

    Here F is the number of factors and d is the dimension of the state.

    Args:
        local_factorial_state: Joint local Kalman state to marginalize and insert.
            With means and chol_covs with shape (len(factorial_inds), d)
            and (len(factorial_inds), d, d) respectively.
        factorial_state: Factorial Kalman state storing means and chol_covs
            with shape (F, d) and (F, d, d) respectively.
        factorial_inds: Indices of the factors to insert. Integer array.

    Returns:
        Joint local Kalman state with no factorial index dimension.
    """
    factorial_inds = jnp.asarray(factorial_inds)
    new_elem = tree.map(
        lambda loc, glob: _insert_arr(loc, glob, factorial_inds),
        local_factorial_state.elem,
        factorial_state.elem,
    )
    new_state = factorial_state._replace(elem=new_elem)

    if isinstance(local_factorial_state, LinearizedKalmanFilterState) and isinstance(
        factorial_state, LinearizedKalmanFilterState
    ):
        new_mean_prev = _insert_arr(
            local_factorial_state.mean_prev, factorial_state.mean_prev, factorial_inds
        )
        new_state = new_state._replace(mean_prev=new_mean_prev)

    return new_state


def _insert_arr(
    local_factorial_arr: Array, factorial_arr: Array, factorial_inds: Array
) -> Array:
    if local_factorial_arr.ndim == 0 or local_factorial_arr.ndim == 1:
        return local_factorial_arr
    else:
        return factorial_arr.at[factorial_inds].set(local_factorial_arr)
