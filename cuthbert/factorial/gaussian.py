"""Factorial utilities for Kalman states."""

from typing import TypeVar

from jax import tree, numpy as jnp
from jax.scipy.linalg import block_diag

from cuthbertlib.linalg import block_marginal_sqrt_cov
from cuthbert.gaussian.kalman import KalmanFilterState
from cuthbert.gaussian.types import LinearizedKalmanFilterState
from cuthbertlib.types import Array, ArrayLike
from cuthbert.factorial.types import Factorializer, GetFactorialIndices


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
        extract_and_join=extract_and_join,
        marginalize_and_insert=marginalize_and_insert,
    )


def extract_and_join(
    factorial_state: KalmanState, factorial_inds: ArrayLike
) -> KalmanState:
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
        factorial_state: Factorial Kalman state storing means and chol_covs
            with shape (F, d) and (F, d, d) respectively.
        factorial_inds: Indices of the factors to extract. Integer array.

    Returns:
        Joint local Kalman state with no factorial index dimension.
    """
    factorial_inds = jnp.asarray(factorial_inds)
    return tree.map(lambda x: _extract_and_join_arr(x, factorial_inds), factorial_state)


def _extract_and_join_arr(arr: Array, factorial_inds: Array) -> Array:
    if arr.ndim == 0 or arr.ndim == 1:
        return arr
    elif arr.ndim == 2:
        return _extract_and_join_means(arr, factorial_inds)
    elif arr.ndim == 3:
        return _extract_and_join_chol_covs(arr, factorial_inds)
    else:
        raise ValueError(f"Array must be 3D or lower, got {arr.ndim}D")


def _extract_and_join_means(means: Array, factorial_inds: Array) -> Array:
    return means[factorial_inds].reshape(-1)


def _extract_and_join_chol_covs(chol_covs: Array, factorial_inds: Array) -> Array:
    selected_chol_covs = chol_covs[factorial_inds]
    return block_diag(*selected_chol_covs)


def marginalize_and_insert(
    local_state: KalmanState,
    factorial_state: KalmanState,
    factorial_inds: ArrayLike,
) -> KalmanState:
    """Marginalize and insert a joint local Kalman state into a factorial Kalman state.

    Single dimensional arrays will be treated as scalars e.g. log normalizing constants.
        This means univariate problems still need to be stored with a dimension array
        (e.g. means with shape (F, 1) and chol_covs with shape (F, 1, 1)).
    Two dimensional arrays will be treated as means with shape (F, d).
        In this case the dimension d will be inferred and then the array split into
        len(factorial_inds) arrays of shape (d,) then inserted into the factorial array
        at the factorial_inds indices.
    Three dimensional arrays will be treated as chol_covs with shape (F, d, d).
        In this case the dimension d will be inferred and then the array split into
        len(factorial_inds) arrays of shape (d, d) (noting that the marginal_sqrt_cov
        function is called to preserve the lower triangular structure) then inserted
        into the factorial array at the factorial_inds indices.

    Here F is the number of factors and d is the dimension of the state.

    Args:
        local_state: Joint local Kalman state to marginalize and insert.
            With means and chol_covs with shape (d * len(factorial_inds),)
            and (d * len(factorial_inds), d * len(factorial_inds)) respectively.
        factorial_state: Factorial Kalman state storing means and chol_covs
            with shape (F, d) and (F, d, d) respectively.\
        factorial_inds: Indices of the factors to insert. Integer array.

    Returns:
        Joint local Kalman state with no factorial index dimension.
    """
    factorial_inds = jnp.asarray(factorial_inds)
    return tree.map(
        lambda loc, fac: _marginalize_and_insert_arr(loc, fac, factorial_inds),
        local_state,
        factorial_state,
    )


def _marginalize_and_insert_arr(
    local_arr: Array, factorial_arr: Array, factorial_inds: ArrayLike
) -> Array:
    factorial_inds = jnp.asarray(factorial_inds)
    if factorial_arr.ndim == 0 or factorial_arr.ndim == 1:
        return local_arr
    elif factorial_arr.ndim == 2:
        return _marginalize_and_insert_mean(local_arr, factorial_arr, factorial_inds)
    elif factorial_arr.ndim == 3:
        return _marginalize_and_insert_chol_cov(
            local_arr, factorial_arr, factorial_inds
        )
    else:
        raise ValueError(f"Array must be 3D or lower, got {local_arr.ndim}D")


def _marginalize_and_insert_mean(
    local_mean: Array,
    factorial_means: Array,
    factorial_inds: Array,
) -> Array:
    local_mean_with_factorial_dimension = local_mean.reshape(len(factorial_inds), -1)
    return factorial_means.at[factorial_inds].set(local_mean_with_factorial_dimension)


def _marginalize_and_insert_chol_cov(
    local_chol_cov: Array,
    factorial_chol_covs: Array,
    factorial_inds: Array,
) -> Array:
    d = factorial_chol_covs.shape[-1]
    marginal_chol_covs = block_marginal_sqrt_cov(local_chol_cov, d)
    return factorial_chol_covs.at[factorial_inds].set(marginal_chol_covs)
