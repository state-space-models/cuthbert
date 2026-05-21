"""Factorial utilities for discrete HMM states."""

from functools import reduce

from jax import numpy as jnp

from cuthbert.discrete.filter import DiscreteFilterState
from cuthbert.factorial.types import Factorializer, GetFactorialIndices
from cuthbertlib.types import Array, ArrayLike


def build_factorializer(
    get_factorial_indices: GetFactorialIndices,
) -> Factorializer:
    """Build a factorializer for discrete HMM filter states.

    Args:
        get_factorial_indices: Function to extract the factorial indices
            from model inputs.

    Returns:
        Factorializer object for discrete states with functions to extract and join
        the relevant factors and marginalize and insert the updated factors.
    """
    return Factorializer(
        get_factorial_indices=get_factorial_indices,
        extract=extract,
        join=join,
        marginalize=marginalize,
        insert=insert,
    )


def extract(
    factorial_state: DiscreteFilterState, factorial_inds: ArrayLike
) -> DiscreteFilterState:
    """Extract the relevant factors from a factorial discrete filter state.

    Single dimensional arrays will be treated as scalars or vectors without
        a factorial axis (e.g. a local log normalizing vector with shape (K,)).
    Multidimensional arrays will be treated as arrays with shape (F, *).
        In this case the factorial_inds indices will be extracted from the first
        dimension and then the remaining dimensions will be preserved.

    Here F is the number of factors and K is the number of states per factor.

    Args:
        factorial_state: Factorial discrete filter state storing transition-like
            arrays and log normalizing vectors with leading factorial dimension F.
        factorial_inds: Indices of the factors to extract. Integer array.
            factorial_inds.ndim == 0 removes the factorial dimension and extracts
                a single factor.
            factorial_inds.ndim == 1 retains the factorial dimension,
                even if len(factorial_inds) == 1.

    Returns:
        Factorial discrete filter state with leading dimension
            len(factorial_inds) on leaves that carry a factorial axis.
            If factorial_inds is a single integer, the returned local factorial
            state will not have a factorial dimension.
    """
    factorial_inds = jnp.asarray(factorial_inds)
    f = factorial_state.elem.f[factorial_inds]
    log_g = factorial_state.elem.log_g[factorial_inds]
    return factorial_state._replace(
        elem=factorial_state.elem._replace(f=f, log_g=log_g)
    )


def join(local_factorial_state: DiscreteFilterState) -> DiscreteFilterState:
    """Convert a local factorial discrete state into a joint local state.

    Single dimensional arrays will be treated as vectors with no factorial axis.
    Two dimensional arrays will be treated as local log-normalizer vectors
    with shape (F, K), which become a joint vector with shape (K**F,).
    Three dimensional arrays will be treated as local transition-like matrices
    with shape (F, K, K), which become a joint matrix with shape (K**F, K**F)
    via Kronecker products.

    Here F is the number of local factors and K is the number of states per factor.

    Args:
        local_factorial_state: Local factorial discrete state storing leaves with
            leading factorial dimension F.

    Returns:
        Joint local discrete state with no factorial index dimension.
    """
    f = _join_matrices(local_factorial_state.elem.f)
    log_g = _join_log_vectors(local_factorial_state.elem.log_g)
    return local_factorial_state._replace(
        elem=local_factorial_state.elem._replace(f=f, log_g=log_g)
    )


def _join_matrices(mats: Array) -> Array:
    return reduce(jnp.kron, mats)


def _join_log_vectors(log_vecs: Array) -> Array:
    num_factors, num_states = log_vecs.shape
    joint_num_states = num_states**num_factors
    return jnp.full((joint_num_states,), log_vecs[0, 0])


def marginalize(
    local_state: DiscreteFilterState,
    num_factors: int,
) -> DiscreteFilterState:
    """Marginalize a joint local discrete state into a local factorial state.

    A joint local state stores arrays over the product state space of size K**F.
    This function returns per-factor arrays by summing over all other factors.

    Args:
        local_state: Joint local discrete state with no factorial index dimension.
        num_factors: Number of factors to marginalize out. Integer.

    Returns:
        Local factorial discrete state with leading factorial dimension
            num_factors.
    """
    f = _marginalize_matrix(local_state.elem.f, num_factors)
    log_g = _marginalize_log_vector(local_state.elem.log_g, num_factors)
    return local_state._replace(elem=local_state.elem._replace(f=f, log_g=log_g))


def _infer_num_states(arr: Array, num_factors: int) -> int:
    joint_num_states = arr.shape[-1]
    num_states = int(round(joint_num_states ** (1 / num_factors)))
    if num_states**num_factors != joint_num_states:
        raise ValueError(
            "Unable to infer per-factor state size from joint state size "
            f"{joint_num_states} and num_factors {num_factors}."
        )
    return num_states


def _marginalize_log_vector(log_vec: Array, num_factors: int) -> Array:
    num_states = _infer_num_states(log_vec, num_factors)
    log_norm = jnp.take(log_vec, 0, axis=-1)
    return jnp.full((num_factors, num_states), log_norm)


def _marginalize_matrix(mat: Array, num_factors: int) -> Array:
    num_states = _infer_num_states(mat, num_factors)
    tensor = mat.reshape((num_states,) * (2 * num_factors))
    factor_mats = []
    all_axes = tuple(range(2 * num_factors))

    for i in range(num_factors):
        keep_axes = (i, num_factors + i)
        reduce_axes = tuple(ax for ax in all_axes if ax not in keep_axes)
        marginal = tensor.sum(axis=reduce_axes)
        row_sums = marginal.sum(axis=-1, keepdims=True)
        row_sums = jnp.where(row_sums > 0, row_sums, 1.0)
        factor_mats.append(marginal / row_sums)

    return jnp.stack(factor_mats)


def insert(
    local_factorial_state: DiscreteFilterState,
    factorial_state: DiscreteFilterState,
    factorial_inds: ArrayLike,
) -> DiscreteFilterState:
    """Insert a local factorial discrete state into a factorial discrete state.

    Single dimensional arrays will be treated as vectors with no factorial axis.
    Multidimensional arrays will be treated as arrays with shape (F, *).
        In this case factorial_inds indices will be inserted into the first
        dimension and the remaining dimensions will be preserved.

    Here F is the number of factors and K is the number of states per factor.

    Args:
        local_factorial_state: Local factorial discrete state to insert.
            Leaves with a factorial axis should have first dimension
            len(factorial_inds).
        factorial_state: Global factorial discrete state with first dimension F
            on leaves that carry a factorial axis.
        factorial_inds: Indices of the factors to insert. Integer array.

    Returns:
        Updated factorial discrete state with inserted factors.
    """
    factorial_inds = jnp.asarray(factorial_inds)
    factorial_inds = jnp.atleast_1d(factorial_inds)
    new_f = factorial_state.elem.f.at[factorial_inds].set(local_factorial_state.elem.f)
    new_log_g = jnp.full_like(
        factorial_state.elem.log_g, local_factorial_state.elem.log_g[0, 0]
    )
    return factorial_state._replace(
        elem=factorial_state.elem._replace(f=new_f, log_g=new_log_g),
        model_inputs=local_factorial_state.model_inputs,
    )
