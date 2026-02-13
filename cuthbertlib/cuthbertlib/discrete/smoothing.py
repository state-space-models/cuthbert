"""Implements the discrete HMM smoothing associative operator."""

import jax.numpy as jnp

from cuthbertlib.types import Array, ArrayLike


def get_reverse_kernel(x_t_dist: ArrayLike, trans_matrix: ArrayLike) -> Array:
    r"""Computes reverse transition probabilities $p(x_{t-1} \mid x_{t}, \dots)$ for a discrete HMM.

    Args:
        x_t_dist: Array of shape (N,) where `x_t_dist[i]` = $p(x_{t} = i \mid \dots)$.
        trans_matrix: Array of shape (N, N) where
            `trans_matrix[i, j]` = $p(x_{t} = j \mid x_{t-1} = i)$.

    Returns:
        An (N, N) matrix `x_tm1_dist[i, j]` = $p(x_{t-1} = j \mid x_{t} = i, \dots)$.
    """
    x_t_dist, trans_matrix = jnp.asarray(x_t_dist), jnp.asarray(trans_matrix)
    pred = jnp.dot(trans_matrix.T, x_t_dist)
    x_tm1_dist = trans_matrix.T * x_t_dist[None, :] / pred[:, None]
    return x_tm1_dist


def smoothing_operator(elem_ij: Array, elem_jk: Array) -> Array:
    """Binary associative operator for smoothing in discrete HMMs.

    Args:
        elem_ij: Smoothing scan element.
        elem_jk: Smoothing scan element.

    Returns:
        The output of the associative operator applied to the input elements.
    """
    return elem_jk @ elem_ij
