import jax.numpy as jnp

from cuthbertlib.types import Array, ArrayLike


def get_reverse_kernel(x_t_dist: ArrayLike, trans_matrix: ArrayLike) -> Array:
    """Computes reverse transition probabilities p(x_{t-1} | x_{t}, ...) for a discrete HMM.

    Args:
        x_t_dist: Array of shape (N,) with x_t_dist[i] = p(x_{t} = i | ...).
        trans_matrix: Array of shape (N, N) with
            trans_matrix[i, j] = p(x_{t} = j | x_{t-1} = i).

    Returns:
        An (N, N) matrix x_tm1_dist[i, j] = p(x_{t-1} = j | x_{t} = i, ...).
    """
    x_t_dist, trans_matrix = jnp.asarray(x_t_dist), jnp.asarray(trans_matrix)
    pred = jnp.dot(trans_matrix.T, x_t_dist)
    x_tm1_dist = trans_matrix.T * x_t_dist[None, :] / pred[:, None]
    return x_tm1_dist


def smoothing_operator(elem_ij: Array, elem_jk: Array) -> Array:
    """Binary associative operator for smoothing in HMMs.

    Args:
        elem_ij, elem_jk: Smoothing scan elements.

    Returns:
        The output of the associative operator applied to the input elements.
    """
    return elem_jk @ elem_ij
