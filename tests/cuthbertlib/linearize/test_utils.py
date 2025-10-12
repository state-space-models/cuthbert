import chex
import jax
import numpy as np
import pytest
from jax import numpy as jnp

from cuthbertlib.linearize import symmetric_inv_sqrt


@pytest.fixture(scope="module", autouse=True)
def config():
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", False)


@pytest.mark.parametrize("seed", [0, 42, 99, 123, 456])
@pytest.mark.parametrize("x_dim", [3])
def test_inv_sqrt_positive_definite(seed, x_dim) -> None:
    rng = np.random.default_rng(seed)

    A = rng.normal(size=(x_dim, x_dim))
    A = A @ A.T

    L = symmetric_inv_sqrt(A)

    chex.assert_trees_all_close(L @ L.T, jnp.linalg.inv(A))
    chex.assert_trees_all_close(jnp.linalg.inv(L @ L.T), A)


@pytest.mark.parametrize("seed", [0, 42, 99, 123, 456])
@pytest.mark.parametrize("x_dim", [3])
def test_inv_sqrt_singular_indefinite(seed, x_dim) -> None:
    rng = np.random.default_rng(seed)

    A = rng.normal(size=(x_dim, x_dim))
    A = A @ A.T

    A[0, 0] = -1
    A[1, 1] = 0.1

    L = symmetric_inv_sqrt(A)

    assert L.shape == (x_dim, x_dim)
    assert jnp.array_equal(jnp.tril(L), L)


@pytest.mark.parametrize("x_dim", [3])
def test_inv_sqrt_zeros(x_dim) -> None:
    A = jnp.zeros((x_dim, x_dim))
    A = A @ A.T

    L = jax.jit(symmetric_inv_sqrt)(A)

    # Expected: NaN on diagonal (dimensions with zero diagonal are treated as invalid),
    # zeros elsewhere
    assert L.shape == (x_dim, x_dim)
    assert jnp.all(jnp.isnan(jnp.diag(L)))
    assert jnp.all(
        L[~jnp.eye(x_dim, dtype=bool)] == 0.0
    )  # Check non-diagonal elements are zero


@pytest.mark.parametrize("seed", [0, 42, 99, 123, 456])
@pytest.mark.parametrize("x_dim", [3])
def test_inv_sqrt_partial_zeros(seed, x_dim) -> None:
    rng = np.random.default_rng(seed)

    A = rng.normal(size=(x_dim, x_dim))
    A = A @ A.T

    A_zeros = jnp.zeros((x_dim, x_dim))
    A_zeros = A_zeros @ A_zeros.T

    A_partial_zeros = jnp.block(
        [
            [A_zeros, A_zeros, A_zeros],
            [A_zeros, A, A_zeros],
            [A_zeros, A_zeros, A_zeros],
        ]
    )

    # # Change diag of A_partial_zeros to be nans where previously zeros
    # A_partial_zeros = A_partial_zeros.at[jnp.diag_indices_from(A_partial_zeros)].set(
    #     jnp.where(jnp.diag(A_partial_zeros) == 0.0, jnp.nan, jnp.diag(A_partial_zeros))
    # )

    L = symmetric_inv_sqrt(A_partial_zeros)

    # Extract the valid (non-NaN) dimensions from the output L
    # (dimensions with zero diagonal in input become NaN in output)
    valid_mask = ~jnp.isnan(jnp.diag(L))
    valid_indices = jnp.where(valid_mask)[0]
    L_valid = L[valid_indices[:, None], valid_indices]

    # Test that extracting the valid block works
    chex.assert_trees_all_close(L_valid @ L_valid.T, jnp.linalg.inv(A))
    chex.assert_trees_all_close(jnp.linalg.inv(L_valid @ L_valid.T), A)

    # Test that L @ L.T also gives correct result in the valid block
    L_LT = L @ L.T
    L_LT_valid = L_LT[valid_indices[:, None], valid_indices]
    chex.assert_trees_all_close(L_LT_valid, jnp.linalg.inv(A))

    # Check that invalid dimensions have NaN on diagonal
    invalid_mask = ~valid_mask
    assert jnp.all(jnp.isnan(jnp.diag(L)) == invalid_mask)

    # Check that all off-diagonal elements in rows/cols with invalid diagonal are zero
    # For lower triangular: L[i, j] should be 0 if i has invalid diagonal and j < i
    # or if j has invalid diagonal and i > j
    invalid_row_mask = invalid_mask[:, None]
    invalid_col_mask = invalid_mask[None, :]
    lower_tri_mask = jnp.tril(jnp.ones_like(L, dtype=bool), k=-1)
    invalid_lower_tri = (invalid_row_mask | invalid_col_mask) & lower_tri_mask
    assert jnp.all(L[invalid_lower_tri] == 0.0)
