import chex
import jax
import numpy as np
import pytest
from jax import numpy as jnp

from cuthbertlib.linalg import chol_cov_with_nans_to_cov, symmetric_inv_sqrt


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
def test_inv_sqrt_positive_definite_jit(seed, x_dim) -> None:
    rng = np.random.default_rng(seed)

    A = rng.normal(size=(x_dim, x_dim))
    A = A @ A.T

    L = jax.jit(symmetric_inv_sqrt)(A)

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


@pytest.mark.parametrize("seed", [0, 42, 99, 123, 456])
@pytest.mark.parametrize("x_dim", [3])
def test_inv_sqrt_singular_indefinite_jit(seed, x_dim) -> None:
    rng = np.random.default_rng(seed)

    A = rng.normal(size=(x_dim, x_dim))
    A = A @ A.T

    A[0, 0] = -1
    A[1, 1] = 0.1

    L = jax.jit(symmetric_inv_sqrt)(A)

    assert L.shape == (x_dim, x_dim)
    assert jnp.array_equal(jnp.tril(L), L)


@pytest.mark.parametrize("x_dim", [3])
def test_inv_sqrt_zeros(x_dim) -> None:
    A = jnp.zeros((x_dim, x_dim))

    L = symmetric_inv_sqrt(A)

    # Expected: all nans (inv of 0 is nan)
    assert L.shape == (x_dim, x_dim)
    assert jnp.all(jnp.isnan(L))


@pytest.mark.parametrize("seed", [0, 42, 99, 123, 456])
@pytest.mark.parametrize("x_dim", [3])
def test_inv_sqrt_partial_nans(seed, x_dim) -> None:
    """Test that NaN dimensions are properly treated as missing."""
    rng = np.random.default_rng(seed)

    A = rng.normal(size=(x_dim, x_dim))
    A = A @ A.T

    A_nans = jnp.full((x_dim, x_dim), jnp.nan)

    A_partial_nans = jnp.block(
        [
            [A_nans, A_nans, A_nans],
            [A_nans, A, A_nans],
            [A_nans, A_nans, A_nans],
        ]
    )

    L = symmetric_inv_sqrt(A_partial_nans, ignore_nan_dims=True)

    # Extract the valid (non-NaN) dimensions from the output L
    # (dimensions with NaN diagonal in input remain NaN in output)
    valid_mask = ~jnp.isnan(jnp.diag(L))
    valid_indices = jnp.where(valid_mask)[0]
    L_valid = L[valid_indices[:, None], valid_indices]

    assert valid_mask.sum() == x_dim

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

    # With ignore_nan_dims=False, we should get all nans as they propagate through
    # entire matrix
    L_ignore_nan_dims_false = symmetric_inv_sqrt(A_partial_nans, ignore_nan_dims=False)
    assert jnp.all(jnp.isnan(L_ignore_nan_dims_false @ L_ignore_nan_dims_false.T))


@pytest.mark.parametrize("seed", [0, 42, 99, 123, 456])
@pytest.mark.parametrize("x_dim", [3])
def test_chol_cov_with_nans_to_cov(seed, x_dim) -> None:
    """Test that chol_cov_with_nans_to_cov properly handles NaN dimensions."""
    rng = np.random.default_rng(seed)

    # Create a valid cholesky factor
    A = rng.normal(size=(x_dim, x_dim))
    chol = jnp.linalg.cholesky(A @ A.T)

    # Create a version with NaN on some diagonals
    chol_with_nans = chol.at[0, 0].set(jnp.nan)

    # Convert to covariance
    cov = chol_cov_with_nans_to_cov(chol_with_nans)

    # Check that dimension 0 has NaN on diagonal
    assert jnp.isnan(cov[0, 0])

    # Check that dimension 0 has zero off-diagonal
    assert jnp.all(cov[0, 1:] == 0.0)
    assert jnp.all(cov[1:, 0] == 0.0)

    # Check that valid dimensions match chol @ chol.T
    expected_cov_valid_dims = chol[1:, 1:] @ chol[1:, 1:].T
    chex.assert_trees_all_close(cov[1:, 1:], expected_cov_valid_dims)

    # Check it works fine with no nans
    cov_no_nans = chol_cov_with_nans_to_cov(chol)
    chex.assert_trees_all_close(cov_no_nans, A @ A.T)
