# Linalg

This sub-repository contains any modular linear algebra primitives that are useful
for `cuthbert` and not already provided by `jax`.

In particular we have:

- `tria`, which computes a lower triangular
matrix square root of a given positive definite matrix `R` such that `R @ R.T = A @ A.T`
for a given matrix `A` that is not necessarily square.
- `collect_nans_chol`, which reorders a generalized Cholesky factor to move a
specified subset of rows and columns to the start with remaining dimensions moved
to the end and parameterized so that they are ignored in a Bayesian update or
logpdf calculation.
- `symmetric_inv_sqrt`, which computes the inverse square root of a symmetric matrix. It
does so exactly in the case that the matrix is positive definite. In the case of zero
or negative singular values, it supports approximate inverse square roots in a similar
manner to (Moore-Penrose) pseudo-inversion.
- `marginal_sqrt_cov`, which extracts a square root submatrix from a joint square root
matrix.
