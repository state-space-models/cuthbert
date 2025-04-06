# Linalg

This sub-repository contains any modular linear algebra primitives that are useful
for `cuthbert` and not already provided by `jax`.

In particular, it contains a `tria` function, which computes a lower triangular
matrix square root of a given positive definite matrix `R` such that `R @ R.T = A @ A.T`
for a given matrix `A` that is not necessarily square.