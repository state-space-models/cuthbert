# Linearize

This sub-repository provides a function for linearizing a conditional log density.

Usage:

```python
from linearize import linearize

def log_density(x, y):
    ... # some conditional log density function log p(y|x) that returns a scalar

x, y = ... # some input points

mat, shift, chol_cov = linearize(log_density, x, y)
```

The output then defines a linear Gaussian approximation to the conditional log_density
around the points `x` and `y`.

It is exact for all points if the log_density is a linear Gaussian

$$
\log p(y \mid x) = -\frac{1}{2}(y - H x - d)^T (LL^T)^{-1} (y - H x - d) + \text{const},
$$

in which case `linearize` recovers `(H, d, L)` for any input points `x` and `y`.

For non-linear Gaussian log_densities, `linearize` provides an approximation
in the case that the Hessian of the log_density is not positive definite.
The user can also linearize with their own approximate covariance Cholesky factor
with `linearize_given_chol_cov`:

```python
from linearize import linearize_given_chol_cov

def log_density(x, y):
    ... # some conditional log density function log p(y|x) that returns a scalar

x, y = ... # some input points

chol_cov = ... # some approximate Cholesky factor of the covariance matrix

mat, shift = linearize_given_chol_cov(log_density, x, y, chol_cov)
```

