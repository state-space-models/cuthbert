# Linearize

This sub-repository provides functions for linearizing conditional distributions
in various forms into a linear Gaussian form. That is form an approximate Gaussian
defined by the tuple `(H, d, L)` such that

$$
\log p(y \mid x) \approx -\frac{1}{2}(y - H x - d)^T (LL^T)^{-1} (y - H x - d) + \text{const},
$$

### Linearization techniques

- `linearize_log_density`: Linearize a conditional log density around given points.
- `linearize_moments`: Linearize conditional mean and Cholesky covariance functions
around a given point.


### Example usage

Specifically for `linearize_log_density`, the usage is as follows:

```python
from linearize import linearize_log_density

def log_density(x, y):
    ... # some conditional log density function log p(y|x) that returns a scalar

x, y = ... # some input points

mat, shift, chol_cov = linearize_log_density(log_density, x, y)
```

Note that when `log_density` is exactly linear Gaussian, then the output from
`linearize_log_density` is exact for all points `x` and `y`. For non-linear and/or
non-Gaussian `log_density`, the output is an approximation that will truncate any
singular values of the precision matrix (negative Hessian of `log_density`).
