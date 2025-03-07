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

