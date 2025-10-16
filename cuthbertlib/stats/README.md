# Stats

This sub-repository contains modular statistical primitives that are useful
for `cuthbert` and not already provided by `jax`.

In particular, it contains a `multivariate_normal` module, which provides
logpdf and pdf functions for the multivariate normal distribution where the
covariance matrix is provided in square-root (Cholesky) form as opposed
to the full covariance matrix required by
[`jax.scipy.stats.multivariate_normal`](https://docs.jax.dev/en/latest/jax.scipy.html#module-jax.scipy.stats.multivariate_normal).
