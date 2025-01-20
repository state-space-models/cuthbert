# Quadrature based linearization

This sub-repository is concerned with the problem of forming linear $Y \approx A X + b + \epsilon$ (where $\epsilon$ is a zero-mean Gaussian with covariance $Q$) approximations to general statistical models $p(y \mid x)$ which either exhibit additive noise:
```math
p(y \mid x) = \mathcal{N}(y; f(x), \Sigma)
```
where $f$ is a deterministic function and $\Sigma$ a given covariance matrix,
or for which the conditional mean and covariance
```math
\mathbb{E}[Y \mid X=x] = m(x), \quad \mathbb{V}[Y \mid X=x] = c(x)
```
are known or can be approximated otherwise.
This approximation is done under minimizing (approximately for the latter, exactly for the former) the expected [Kullback-Leibler divergence](https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence)
```math
A, b, Q = \textrm{arg min } \mathbb{E}_{\mathcal{N}(X \mid m, P)}\left[\textrm{KL}(p(y \mid X) || \mathcal{N}(y; AX + b, Q)\right].
```
This can be done either directly in the covariance form (where $\Sigma$ is provided and $Q$ is obtained as covariance matrices) or in the square-root form, more stable but computationally more expensive (where $\Sigma$ is provided as a Cholesky decomposition and $L$ obtained such that $Q = L L^{T}$ is the covariance matrix of interest).

A typical call to the library would then be:
```python
mean_fn = lambda x: jnp.sin(x)
cov_fn = lambda x: 1e-3 * jnp.eye(2)
quadrature_method = quadrature.gauss_hermite.weights(n_dim=2, order=3)
m = jnp.zeros((2,))
cov = jnp.eye(2)
A, b, Q = quadrature.conditional_moments(mean_fn, cov_fn, m, cov, quadrature_method, mode="covariance")
```
