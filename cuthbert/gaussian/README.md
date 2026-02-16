# Gaussian Filters and Smoothers

The core atomic functions can be found in [`cuthbertlib.kalman`](../../cuthbertlib/kalman).

<!--gaussian-filters-and-smoothers-start-->
Gaussian filters in `cuthbert` provide filtering distributions of the form:

$$
p(x_t \mid y_{1:t}) = \mathcal{N}(x_t \mid \mu_{t|t}, L_{t|t} L_{t|t}^T),
$$

where $\mu_{t|t}$ is the filtering mean and $L_{t|t}$ is a generalized Cholesky factor
of the filtering covariance.

The Cholesky factor is generalized in the sense that
$L_{t|t} L_{t|t}^T = \Sigma_{t|t}$ is the filtering covariance matrix and $L_{t|t}$ is
lower triangular but does not necessarily match the unique Cholesky factor the would be
obtained from e.g. `jax.numpy.linalg.cholesky`.

Similarly, Gaussian smoothers in `cuthbert` provide smoothing distributions of the form:

$$
p(x_t \mid y_{1:T}) = \mathcal{N}(x_t \mid \mu_{t|T}, L_{t|T} L_{t|T}^T),
$$

where $\mu_{t|T}$ is the smoothing mean and $L_{t|T}$ is a generalized Cholesky factor
of the smoothing covariance.

The distributions are exact for linear Gaussian state-space models, otherwise
an approximation is induced depending on the `cuthbert` method used.

## Two-step smoothing distributions

Additionally, Gaussian smoothers have optional boolean arguments `store_gain` and `store_chol_cov_given_next` which can be used to store in the smoother state the
smoothing gain matrix $G_{t|T}$ and $V_{t|T}$ where
$V_{t|T}V_{t|T}^T = \text{Cov}[x_t \mid x_{t+1}, y_{1:T}]$.
These matrices can be used to compute the two-step smoothing distributions:

$$
p\left(\begin{pmatrix}
x_{t+1} \\
x_t
\end{pmatrix} \mid y_{1:T}\right) = \mathcal{N}\left(\begin{pmatrix}
x_{t+1} \\
x_t
\end{pmatrix} \mid \begin{pmatrix}
\mu_{t+1|T} \\
\mu_{t|T}
\end{pmatrix}, L_{t:t+1|T}L_{t:t+1|T}^T\right),
$$

where

$$
L_{t:t+1|T} = \begin{pmatrix}
L_{t+1|T} & 0 \\
G_{t|T} L_{t+1|T} & V_{t|T}
\end{pmatrix}.
$$

<!--gaussian-filters-and-smoothers-end-->