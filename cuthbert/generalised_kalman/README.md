# Generalised Kalman

This module contains an interface for online and offline inference with Gaussian
filtering and smoothing distributions.

This includes exact inference for linear Gaussian state-space-models, as well as
approximate inference for general state-space-models using sigma point quadrature or
linearization (i.e. extended Kalman filtering).

All that is required is a `LinearGaussianSSM` which defines the conditional parameters
of the state-space-model:

$$
\begin{align*}
    p(x_0 \mid u_0) &= \mathrm{N}(m_0, Q_0), \\
    p(x_{t+1} | x_t, u_t) &= \mathrm{N}(F_t x_t + c_t, Q_t), \\
    p(y_t | x_t, u_t) &= \mathrm{N}(H_t x_t + d_t, R_t),
\end{align*}
$$

where the `LinearGaussianSSM` generates the Gaussian parameters
$m_0, Q_0, F_t, c_t, Q_t, H_t, d_t, R_t$ as specified by the protocols in
[`linear_gaussian_ssm.py`](linear_gaussian_ssm.py) (although we enforce all covariances to be provided as Cholesky
factors).


This can then be passed to the `build_inference` function to generate an inference object
which can be used to do filtering (online and offline) and smoothing in a unified interface.

