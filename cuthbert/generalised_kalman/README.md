# Generalised Kalman

This module contains an interface for online and offline inference with Gaussian
filtering and smoothing distributions.

This includes exact inference for linear Gaussian state-space-models, and approximate inference
for general state-space-models using sigma point quadrature or linearisation (i.e. extended Kalman filtering).

All that is required is a `ConditionalMomentsSSM` which defines the conditional moments
of the state-space-model:

$$
\begin{align*}
    p(x_0 \mid u_0) &= \mathrm{N}(m_0, Q_0), \\
    p(x_{t+1} | x_t, u_t) &= \mathrm{N}(F_t x_t + c_t, Q_t), \\
    p(y_t | x_t, u_t) &= \mathrm{N}(H_t x_t + d_t, R_t),
\end{align*}
$$

where the `ConditionalMomentsSSM` generates the Gaussian parameters $m_0, Q_0, F_t, c_t, Q_t, H_t, d_t, R_t$ as specified by the protocols in [`conditional_moments.py`](conditional_moments.py)
(although we enforce all covariances to be provided as Cholesky factors).

