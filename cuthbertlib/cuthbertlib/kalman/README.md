# Kalman

This sub-repository provides modular functions for Kalman filtering and smoothing.

The core functions are:

- `predict`: Single prediction step.
- `filter_update`: Single update step.
- `smoother_update`: Single Rauch-Tung-Striebel smoothing step.

Together, `predict` and `filter_update` can be used to perform an online filtering step.

In all cases, we operate on the **square-root form of the covariance matrix**, which is
more numerically stable (in low-precision floating point arithmetic) as the outputs are 
guaranteed to be positive-definite. This means **we also require input
covariance matrices to be provided in square-root (Cholesky) form**.
