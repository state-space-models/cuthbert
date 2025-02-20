# Kalman


This sub-repository provides modular functions for Kalman filtering and smoothing.


The core functions are:

- `offline_filter`: Offline Kalman filtering.
- `predict`: Single prediction step.
- `update`: Single update step.
- `smoother`: Offline Rauch-Tung-Striebel smoothing.
- `smoother_update`: Single Rauch-Tung-Striebel smoothing step.

Together, `predict` and `update` can be used to perform an online filtering step.

In all cases, we operate on the **square-root form of the covariance matrix**, which is
more numerically stable (in low-precision floating point arithmetic) as the outputs are 
guaranteed to be positive-definite. This means **we also require input
covariance matrices to be provided in square-root (Cholesky) form**.

The offline `offline_filter` and `smoother` functions both support temporal
parallelization via the `parallel` argument which defaults to `True`.


