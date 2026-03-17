# `cuthbert` conventions

On this page we'll explain and justify some of the conventions and design decisions
made in `cuthbert`.


## Unified Interface

{%
    include-markdown "../cuthbert/README.md"
    start="<!--unified-interface-start-->"
    end="<!--unified-interface-end-->"
%}

!!! success "Justification"
    The unified interface allows all inference-specific details to be encapsulated
    within the `build_filter` arguments with the subsequent methods being unified
    across all inference methods. This allows for the user to swap between inference
    methods and use model agnostic `cuthbert.filter` and `cuthbert.smoother` methods.


## Filter as a unified operation (no `predict` or `update`)

`cuthbert` methods do not have individual `predict` or `update` methods. Instead, they
are unified into a single `filter` method. Of course, in practice the user can still
call separate `predict` and `update` methods directly if they so desire.

The user can achieve a `predict` step through a degenerate observation
i.e. $p(y_t \mid x_t) \propto 1$.

Similarly an `update` step can be achieved through degenerate dynamics
i.e. $p(x_t \mid x_{t-1}) = \delta(x_t \mid x_{t-1})$.

All `cuthbert` methods support these degenerate cases through appropriate specification
`model_inputs` and the functions passed to `build_filter`.

### Degenerate observation (predict)

- For discrete methods a degenerate observation is achieved by making
`get_obs_lls` return a constant array (i.e. all zeros).

- For Gaussian methods a degenerate observation is achieved by setting the observation
to an array of all `jnp.nan`.

- For SMC methods a degenerate observation is achieved by making
`log_potential` return a constant (i.e. zero).


### Degenerate dynamics (update)

- For discrete methods degenerate dynamics are achieved by making
`get_trans_matrix` return the identity matrix.

- For Gaussian methods degenerate dynamics are achieved by setting `chol_Q` to a zero
matrix. With the exception of `gaussian.taylor` where the user does not define
`chol_Q` directly, in this case degenerate dynamics are achieved with a `linearization_point`
of all `jnp.nan`.

- For SMC methods degenerate dynamics are achieved by making
`propagate_sample` return the previous state unchanged.


!!! success "Justification"
    The single `filter` method represents a simplified interface which is easier to test
    and maintain whilst providing the user with only one method to call.



## No initial observation

`cuthbert` adopts the convention of defining state-space models as $p(x_{0:T} \mid y_{1:T})$ rather than $p(x_{1:T} \mid y_{1:T})$ as is common in some implementations
(such as [`dynamax`](https://github.com/probml/dynamax)).

Given the above support for degenerate dynamics and observations, the user can achieve
a $p(x_{1:T} \mid y_{1:T})$ model by passing degenerate dynamics to the first step
of a $p(x_{0:T} \mid y_{1:T})$ model.

!!! success "Justification"
    The decision to omit of an initial observation was to support [factorial state-space models](https://doi.org/10.1093/jrsssc/qlae035) where the initialisation applies globally
    to all factors vs filter steps which act locally on a small number of factors.
    Including an initial observation resulted in awkward shape mismatches in the
    `init_prepare` function.



## Square root covariance matrices




## Why `prepare` and `combine`?




## What goes in `model_inputs`?




## No dedicated methods for parameter estimation
