# Factorial State-Space Models

A factorial state-space model is a state-space model where the dynamics distribution
factors into a product of independent distributions across factors

$$
p(x_t \mid x_{t-1}) = \prod_{f=1}^F p(x_t^f \mid x_{t-1}^f),
$$
for factorial index $f \in \{1, \ldots, F\}$. We additionally assume that observations
act locally on some subset of factors $S_t \subseteq \{1, \ldots, F\}$.

$$
p(y_t \mid x_t) = p(y_t \mid x_t^{S_t}).
$$

This motivates a factored approximation of filtering and smoothing distributions, e.g.

$$
p(x_t \mid y_{0:t}) = \prod_{f=1}^F p(x_t^f \mid y_{0:t}).
$$

A tutorial on factorial state-space models can be found in [Duffield et al](https://doi.org/10.1093/jrsssc/qlae035).

The factorial approximation allows us to exploit significant benefits in terms of
memory, compute and parallelization.

Note that although the dynamics are factorized, `cuthbert` does not differentiate
between `predict` and `update` (instead favouring a unified filter operation
via `filter_prepare` and `filter_combine`). Thus the dynamics and model inputs
should be specified to act on the joint local state (i.e. block diagonal
where appropriate).


## Factorial filtering with `cuthbert`

Filtering in a factorial state-space model is similar to standard filtering, but with
an additional step before the filtering operation to extract the relevant 
factors as well as an additional step after the filtering operation to insert the
updated factors back into the factorial state.


```python
from jax import tree
import cuthbert

# Define model_inputs
model_inputs = ...

# Define function to extract the factorial indices from model inputs
# Here we assume model_inputs is a NamedTuple with a field `factorial_inds`
get_factorial_indices = lambda mi: mi.factorial_inds

# Build factorializer for the inference method
factorializer = cuthbert.factorial.gaussian.build_factorializer(get_factorial_indices)

# Load inference method, with parameter extraction functions defined for factorial inference
kalman_filter = cuthbert.gaussian.kalman.build_filter(
    get_init_params=get_init_params,  # Init specified to generate factorial state
    get_dynamics_params=get_dynamics_params,  # Dynamics specified to act on joint local state
    get_observation_params=get_observation_params,  # Observation specified to act on joint local state
)

# Online inference
factorial_state = kalman_filter.init_prepare(tree.map(lambda x: x[0], model_inputs))

for t in range(1, T):
    model_inputs_t = tree.map(lambda x: x[t], model_inputs)
    factorial_inds = get_factorial_indices(model_inputs_t)
    local_state = factorializer.extract_and_join(factorial_state, factorial_inds)
    prepare_state = kalman_filter.filter_prepare(model_inputs_t)
    filtered_local_state = kalman_filter.filter_combine(local_state, prepare_state)
    factorial_state = factorializer.marginalize_and_insert(
        filtered_local_state, factorial_state, factorial_inds
    )
```

You can also use `cuthbert.factorial.filter` for convenient offline filtering.
Note that associative/parallel filtering is not supported for factorial filtering.

```python
init_factorial_state, local_filter_states = cuthbert.factorial.filter(
    kalman_filter, factorializer, model_inputs, output_factorial=False
)
```

## Factorial smoothing with `cuthbert`

Smoothing in factorial state-space models can be performed embarrassingly parallel
along the factors since the dynamics and factorial approximation are independent
across factors (the observations are fully absorbed in the filtering and
are not accessed during smoothing).

The model inputs and filter states require some preprocessing to convert from being
single sequence with each state containing all factors into a sequence or multiple
sequences with each state corresponding to a single factor. This can be
fiddly but is left to the user for maximum freedom.

TODO: Document some use cases in the examples.

After this preprocessing, smoothing can be performed as usual:

```python
# Define model_inputs for a single factor
model_inputs_single_factor = ...

# Similarly, we need to extract the filter states for the single factor we're smoothing.
filter_states_single_factor = ...

# Load smoother, with parameter extraction functions defined for factorial inference
kalman_smoother = cuthbert.gaussian.kalman.build_smoother(
    get_dynamics_params=get_dynamics_params,  # Dynamics specified to act on joint local state
)

smoother_state = kalman_smoother.convert_filter_to_smoother_state(
    tree.map(lambda x: x[-1], filter_states_single_factor),
    model_inputs=tree.map(lambda x: x[-1], model_inputs_single_factor),
)

for t in range(T - 1, -1, -1):
    model_inputs_single_factor_t = tree.map(lambda x: x[t], model_inputs_single_factor)
    filter_state_single_factor_t = tree.map(lambda x: x[t], filter_states_single_factor)
    prepare_state = kalman_smoother.smoother_prepare(
        filter_state_single_factor_t, model_inputs_single_factor_t
    )
    smoother_state = kalman_smoother.smoother_combine(prepare_state, smoother_state)
```

Or directly using the `cuthbert.smoother`:

```python
smoother_states = cuthbert.smoother(
    kalman_smoother, filter_states_single_factor, model_inputs_single_factor
)
```