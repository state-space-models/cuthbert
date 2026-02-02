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


## Factorial filtering with `cuthbert`

Filtering in a factorial state-space model is similar to standard filtering, but with
additional an additional step before the filtering operation to extract the relevant 
factors as well as an additional step after the filtering operation to insert the
updated factors back into the factorial state.


```python
from jax import tree

# Define model_inputs
model_inputs = ...

# Define factorial function to extract relevant factors and combine into a joint local state
def extract_and_join(state, model_inputs):
    ....

# Define factorial function to marginalize joint local state into a factored state
# and insert into factorial state
def factorial_marginalize_and_insert(state, local_state, model_inputs):
    ....

# Load inference method, with parameter extraction functions defined for factorial inference
kalman_filter = cuthbert.gaussian.kalman.build_filter(
    get_init_params=get_init_params,    # Init specified to generate factorial state
    get_dynamics_params=get_dynamics_params,    # Dynamics specified to act on joint local state
    get_observation_params=get_observation_params,    # Observation specified to act on joint local state
)

# Online inference
factorial_state = kalman_filter.init_prepare(tree.map(lambda x: x[0], model_inputs))

for t in range(1, T):
    model_inputs_t = tree.map(lambda x: x[t], model_inputs)
    local_state = extract_and_join(factorial_state, model_inputs_t)
    prepare_state = kalman_filter.filter_prepare(model_inputs_t)
    filtered_local_state = kalman_filter.filter_combine(local_state, prepare_state)
    factorial_state = factorial_marginalize_and_insert(factorial_state, filtered_local_state, model_inputs_t)
```


## Factorial smoothing with `cuthbert`

Smoothing in factorial state-space models can be performed embarassingly parallel
along the factors since the dynamics and factorial approximation are independent
across factors (the observations are fully absorbed in the filtering and
are not accessed during smoothing).

The model inputs and filter states require some preprocessing to convert from being
single sequence with each state containing all factors into a sequence or multiple
sequences with each state corresponding to a single factor. This can be quite
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

