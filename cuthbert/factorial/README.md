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


## Factorial filtering with `cuthbert`



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


