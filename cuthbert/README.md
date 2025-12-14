# cuthbert

This folder contains the code for the main `cuthbert` package.

<!--unified-interface-start-->
All inference methods are implemented with the following unified interface:

```python
from jax import tree

# Define model_inputs
model_inputs = ...

# Load inference method
kalman_filter = cuthbert.gaussian.kalman.build_filter(
    get_init_params=get_init_params,
    get_dynamics_params=get_dynamics_params,
    get_observation_params=get_observation_params,
)   # build_filter function takes all inference-specific arguments, swap this out for different inference methods.

# Online inference
state = kalman_filter.init_prepare(tree.map(lambda x: x[0], model_inputs))

for t in range(1, T):
    model_inputs_t = tree.map(lambda x: x[t], model_inputs)
    prepare_state = kalman_filter.filter_prepare(model_inputs_t)
    state = kalman_filter.filter_combine(state, prepare_state)
```

Or for offline inference:

```python
kalman_smoother = cuthbert.gaussian.kalman.build_smoother(get_dynamics_params)

filter_states = cuthbert.filter(kalman_filter, model_inputs)
smoother_states = cuthbert.smoother(kalman_smoother, filter_states, model_inputs)
```
<!--unified-interface-end-->