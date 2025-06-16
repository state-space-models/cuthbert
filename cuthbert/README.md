# cuthbert

This folder contains the code for the main `cuthbert` package.

All inference methods are implemented with the following unified interface:

```python
from jax import tree

# Define model_inputs
model_inputs = ...

# Load inference method
inference = cuthbert.gaussian.kalman.build(
    get_init_params=get_init_params,
    get_dynamics_params=get_dynamics_params,
    get_observation_params=get_observation_params,
)   # Build function takes all inference-specific arguments, swap this out for different inference methods.

# Online inference
state = inference.init_prepare(tree.map(lambda x: x[0], model_inputs))

for t in range(1, T):
    model_inputs_t = tree.map(lambda x: x[t], model_inputs)
    prepare_state = inference.filter_prepare(model_inputs_t)
    state = inference.filter_combine(state, prepare_state)
```

Or for offline inference:

```python
filter_states = cuthbert.filter(inference, model_inputs)
smoother_states = cuthbert.smoother(inference, filter_states, model_inputs)
```