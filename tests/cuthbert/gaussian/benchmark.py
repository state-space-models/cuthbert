"""Benchmark compilation time and run time for the Kalman filter.

Run with

```python
python -m tests.cuthbert.gaussian.benchmark
```

from the root folder.

Example output as of commit f20c6ff:
1. AMD Ryzen 7 PRO 7840U CPU
    Compile time: 5.239s
    Runtime (min): 0.072s
    Runtime (median): 0.076s
2. NVIDIA A100-SXM4-80GB GPU
    Compile time: 16.241s
    Runtime (min): 0.022s
    Runtime (median): 0.022s
"""

import timeit

import jax
import jax.numpy as jnp
import numpy as np

from cuthbert import filter
from cuthbert.gaussian import kalman
from tests.cuthbertlib.kalman.utils import generate_lgssm

seed = 0
x_dim = 20
y_dim = 10
num_time_steps = 1000

m0, chol_P0, Fs, cs, chol_Qs, Hs, ds, chol_Rs, ys = generate_lgssm(
    seed, x_dim, y_dim, num_time_steps
)


def get_init_params(model_inputs):
    return m0, chol_P0


def get_dynamics_params(model_inputs):
    return Fs[model_inputs - 1], cs[model_inputs - 1], chol_Qs[model_inputs - 1]


def get_observation_params(model_inputs):
    return Hs[model_inputs], ds[model_inputs], chol_Rs[model_inputs], ys[model_inputs]


filter_obj = kalman.build_filter(
    get_init_params, get_dynamics_params, get_observation_params
)
model_inputs = jnp.arange(num_time_steps + 1)

jitted_filter = jax.jit(filter, static_argnames=("filter_obj", "parallel"))

# Check compilation time
compile_timer = timeit.Timer(
    lambda: jax.block_until_ready(
        jitted_filter(filter_obj, model_inputs, parallel=True)
    )
)
compile_time = compile_timer.timeit(number=1)

# Check run time
num_runs = 10
runtime_timer = timeit.Timer(
    lambda: jax.block_until_ready(
        jitted_filter(filter_obj, model_inputs, parallel=True)
    )
)
runtimes = runtime_timer.repeat(repeat=num_runs, number=1)

min_runtime = np.min(runtimes).item()
median_runtime = np.median(runtimes).item()

print(f"Compile time: {compile_time:.3f}s")
print(f"Runtime (min): {min_runtime:.3f}s")
print(f"Runtime (median): {median_runtime:.3f}s")
