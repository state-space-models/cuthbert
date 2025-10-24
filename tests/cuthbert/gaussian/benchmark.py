"""Benchmark compile time and run time for the Kalman filter.

Run with

```python
python -m tests.cuthbert.gaussian.benchmark
```

from the root folder.

Example output as of commit f20c6ff:
1. AMD Ryzen 7 PRO 7840U CPU
    Compile time: 5.215s
    Runtime: 0.074 pm 0.00290s
2. NVIDIA A100-SXM4-80GB GPU
    Compile time: 16.229s
    Runtime: 0.022 pm 0.00010s
"""

import time

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

num_runs = 10
runtimes = []
for _ in range(num_runs):
    start_time = time.time()
    filt_states = jitted_filter(filter_obj, model_inputs, parallel=True)
    jax.block_until_ready(filt_states)
    runtimes.append(time.time() - start_time)

compile_time = runtimes[0]
mean_runtime = np.mean(runtimes[1:]).item()
std_runtime = np.std(runtimes[1:]).item()

print(f"Compile time: {compile_time:.3f}s")
print(f"Runtime: {mean_runtime:.3f} pm {std_runtime:.5f}s")
