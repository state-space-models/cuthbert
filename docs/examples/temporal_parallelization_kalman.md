# Temporally-Parallelized Kalman Filter

In `cuthbert`, we provide an implementation of the Kalman filter that can be
executed in parallel across time steps. For a problem with $T$ time steps, if
there are $T$ available parallel workers, this implementation achieves
logarithmic time complexity $\mathcal{O}(\log(T))$ as opposed to the standard
linear time complexity, as shown in [Särkka and Garcia-Fernández](https://doi.org/10.1109/TAC.2020.2976316). Users can decide whether to run the filter in parallel
via the `parallel` argument to the [`filter`][cuthbert.filtering.filter]
function. In this example, we demonstrate the usage and performance of the
temporally-parallelized Kalman filter.

## Setup and imports

We first import the necessary libraries and specify a linear-Gaussian
state-space model. We use a helper function called `generate_lgssm` to create
example model parameters and observations, and then build the Kalman filter
object like we covered in the [Kalman tracking example](./kalman_tracking.md).

```{.python #parallel-kalman-setup}
import timeit

import jax
import jax.numpy as jnp
import numpy as np

from cuthbert import filter
from cuthbert.gaussian import kalman
from cuthbertlib.kalman.generate import generate_lgssm

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
```

## Time Everything

We JIT-compile the [`filter`][cuthbert.filtering.filter] function, making
sure to mark the `filter_obj` and `parallel` arguments as static. We then
measure the compilation times using the
[`timeit`](https://docs.python.org/3/library/timeit.html) module for both the
sequential and parallel implementations.

```{.python #parallel-kalman-compiletime}
jitted_filter = jax.jit(filter, static_argnames=("filter_obj", "parallel"))

seq_compile_time = timeit.Timer(
    lambda: jax.block_until_ready(
        jitted_filter(filter_obj, model_inputs, parallel=False)
    )
).timeit(number=1)
par_compile_time = timeit.Timer(
    lambda: jax.block_until_ready(
        jitted_filter(filter_obj, model_inputs, parallel=True)
    )
).timeit(number=1)
```

Let's do the same for the runtimes. We run each implementation 10 times and
report the minimum and median runtimes.

```{.python #parallel-kalman-runtime}
num_runs = 10

seq_runtimes = timeit.Timer(
    lambda: jax.block_until_ready(
        jitted_filter(filter_obj, model_inputs, parallel=False)
    )
).repeat(repeat=num_runs, number=1)
par_runtimes = timeit.Timer(
    lambda: jax.block_until_ready(
        jitted_filter(filter_obj, model_inputs, parallel=True)
    )
).repeat(repeat=num_runs, number=1)

print("             Sequential | Parallel")
print("-" * 35)
print(f"Compile time  : {seq_compile_time: >7.3f}s | {par_compile_time: >7.3f}s")
print(f"Min runtime   : {np.min(seq_runtimes): >7.3f}s | {np.min(par_runtimes): >7.3f}s")
print(f"Median runtime: {np.median(seq_runtimes): >7.3f}s | {np.median(par_runtimes): >7.3f}s")
```

## Example Results

Running the above code on an AMD Ryzen 7 PRO 7840U CPU yields:

```txt
              Sequential | Parallel
------------------------------------
Compile time  :   0.422s |    4.932s
Min runtime   :   0.042s |    0.071s
Median runtime:   0.043s |    0.076s
```

We highlight two things. First, the compile time for the parallel version is
higher, and this is because the parallel implementation is more complex and has
more operations (thus more work for the compiler). Second, since this CPU
only has 16 threads, there's not enough opportunity for parallelism, and hence
the parallel version ends up being slower due to the higher computational
complexity.

The benefit of the parallel version becomes clear when we run it on a GPU, in
this case on an NVIDIA A100-SXM4-80GB:

```txt
               Sequential | Parallel
--------------------------------------
Compile time  :    2.541s |   15.345s
Min runtime   :    0.597s |    0.022s
Median runtime:    0.598s |    0.022s
```

The parallel implementation is now about 27 times faster than the sequential
one, and this difference will only increase with increasing $T$. So if you have
a problem where you have to run the Kalman filter (or smoother) repeatedly for
the same model, and you have a GPU available, it might be beneficial to pay the
higher compilation cost and use the parallel implementation.

<!--- entangled-tangle-block
```{.python file=examples_scripts/temporal_parallelization_kalman.py}
<<parallel-kalman-setup>>
<<parallel-kalman-compiletime>>
<<parallel-kalman-runtime>>
```
-->
