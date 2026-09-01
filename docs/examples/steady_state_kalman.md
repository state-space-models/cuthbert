# Steady-state Kalman filter

For **time-invariant** linear Gaussian SSMs the Kalman gain converges to a constant after a few steps.  
`cuthbert` lets you pre-compute that constant gain once — outside
of JIT — and reuse it at every time step, replacing the per-step QR decomposition
inside [`filter_prepare`][cuthbert.gaussian.kalman.filter_prepare] with cheap
matrix–vector products.

This example shows how to use [`compute_steady_state_filter_params`][cuthbertlib.kalman.filtering.compute_steady_state_filter_params]
together with [`build_filter`][cuthbert.gaussian.kalman.build_filter], and benchmarks
the result against the standard filter.

### Setup and imports

```{.python #steady-state-kalman-imports}
import timeit

import jax
import jax.numpy as jnp
from jax import jit, random

from cuthbert import filter as run_filter
from cuthbert.gaussian import kalman
```

### Build a synthetic LG-SSM

We use a random time-invariant system with state dimension $n_x = 20$ and
observation dimension $n_y = 4$.  The larger dimensions make the QR savings more
pronounced than in the small car-tracking example.

```{.python #steady-state-kalman-model}
KEY = random.key(42)
NUM_STEPS = 5_000
NX, NY = 20, 4

key, sk = random.split(KEY)
F = jnp.eye(NX) * 0.99 + 0.01 * random.normal(sk, (NX, NX)) / NX
key, sk = random.split(key)
_noise = random.normal(sk, (NX, NX)) / NX**0.5
chol_Q = jnp.linalg.cholesky(jnp.eye(NX) + 0.1 * _noise @ _noise.T)
key, sk = random.split(key)
H = random.normal(sk, (NY, NX)) / NX**0.5
chol_R = 0.5 * jnp.eye(NY)
c = jnp.zeros(NX)
d = jnp.zeros(NY)
m0 = jnp.zeros(NX)
chol_P0 = jnp.eye(NX)

# Simulate trajectory
key, sk = random.split(key)
x = m0 + chol_P0 @ random.normal(sk, (NX,))
ys = []
for _ in range(NUM_STEPS):
    key, sk1, sk2 = random.split(key, 3)
    x = F @ x + chol_Q @ random.normal(sk1, (NX,))
    y = H @ x + chol_R @ random.normal(sk2, (NY,))
    ys.append(y)
ys = jnp.stack(ys)
```

### Build both filters

The standard filter is built in the usual way.  The steady-state filter requires
only one extra call — [`compute_steady_state_filter_params`][cuthbertlib.kalman.filtering.compute_steady_state_filter_params]
— which performs the block-triangularization of `associative_params_single` **once**
and packages the constant results into a
[`SteadyStateFilterParams`][cuthbertlib.kalman.filtering.SteadyStateFilterParams] struct.

```{.python #steady-state-kalman-build}
def get_init_params(model_inputs):
    return m0, chol_P0


def get_dynamics_params(model_inputs):
    return F, c, chol_Q


def get_observation_params(model_inputs):
    return H, d, chol_R, ys[model_inputs - 1]


model_inputs = jnp.arange(len(ys) + 1)

# Standard filter
standard_filter = kalman.build_filter(
    get_init_params, get_dynamics_params, get_observation_params
)

# Steady-state filter: the only required change is passing ss_params.
ss_params = kalman.compute_steady_state_filter_params(F, chol_Q, H, chol_R)
steady_state_filter = kalman.build_filter(
    get_init_params, get_dynamics_params, get_observation_params,
    steady_state_params=ss_params,
)
```

!!! note "What does `compute_steady_state_filter_params` compute?"
    In the parallel associative scan each `FilterScanElement` carries constant
    matrices `A`, `U`, `Z` (and the gain `K`) that are identical at every time step
    for a time-invariant model.  `compute_steady_state_filter_params` extracts these
    by running the same block-QR as `associative_params_single` **once** (outside JIT),
    so that per-step `filter_prepare` calls need only evaluate the
    observation-dependent `b`, `eta`, and `ell` — purely matrix–vector products.

    This is distinct from the *Riccati* steady-state gain used in sequential
    filters; see [`SteadyStateFilterParams`][cuthbertlib.kalman.filtering.SteadyStateFilterParams]
    for details on the distinction.

### JIT-compile and verify correctness

```{.python #steady-state-kalman-run}
jitted_filter = jit(run_filter, static_argnames=["filter_obj", "parallel"])

# Warm up (triggers XLA compilation)
standard_result = jitted_filter(standard_filter, model_inputs, parallel=True)
jax.block_until_ready(standard_result.mean)

ss_result = jitted_filter(steady_state_filter, model_inputs, parallel=True)
jax.block_until_ready(ss_result.mean)

# Both filters compute mathematically identical operations; any difference is
# purely float32 rounding from evaluating the same QR inside vs. outside vmap.
max_abs_err = float(jnp.max(jnp.abs(standard_result.mean - ss_result.mean)))
mean_magnitude = float(jnp.mean(jnp.abs(standard_result.mean)))
rel_err = max_abs_err / (mean_magnitude + 1e-8)
print(f"Max absolute difference in posterior means: {max_abs_err:.2e}  (float32 rounding)")
print(f"Relative error (max abs / mean magnitude):  {rel_err:.2e}")
assert rel_err < 5e-2, (
    f"Steady-state and standard filters disagree beyond float32 rounding: "
    f"relative error = {rel_err:.2e}"
)
```

### Benchmark

We compare wall-clock time for both **parallel** (associative scan,
$\mathcal{O}(\log T)$ depth) and **sequential** ($\mathcal{O}(T)$) modes.

In the parallel case the bottleneck is the `filtering_operator` combine step —
which also contains `tria` calls — so the gain from skipping `filter_prepare`'s
QR is modest.  In the sequential case every `filter_prepare` call is on the
critical path, so the savings are larger.

```{.python #steady-state-kalman-benchmark}
REPS = 30


def time_filter(filter_obj, parallel):
    def run():
        result = jitted_filter(filter_obj, model_inputs, parallel=parallel)
        jax.block_until_ready(result.mean)
    return timeit.timeit(run, number=REPS) / REPS


# Warm up sequential variants
_ = jitted_filter(standard_filter, model_inputs, parallel=False)
jax.block_until_ready(_.mean)
_ = jitted_filter(steady_state_filter, model_inputs, parallel=False)
jax.block_until_ready(_.mean)

t_par_std = time_filter(standard_filter,     parallel=True)
t_par_ss  = time_filter(steady_state_filter, parallel=True)
t_seq_std = time_filter(standard_filter,     parallel=False)
t_seq_ss  = time_filter(steady_state_filter, parallel=False)

print(f"\nBenchmark over {NUM_STEPS} time steps, nx={NX}, ny={NY} ({REPS} reps each)")
print(f"  {'':30s}  {'standard':>12s}  {'steady-state':>12s}  {'speed-up':>9s}")
print(f"  {'parallel (associative scan)':30s}  {t_par_std*1e3:12.2f}  {t_par_ss*1e3:12.2f}  {t_par_std/t_par_ss:9.2f}x")
print(f"  {'sequential (scan)':30s}  {t_seq_std*1e3:12.2f}  {t_seq_ss*1e3:12.2f}  {t_seq_std/t_seq_ss:9.2f}x")
```

### Observed output

#### CPU (float32)

```text
Max absolute difference in posterior means: 2.81e-03  (float32 rounding)
Relative error (max abs / mean magnitude):  1.05e-03

Benchmark over 5000 time steps, nx=20, ny=4 (30 reps each)
                                      standard  steady-state   speed-up
  parallel (associative scan)           668.16        621.28       1.08x
  sequential (scan)                     344.74        278.22       1.24x
```

#### GPU (float64)

```text
Max absolute difference in posterior means: 3.69e-12
Relative error (max abs / mean magnitude):  1.47e-12

Benchmark over 5000 time steps, nx=20, ny=4 (30 reps each)
                                      standard  steady-state   speed-up
  parallel (associative scan)            85.27         80.65       1.06x
  sequential (scan)                    5452.67       4225.71       1.29x
```

## Key Takeaways

- **One extra line**: the only change from a standard filter is calling
  `compute_steady_state_filter_params` once and passing the result to
  `build_filter`.
- **No approximation**: results are mathematically identical to the standard
  filter (differences are float32 rounding only; vanish with float64).
- **Where the speedup is**: the QR decomposition is eliminated from every
  `filter_prepare` call.  The gain is most visible in sequential mode and
  grows with state / observation dimension and on GPU.
- **Parallel-scan vs Riccati**: `compute_steady_state_filter_params` computes
  the *parallel-scan* gain (prior = $Q$, single block-QR), not the Riccati
  steady-state gain.  For the sequential `update` function, supply a
  `SteadyStateFilterParams` built from the DARE solution instead.

## Next Steps

- **Kalman tracking**: See [the car tracking example](kalman_tracking.md) for a
  gentle introduction to `cuthbert`'s filter API.
- **Temporal parallelization**: The [temporal parallelization example](temporal_parallelization_kalman.md)
  explains the parallel associative scan in more depth.
- **Parameter estimation**: Combine with gradient-based optimization — see the
  [EM example](parameter_estimation_em.md).



<!--- entangled-tangle-block
```{.python file=examples_scripts/steady_state_kalman.py}
<<steady-state-kalman-imports>>
<<steady-state-kalman-model>>
<<steady-state-kalman-build>>
<<steady-state-kalman-run>>
<<steady-state-kalman-benchmark>>
```
-->
