import time

import jax
import jax.numpy as jnp
import numpy as np

from cuthbertlib.kalman.filtering import filter

PLATFORM_NAME = "cuda"
jax.config.update("jax_platform_name", PLATFORM_NAME)


def generate_cholesky_factor(rng, dim):
    chol_A = rng.random((dim, dim))
    chol_A[np.triu_indices(dim, 1)] = 0.0
    return chol_A


def generate_trans_model(rng, x_dim):
    F = rng.random((x_dim, x_dim))
    b = rng.random(x_dim)
    chol_Q = generate_cholesky_factor(rng, x_dim)
    return F, b, chol_Q


def generate_obs_model(rng, x_dim, y_dim):
    H = rng.random((y_dim, x_dim))
    c = rng.random(y_dim)
    chol_R = generate_cholesky_factor(rng, y_dim)
    y = rng.random(y_dim)
    return H, c, chol_R, y


seed = 0
x_dim = 20
y_dim = 10
num_time_steps = 1000

offline_filter = jax.jit(filter, static_argnames="parallel")

rng = np.random.default_rng(seed)
m0 = rng.normal(size=x_dim)
chol_P0 = generate_cholesky_factor(rng, x_dim)
F, c, chol_Q = generate_trans_model(rng, x_dim)
H, d, chol_R, y = generate_obs_model(rng, x_dim, y_dim)


def batch_arrays(t, *args):
    out = []
    for arg in args:
        out.append(jnp.repeat(arg[None, ...], t, axis=0))
    return out


# Make copies for T time steps.
Fs, cs, chol_Qs, Hs, ds, chol_Rs, ys = batch_arrays(
    num_time_steps, F, c, chol_Q, H, d, chol_R, y
)

num_runs = 10
runtimes = []
for _ in range(num_runs):
    start_time = time.time()
    filt_states, ell = filter(
        m0, chol_P0, Fs, cs, chol_Qs, Hs, ds, chol_Rs, ys, parallel=True
    )
    jax.block_until_ready(filt_states)
    runtimes.append(time.time() - start_time)

print(f"Compile time: {runtimes[0]:.3f}s")
print(f"Runtime: {np.mean(runtimes[1:]):.3f} pm {np.std(runtimes[1:]):5f}s")
