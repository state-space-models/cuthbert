import jax.numpy as jnp
from jax import Array, random

from cuthbertlib.types import KeyArray


def generate_lgssm(seed: int, x_dim: int, y_dim: int, num_time_steps: int):
    key = random.key(seed)
    key, init_key, trans_key, obs_key = random.split(key, 4)

    # Initial, transition, and observation models.
    m0, chol_P0 = generate_init_model(init_key, x_dim)
    F, c, chol_Q = generate_trans_model(trans_key, x_dim)
    H, d, chol_R = generate_obs_model(obs_key, x_dim, y_dim)

    # Make copies for every time step.
    Fs, cs, chol_Qs, Hs, ds, chol_Rs = batch_arrays(
        num_time_steps, F, c, chol_Q, H, d, chol_R
    )

    # Simulate observations
    ys = []

    key, sub_key = random.split(key)
    x = m0 + chol_P0 @ random.normal(sub_key, (x_dim,), dtype=jnp.float64)

    for t in range(num_time_steps):
        key, state_key, obs_key = random.split(key, 3)
        state_noise = chol_Q @ random.normal(state_key, (x_dim,), dtype=jnp.float64)
        x = F @ x + c + state_noise
        obs_noise = chol_R @ random.normal(obs_key, (y_dim,), dtype=jnp.float64)
        y = H @ x + d + obs_noise
        ys.append(y)

    ys = jnp.asarray(ys)
    return m0, chol_P0, Fs, cs, chol_Qs, Hs, ds, chol_Rs, ys


def generate_cholesky_factor(key: KeyArray, dim: int) -> Array:
    chol_A = random.uniform(key, (dim, dim), dtype=jnp.float64)
    chol_A = chol_A.at[jnp.triu_indices(dim, 1)].set(0.0)
    return chol_A


def generate_init_model(key: KeyArray, x_dim: int) -> tuple[Array, Array]:
    keys = random.split(key)
    m0 = random.normal(keys[0], (x_dim,), dtype=jnp.float64)
    chol_P0 = generate_cholesky_factor(keys[1], x_dim)
    return m0, chol_P0


def generate_trans_model(key: KeyArray, x_dim: int) -> tuple[Array, Array, Array]:
    keys = random.split(key, 3)
    # Generate transition matrix F and vector c such that the system is stable
    F = jnp.eye(x_dim) + 0.1 * random.normal(keys[0], (x_dim, x_dim), dtype=jnp.float64)
    c = 0.1 * random.normal(keys[1], (x_dim,), dtype=jnp.float64)
    chol_Q = generate_cholesky_factor(keys[2], x_dim)
    return F, c, chol_Q


def generate_obs_model(
    key: KeyArray, x_dim: int, y_dim: int
) -> tuple[Array, Array, Array]:
    keys = random.split(key, 3)
    H = random.uniform(keys[0], (y_dim, x_dim), dtype=jnp.float64)
    d = random.uniform(keys[1], (y_dim,), dtype=jnp.float64)
    chol_R = generate_cholesky_factor(keys[2], y_dim)
    return H, d, chol_R


def batch_arrays(t, *args):
    out = []
    for arg in args:
        out.append(jnp.repeat(arg[None], t, axis=0))
    return out
