"""Utilities to generate linear-Gaussian state-space models (LGSSMs)."""

from functools import partial

import jax
import jax.numpy as jnp
from jax import Array, random

from cuthbertlib.types import KeyArray


@partial(jax.jit, static_argnames=("x_dim", "y_dim", "num_time_steps"))
def generate_lgssm(seed: int, x_dim: int, y_dim: int, num_time_steps: int):
    """Generates an LGSSM along with a set of observations."""
    key = random.key(seed)

    key, init_key, sample_key, obs_model_key, obs_key = random.split(key, 5)
    m0, chol_P0 = generate_init_model(init_key, x_dim)
    x0 = m0 + chol_P0 @ random.normal(sample_key, (x_dim,))

    def body(_x, _key):
        trans_model_key, trans_key, obs_model_key, obs_key = random.split(_key, 4)

        F, c, chol_Q = generate_trans_model(trans_model_key, x_dim)
        state_noise = chol_Q @ random.normal(trans_key, (x_dim,))
        x = F @ _x + c + state_noise

        H, d, chol_R = generate_obs_model(obs_model_key, x_dim, y_dim)
        obs_noise = chol_R @ random.normal(obs_key, (y_dim,))
        y = H @ x + d + obs_noise

        return x, (F, c, chol_Q, H, d, chol_R, y)

    scan_keys = random.split(key, num_time_steps)
    _, (Fs, cs, chol_Qs, Hs, ds, chol_Rs, ys) = jax.lax.scan(body, x0, scan_keys)

    return m0, chol_P0, Fs, cs, chol_Qs, Hs, ds, chol_Rs, ys


def generate_cholesky_factor(key: KeyArray, dim: int) -> Array:
    """Generates a random Cholesky factor (lower-triangular matrix)."""
    chol_A = random.uniform(key, (dim, dim))
    chol_A = chol_A.at[jnp.triu_indices(dim, 1)].set(0.0)
    return chol_A


def generate_init_model(key: KeyArray, x_dim: int) -> tuple[Array, Array]:
    """Generates a random initial state for an LGSSM."""
    keys = random.split(key)
    m0 = random.normal(keys[0], (x_dim,))
    chol_P0 = generate_cholesky_factor(keys[1], x_dim)
    return m0, chol_P0


def generate_trans_model(key: KeyArray, x_dim: int) -> tuple[Array, Array, Array]:
    """Generates a random transition model for an LGSSM."""
    keys = random.split(key, 3)
    exp_eig_max = 0.75  # Chosen less than one to stop exploding states (in expectation)
    F = exp_eig_max * random.normal(keys[0], (x_dim, x_dim)) / jnp.sqrt(x_dim)
    c = 0.1 * random.normal(keys[1], (x_dim,))
    chol_Q = generate_cholesky_factor(keys[2], x_dim)
    return F, c, chol_Q


def generate_obs_model(
    key: KeyArray, x_dim: int, y_dim: int
) -> tuple[Array, Array, Array]:
    """Generates a random observation model for an LGSSM."""
    keys = random.split(key, 3)
    H = random.normal(keys[0], (y_dim, x_dim))
    d = random.normal(keys[1], (y_dim,))
    chol_R = generate_cholesky_factor(keys[2], y_dim)
    return H, d, chol_R
