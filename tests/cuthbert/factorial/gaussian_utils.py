import jax.numpy as jnp
from jax import random, vmap

from cuthbertlib.kalman import generate


def generate_factorial_kalman_model(
    seed, x_dim, y_dim, num_factors, num_factors_local, num_time_steps
):
    # T = num_time_steps, F = num_factors

    key = random.key(seed)
    init_key, factorial_indices_key = random.split(key, 2)

    # m0 with shape (F, x_dim)
    # chol_P0 with shape (F, x_dim, x_dim)
    init_keys_factorial = random.split(init_key, num_factors)
    m0s, chol_P0s = vmap(generate.generate_init_model, in_axes=(0, None))(
        init_keys_factorial, x_dim
    )

    # Fs with shape (T, num_factors_local * x_dim, num_factors_local * x_dim)
    # cs with shape (T, num_factors_local * x_dim)
    # chol_Qs with shape (T, num_factors_local * x_dim, num_factors_local * x_dim)
    # Hs with shape (T, d_y, num_factors_local * x_dim)
    # ds with shape (T, y_dim)
    # chol_Rs with shape (T, num_factors_local * y_dim, num_factors_local * y_dim)
    # ys with shape (T, d_y)
    _, _, Fs, cs, chol_Qs, Hs, ds, chol_Rs, ys = generate.generate_lgssm(
        seed + 1, num_factors_local * x_dim, y_dim, num_time_steps
    )

    # factorial_indices with shape (T, num_factors_local)
    # Each entry is a random integer in {0, ..., num_factors - 1}
    # But each row must have unique entries
    def rand_unique_indices(key):
        indices = random.choice(
            key, jnp.arange(num_factors), (num_factors_local,), replace=False
        )
        return indices

    factorial_indices_keys = random.split(factorial_indices_key, num_time_steps)
    factorial_indices = vmap(rand_unique_indices)(factorial_indices_keys)
    return m0s, chol_P0s, Fs, cs, chol_Qs, Hs, ds, chol_Rs, ys, factorial_indices
