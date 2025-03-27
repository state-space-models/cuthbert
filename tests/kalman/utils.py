import numpy as np


def generate_lgssm(seed, x_dim, y_dim, num_time_steps):
    rng = np.random.default_rng(seed)

    # Init, transition and observation models, and observations.
    m0, chol_P0 = generate_init_model(rng, x_dim)
    F, c, chol_Q = generate_trans_model(rng, x_dim)
    H, d, chol_R, y = generate_obs_model(rng, x_dim, y_dim)

    # Make copies for every time step.
    Fs, cs, chol_Qs, Hs, ds, chol_Rs, ys = batch_arrays(
        num_time_steps, F, c, chol_Q, H, d, chol_R, y
    )
    return m0, chol_P0, Fs, cs, chol_Qs, Hs, ds, chol_Rs, ys


def generate_cholesky_factor(rng, dim):
    chol_A = rng.random((dim, dim))
    chol_A[np.triu_indices(dim, 1)] = 0.0
    return chol_A


def generate_init_model(rng, x_dim):
    m0 = rng.normal(size=x_dim)
    chol_P0 = generate_cholesky_factor(rng, x_dim)
    return m0, chol_P0


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


def batch_arrays(t, *args):
    out = []
    for arg in args:
        out.append(np.repeat(arg[None], t, axis=0))
    return out
