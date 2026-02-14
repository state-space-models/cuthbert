import jax.numpy as jnp
from jax import random

from cuthbertlib.kalman import generate


def assert_is_cholesky_factor(chol):
    assert jnp.allclose(chol, jnp.tril(chol))
    assert jnp.all(jnp.diagonal(chol, axis1=-2, axis2=-1) != 0.0)


def test_generate_cholesky_factor():
    key = random.key(0)
    chol = generate.generate_cholesky_factor(key, 4)

    assert chol.shape == (4, 4)
    assert_is_cholesky_factor(chol)


def test_generate_init_model_shapes():
    key = random.key(0)
    m0, chol_P0 = generate.generate_init_model(key, 4)

    assert m0.shape == (4,)
    assert chol_P0.shape == (4, 4)
    assert_is_cholesky_factor(chol_P0)


def test_generate_trans_model():
    key = random.key(0)
    F, c, chol_Q = generate.generate_trans_model(key, 4)

    assert F.shape == (4, 4)
    assert c.shape == (4,)
    assert chol_Q.shape == (4, 4)
    assert_is_cholesky_factor(chol_Q)


def test_generate_obs_model():
    key = random.key(0)
    x_dim, y_dim = 4, 3
    H, d, chol_R = generate.generate_obs_model(key, x_dim, y_dim)

    assert H.shape == (y_dim, x_dim)
    assert d.shape == (y_dim,)
    assert chol_R.shape == (y_dim, y_dim)
    assert_is_cholesky_factor(chol_R)


def test_generate_lgssm_shapes():
    seed = 0
    x_dim = 4
    y_dim = 3
    num_time_steps = 5
    outputs = generate.generate_lgssm(seed, x_dim, y_dim, num_time_steps)

    (m0, chol_P0, Fs, cs, chol_Qs, Hs, ds, chol_Rs, ys) = outputs

    assert m0.shape == (x_dim,)
    assert chol_P0.shape == (x_dim, x_dim)
    assert_is_cholesky_factor(chol_P0)

    assert Fs.shape == (num_time_steps, x_dim, x_dim)
    assert cs.shape == (num_time_steps, x_dim)
    assert chol_Qs.shape == (num_time_steps, x_dim, x_dim)
    assert_is_cholesky_factor(chol_Qs)

    assert Hs.shape == (num_time_steps, y_dim, x_dim)
    assert ds.shape == (num_time_steps, y_dim)
    assert chol_Rs.shape == (num_time_steps, y_dim, y_dim)
    assert_is_cholesky_factor(chol_Rs)
    assert ys.shape == (num_time_steps, y_dim)

    # Ensure randomness of the parameters
    assert not jnp.allclose(Fs[0], Fs[1])
    assert not jnp.allclose(cs[0], cs[1])
    assert not jnp.allclose(chol_Qs[0], chol_Qs[1])
    assert not jnp.allclose(Hs[0], Hs[1])
    assert not jnp.allclose(ds[0], ds[1])
    assert not jnp.allclose(chol_Rs[0], chol_Rs[1])
