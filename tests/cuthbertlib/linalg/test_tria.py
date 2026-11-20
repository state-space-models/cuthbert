import jax
import jax.numpy as jnp
import pytest
from jax import random

from cuthbertlib.linalg.tria import tria


@pytest.fixture(scope="module", autouse=True)
def config():
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", False)


@pytest.mark.parametrize("seed", [0, 42, 99])
@pytest.mark.parametrize("shape", [(3, 2), (4, 4), (4, 5)])
def test_tria(seed, shape):
    key = random.key(seed)
    A = random.normal(key, shape)

    R = tria(A)

    # Check that R is lower triangular
    assert jnp.allclose(R, jnp.tril(R))

    # Check that R @ R.T = A @ A.T
    assert jnp.allclose(R @ R.T, A @ A.T)


def test_tria_jvp_preserves_gram_matrix_for_rank_deficient_input():
    A = jnp.array([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]])
    dA = jnp.array([[0.0, 0.0], [1.0, 2.0], [3.0, 4.0]])

    def gram_via_tria(x):
        R = tria(x)
        return R @ R.T

    def gram_direct(x):
        return x @ x.T

    gram_jvp_from_tria = jax.jvp(gram_via_tria, (A,), (dA,))[1]
    gram_jvp_direct = jax.jvp(gram_direct, (A,), (dA,))[1]

    assert jnp.allclose(gram_jvp_from_tria, gram_jvp_direct)


def _rank_deficient(key, n, m, rank):
    k1, k2 = random.split(key)
    return random.normal(k1, (n, rank)) @ random.normal(k2, (rank, m))


@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("n_lower", [0, 1, 2])
@pytest.mark.parametrize("rank", [3, 4, 5])
def test_tria_jvp_gramian_identity_and_structure(seed, n_lower, rank):
    n, m = 5, 7
    k1, k2 = random.split(random.key(seed))
    A = _rank_deficient(k1, n, m, rank)
    dA = random.normal(k2, (n, m))

    L, dL = jax.jvp(lambda x: tria(x, n_lower), (A,), (dA,))

    # Gramian differential identity at any rank
    assert jnp.allclose(dL @ L.T + L @ dL.T, dA @ A.T + A @ dA.T)

    # Leading block of the tangent is lower triangular and the upper-right block is zero
    assert jnp.allclose(dL[:n_lower, :], jnp.tril(dL[:n_lower, :]))


@pytest.mark.parametrize("seed", [0, 42])
@pytest.mark.parametrize("shape", [(4, 4), (4, 6)])
@pytest.mark.parametrize("n_lower", [0, 2, 4])
def test_tria_jvp_matches_qr_rule_at_full_rank(seed, shape, n_lower):
    k1, k2 = random.split(random.key(seed))
    A = random.normal(k1, shape)
    dA = random.normal(k2, shape)

    def tria_qr(x):
        return jnp.linalg.qr(x.T, mode="reduced")[1].T

    _, dL = jax.jvp(lambda x: tria(x, n_lower), (A,), (dA,))
    _, dL_qr = jax.jvp(tria_qr, (A,), (dA,))
    assert jnp.allclose(dL, dL_qr)


def test_kalman_gradient_with_rotating_singular_observation_noise():
    # V = v v^T with v = (1, c) is rank one for every c and its column space
    # rotates with c, so the update matrix is rank deficient at every step.
    from jax.scipy.stats import multivariate_normal

    from cuthbertlib.kalman.filtering import predict, update

    dx, dy, T = 4, 2, 20
    A = 0.9 * jnp.eye(dx) + 0.05 * jnp.diag(jnp.ones(dx - 1), 1)
    B = jnp.concatenate([jnp.eye(dy), jnp.ones((dy, dx - dy))], axis=1)
    chol_U = 0.3 * jnp.eye(dx)

    def chol_V(c):
        return jnp.array([[1.0, 0.0], [c, 0.0]])

    def simulate(c, key):
        def step(x, k):
            k1, k2 = random.split(k)
            x = A @ x + chol_U @ random.normal(k1, (dx,))
            return x, B @ x + chol_V(c) @ random.normal(k2, (dy,))

        return jax.lax.scan(step, jnp.zeros(dx), random.split(key, T))[1]

    ys = simulate(0.4, random.key(3))

    def loglik_sqrt(c):
        def step(carry, y):
            m, chol_P, ell = carry
            m, chol_P = predict(m, chol_P, A, jnp.zeros(dx), chol_U)
            (m, chol_P), ell_t = update(m, chol_P, B, jnp.zeros(dy), chol_V(c), y)
            return (m, chol_P, ell + ell_t), None

        return jax.lax.scan(step, (jnp.zeros(dx), jnp.eye(dx), 0.0), ys)[0][2]

    def loglik_cov(c):
        V = chol_V(c) @ chol_V(c).T

        def step(carry, y):
            m, P, ell = carry
            m, P = A @ m, A @ P @ A.T + chol_U @ chol_U.T
            S = B @ P @ B.T + V
            K = P @ B.T @ jnp.linalg.inv(S)
            ell = ell + multivariate_normal.logpdf(y, B @ m, S)
            return (m + K @ (y - B @ m), P - K @ S @ K.T, ell), None

        return jax.lax.scan(step, (jnp.zeros(dx), jnp.eye(dx), 0.0), ys)[0][2]

    assert jnp.allclose(loglik_sqrt(0.4), loglik_cov(0.4))
    assert jnp.allclose(jax.grad(loglik_sqrt)(0.4), jax.grad(loglik_cov)(0.4))
