from jax import numpy as jnp, random, tree
from jax.lax import scan

from kalman import kalman


# TODO: Make tests independent, cleaner and more comprehensive
# TODO: Add tests when inputs is used, i.e. time inhomogeneous dynamics


def test_kalman():
    # Non-time-varying parameters
    m0 = jnp.array([0.0, 0.0, 0.0])
    C0 = jnp.diag(jnp.array([0.1, 0.2, 0.3]))

    F = jnp.diag(jnp.array([0.5, 1.0, 1.5]))
    c = jnp.array([0.1, 0.2, 0.3])
    Q = jnp.diag(jnp.array([0.1, 0.2, 0.3]))

    H = jnp.array([[0.5, 0.0, 0.5], [0.0, 1.0, 0.0]])
    d = jnp.array([-0.1, -0.2])
    R = jnp.diag(jnp.array([0.1, 0.2]))

    def init_params(inputs):
        return m0, C0

    def dynamics_params(inputs):
        return c, F, Q

    def observation_params(inputs):
        return d, H, R

    # Simulate some data
    K = 10
    us = jnp.zeros(K + 1)
    keys = random.split(random.PRNGKey(0), 2 * K + 1)
    xs = [random.multivariate_normal(keys[0], m0, C0)]
    ys = []
    for k in range(K):
        x = F @ xs[-1] + c + random.multivariate_normal(keys[k + 1], jnp.zeros(3), Q)
        xs.append(x)

        y = H @ x + d + random.multivariate_normal(keys[k + 1 + K], jnp.zeros(2), R)
        ys.append(y)

    xs = jnp.stack(xs)
    ys = jnp.stack(ys)

    # Check init
    init_state = kalman.init(us[0], init_params)
    assert jnp.allclose(init_state.mean, m0)
    assert jnp.allclose(init_state.cov, C0)

    # Check predict
    predict_state = kalman.predict(init_state, us[1], dynamics_params)
    assert jnp.allclose(predict_state.mean, c + F @ m0)
    assert jnp.allclose(predict_state.cov, F @ C0 @ F.T + Q)

    # Check online filter
    kal_gain = predict_state.cov @ H.T @ jnp.linalg.inv(H @ predict_state.cov @ H.T + R)
    update_mean = predict_state.mean + kal_gain @ (ys[0] - H @ predict_state.mean - d)
    update_cov = predict_state.cov - kal_gain @ H @ predict_state.cov

    update_state = kalman.online_filter(
        init_state,
        us[1],
        ys[0],
        dynamics_params,
        observation_params,
    )
    assert jnp.allclose(update_state.mean, update_mean)
    assert jnp.allclose(update_state.cov, update_cov)

    # Check offline filter
    def filter_scan_body(state, k):
        new_state = kalman.online_filter(
            state,
            us[k],
            ys[k - 1],
            dynamics_params,
            observation_params,
        )
        return new_state, new_state

    expected_offline_filter_states = scan(
        filter_scan_body, init_state, jnp.arange(1, K + 1)
    )[1]
    expected_offline_filter_states = tree.map(
        lambda x0, xs: jnp.vstack([x0[jnp.newaxis], xs]),
        init_state,
        expected_offline_filter_states,
    )
    offline_filter_states = kalman.offline_filter(
        us, ys, init_params, dynamics_params, observation_params
    )

    assert offline_filter_states.mean.shape == (K + 1,) + m0.shape
    assert offline_filter_states.cov.shape == (K + 1,) + C0.shape
    assert jnp.allclose(expected_offline_filter_states.mean, offline_filter_states.mean)
    assert jnp.allclose(expected_offline_filter_states.cov, offline_filter_states.cov)

    # Check smoother
    smoother_states = kalman.smoother(offline_filter_states, us, dynamics_params)

    def smoother_body_scan(state, k):
        dynamics_shift, F, Q = dynamics_params(us[k + 1])

        filter_mean = offline_filter_states.mean[k]
        filter_cov = offline_filter_states.cov[k]

        K = filter_cov @ F.T @ jnp.linalg.inv(F @ filter_cov @ F.T + Q)
        smoother_mean = filter_mean + K @ (
            state.mean - F @ filter_mean - dynamics_shift
        )
        cross_cov = K @ state.cov
        smoother_cov = filter_cov + K @ (state.cov - F @ filter_cov @ F.T - Q) @ K.T
        smoother_state = kalman.KalmanState(mean=smoother_mean, cov=smoother_cov)
        return smoother_state, (smoother_state, cross_cov)

    final_smoother_state = tree.map(lambda x: x[-1], offline_filter_states)
    expected_smoother_states, cross_covs = scan(
        smoother_body_scan,
        final_smoother_state,
        jnp.arange(K),
        reverse=True,
    )[1]

    expected_smoother_states = tree.map(
        lambda xs, xf: jnp.vstack([xs, xf[jnp.newaxis]]),
        expected_smoother_states,
        final_smoother_state,
    )
    assert smoother_states.mean.shape == (K + 1,) + m0.shape
    assert smoother_states.cov.shape == (K + 1,) + C0.shape
    assert jnp.allclose(smoother_states.mean[-1], offline_filter_states.mean[-1])
    assert jnp.allclose(smoother_states.cov[-1], offline_filter_states.cov[-1])
    assert jnp.allclose(expected_smoother_states.mean, smoother_states.mean, atol=1e-5)
    assert jnp.allclose(expected_smoother_states.cov, smoother_states.cov)
    assert smoother_states.cross_cov is not None
    assert jnp.allclose(cross_covs, smoother_states.cross_cov)
