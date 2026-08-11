import chex
import jax
import jax.numpy as jnp
import pytest

from cuthbertlib.ensemble_kalman.smoothing import update


@pytest.fixture(scope="module", autouse=True)
def config():
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", False)


def test_update_affine_dynamics():
    filtered = jnp.array(
        [
            [-2.0, -1.0],
            [-1.0, 2.0],
            [0.0, 0.0],
            [1.0, -2.0],
            [2.0, 1.0],
        ]
    )
    F = jnp.array([[1.5, -0.25], [0.5, 2.0]])
    c = jnp.array([0.3, -0.7])
    predicted = filtered @ F.T + c
    next_smoothed = predicted + jnp.array(
        [
            [0.1, -0.2],
            [-0.3, 0.4],
            [0.5, 0.1],
            [-0.2, -0.3],
            [0.4, 0.2],
        ]
    )

    smoothed, gain = update(filtered, predicted, next_smoothed)
    expected_gain = jnp.linalg.inv(F)
    expected_smoothed = filtered + (next_smoothed - predicted) @ expected_gain.T

    chex.assert_trees_all_close(gain, expected_gain, rtol=1e-10, atol=1e-10)
    chex.assert_trees_all_close(smoothed, expected_smoothed, rtol=1e-10, atol=1e-10)

    unchanged, _ = update(filtered, predicted, predicted)
    chex.assert_trees_all_close(unchanged, filtered, rtol=0.0, atol=0.0)

    filtered_wide = jnp.array(
        [
            [-2.0, -1.0, 0.0, 1.0, 2.0],
            [1.0, -1.0, 2.0, -2.0, 0.0],
            [3.0, 2.0, -1.0, 0.0, -2.0],
        ]
    )
    predicted_wide = filtered_wide + jnp.array([0.3, -0.7, 0.2, 1.0, -0.5])
    predicted_wide_dev = predicted_wide - jnp.mean(predicted_wide, axis=0)
    next_smoothed_wide = predicted_wide + predicted_wide_dev

    smoothed_wide, gain_wide = update(
        filtered_wide,
        predicted_wide,
        next_smoothed_wide,
    )

    chex.assert_trees_all_close(
        smoothed_wide,
        filtered_wide + predicted_wide_dev,
        rtol=1e-10,
        atol=1e-10,
    )
    chex.assert_trees_all_close(gain_wide, gain_wide.T, rtol=1e-10, atol=1e-10)
    chex.assert_trees_all_close(
        gain_wide @ gain_wide,
        gain_wide,
        rtol=1e-10,
        atol=1e-10,
    )
