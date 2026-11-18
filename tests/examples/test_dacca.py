"""Check the Dacca input, parameter transformation and observation floor."""

import hashlib
import importlib.util
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

EXAMPLE = Path(__file__).resolve().parents[2] / "examples/dacca"
spec = importlib.util.spec_from_file_location("dacca_model", EXAMPLE / "model.py")
model = importlib.util.module_from_spec(spec)
spec.loader.exec_module(model)


def test_input_identity_and_dimensions():
    assert hashlib.sha256((EXAMPLE / "data.npz").read_bytes()).hexdigest() == (
        "71446b8d04f4790bb636b17c15d51a9ce8984294a6ceceda500a3aef05bf9ba2"
    )
    values = model.inputs(600)
    assert values["y"].shape == (600,)
    assert values["cov"].shape == (600, 20, 9)
    assert values["dt"].shape == (600, 20)
    assert np.isfinite(np.asarray(values["cov"])).all()
    assert np.asarray(values["dt"]).min() > 0


def test_parameter_transform_and_initial_population():
    natural = np.asarray(model.from_est(model.PUBLISHED))
    expected = model.DATA["theta_published"][0].copy()
    indices = [model.INDEX[n] for n in model.IVPS]
    expected[indices] /= expected[indices].sum()
    np.testing.assert_allclose(natural, expected, rtol=2e-6, atol=2e-7)
    state = np.asarray(model.initial_state(jnp.asarray(natural)))
    assert state.shape == (8,)
    np.testing.assert_allclose(
        state[[0, 1, 2, 4, 5, 6]].sum(), model.DATA["covars"][0, 2], rtol=2e-6
    )
    assert state[3] == state[7] == 0


def test_violation_counter_gives_density_floor():
    theta = model.from_est(model.PUBLISHED)
    state = model.initial_state(theta).at[7].set(1)
    np.testing.assert_allclose(
        model.observation(theta, state, 2.0), model.LTOL, atol=2e-6
    )


@pytest.mark.parametrize("months", [0, 601, -1, 1.5, True])
def test_invalid_observation_count(months):
    with pytest.raises(ValueError, match="T must"):
        model.inputs(months)
