import jax
import jax.numpy as jnp
import pytest
from jax import Array, vmap


from cuthbert import factorial
from cuthbert.gaussian import kalman
from cuthbert.inference import Filter, Smoother
from cuthbertlib.kalman.generate import generate_lgssm
from tests.cuthbertlib.kalman.test_filtering import std_predict, std_update
from tests.cuthbertlib.kalman.test_smoothing import std_kalman_smoother


@pytest.fixture(scope="module", autouse=True)
def config():
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", False)


def load_kalman_pairwise_factorial_inference(
    m0: Array,  # (F, d)
    chol_P0: Array,  # (F, d, d)
    Fs: Array,  # (T, 2 * d, 2 * d)
    cs: Array,  # (T, 2 * d)
    chol_Qs: Array,  # (T, 2 * d, 2 * d)
    Hs: Array,  # (T, 2 * d, d_y)
    ds: Array,  # (T, d_y)
    chol_Rs: Array,  # (T, d_y, d_y)
    ys: Array,  # (T + 1, d_y)
    factorial_indices: Array,  # (T, 2)
) -> tuple[Filter, Smoother, Array]:
    """Builds Kalman filter and smoother objects and model_inputs for a linear-Gaussian SSM."""

    def get_init_params(model_inputs: int) -> tuple[Array, Array]:
        return m0, chol_P0

    def get_dynamics_params(model_inputs: int) -> tuple[Array, Array, Array]:
        return Fs[model_inputs - 1], cs[model_inputs - 1], chol_Qs[model_inputs - 1]

    def get_observation_params(model_inputs: int) -> tuple[Array, Array, Array, Array]:
        return (
            Hs[model_inputs],
            ds[model_inputs],
            chol_Rs[model_inputs],
            ys[model_inputs],
        )

    def extract_and_join(factorial_state, model_inputs):
        fac_inds = factorial_indices[model_inputs - 1]
        
        means = 
        
        

    filter = kalman.build_filter(
        get_init_params, get_dynamics_params, get_observation_params
    )
    smoother = kalman.build_smoother(
        get_dynamics_params, store_gain=True, store_chol_cov_given_next=True
    )
    model_inputs = jnp.arange(len(ys))
    return filter, smoother, model_inputs
