import numpy as np

from cuthbertlib.resampling.utils import inverse_cdf_cpu


def main():
    # Dummy small inputs
    uniforms = np.linspace(0, 1, 64, dtype=np.float32)
    logits = np.random.randn(64).astype(np.float32)

    # Trigger Numba compilation
    inverse_cdf_cpu(uniforms, logits)


if __name__ == "__main__":
    main()
