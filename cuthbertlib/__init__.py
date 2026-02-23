from importlib.metadata import version

__version__ = version("cuthbertlib")
del version

from cuthbertlib import (
    discrete,
    kalman,
    linalg,
    linearize,
    quadrature,
    resampling,
    smc,
    stats,
    types,
)
