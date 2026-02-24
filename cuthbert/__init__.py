from importlib.metadata import version

__version__ = version("cuthbert")
del version

from cuthbert import discrete, gaussian, smc
from cuthbert.filtering import filter
from cuthbert.inference import (
    Filter,
    FilterCombine,
    FilterPrepare,
    InitPrepare,
    Smoother,
    SmootherCombine,
    SmootherPrepare,
)
from cuthbert.smoothing import smoother
