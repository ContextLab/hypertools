from .plot import plot
from .reduce import reduce
from .align import align, pad, trim_and_pad, Aligner, HyperAlign, Procrustes, SharedResponseModel,\
    RobustSharedResponseModel, DeterministicSharedResponseModel, NullAlign
from .cluster import cluster
from .manip import manip
from .core import get_default_options, apply_model as analyze, RobustDict
from .core.configurator import __version__
from .io import load, save

__version__ = str(__version__).split()[1]
