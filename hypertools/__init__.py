from .plot import plot
from .reduce import reduce
from .align import align
from .cluster import cluster
from .manip import manip
from .core import get_default_options, apply_model as analyze
from .io import load, save
from .core.configurator import __version__

__version__ = str(__version__).split()[1]