#!/usr/bin/env python

from .config import __version__
from .plot.plot import plot
from .plot.backend import set_interactive_backend
from .io.load import load
from .tools.analyze import analyze
from .reduce.reduce import reduce
from .tools.align import align
from .tools.normalize import normalize
from .reduce.describe import describe
from .cluster.cluster import cluster
from .core.model import apply_model
from .manip.manip import manip
from .predict.predict import predict
from .io.save import save
from . import io
