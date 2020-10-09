#!/usr/bin/env python
from .config import __version__
from .plot.plot import plot
from .plot.backend import set_backend
from .tools.load import load
from .tools.analyze import analyze
from .tools.reduce import reduce
from .tools.align import align
from .tools.normalize import normalize
from .tools.describe import describe
from .tools.cluster import cluster
from .datageometry import DataGeometry
