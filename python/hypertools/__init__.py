#!/usr/bin/env python

from .plot import plot
from .align import align
from .reduce import reduce
from .describe import describe
from .procrustean import Procrustes

class hypertools(object):
    '''Hypertools module'''

    def __init__(self, plot=plot, align=align, reduce=reduce, describe=describe, procrustes=Procrustes):
        self.plot = plot.plot
        self.align = align.align
        self.reduce = reduce.reduce
        self.describe = describe.describe
        self.Procrustes = Procrustes
