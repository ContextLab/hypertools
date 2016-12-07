# -*- coding: utf-8 -*-
from .plot import plot
from .align import align
from .reduce import reduce
# from .describe import describe
describe=[]

class Hypertools(object):
    '''Hypertools module'''

    def __init__(self, plot=plot, align=align, reduce=reduce, describe=describe):
        self.plot = plot
        self.align = align
        self.reduce = reduce
        self.describe = describe
