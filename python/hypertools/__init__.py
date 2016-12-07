from .plot import *
from .align import *
from .reduce import *
# from .describe import describe
describe=[]

class hypertools(object):
    '''Hypertools module'''

    def __init__(self, plot=plot, align=align, reduce=reduce, describe=describe):
        self.plot = plot.plot
        self.align = align.align
        self.reduce = reduce.reduce
        self.describe = describe
