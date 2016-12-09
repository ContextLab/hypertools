<<<<<<< HEAD
#!/usr/bin/env python

from .plot import *
from .align import *
from .reduce import *
from .describe import describe
=======
from .plot import *
from .align import *
from .reduce import *
# from .describe import describe
describe=[]
>>>>>>> 3254b4a6b6a76596c17f7b470a4e4ce9b6420de0

class hypertools(object):
    '''Hypertools module'''

    def __init__(self, plot=plot, align=align, reduce=reduce, describe=describe):
        self.plot = plot.plot
        self.align = align.align
        self.reduce = reduce.reduce
<<<<<<< HEAD
        self.describe = describe.describe
=======
        self.describe = describe
>>>>>>> 3254b4a6b6a76596c17f7b470a4e4ce9b6420de0
