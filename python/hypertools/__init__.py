#!/usr/bin/env python

from plot.plot import plot
from util import *

class hypertools(object):
    '''Hypertools module'''

    def __init__(self, plot=plot, util=util):
        self.plot = plot
        self.util = util
