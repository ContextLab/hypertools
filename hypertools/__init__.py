#!/usr/bin/env python

from builtins import object
from .plot.plot import plot
from .tools import *

class hypertools(object):
    '''Hypertools module'''

    def __init__(self, plot=plot, tools=tools):
        self.plot = plot
        self.tools = tools
