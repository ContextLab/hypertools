#!/usr/bin/env python

from . import plot
from . import tools

class hypertools(object):
    '''Hypertools module'''

    def __init__(self, plot=plot, tools=tools):
        self.plot = plot
        self.tools = tools
