#!/usr/bin/env python

from __future__ import division
import sys
import warnings
import re
import itertools

import seaborn as sns

from .helpers import *
from .static import plot_coords as static_plot
from .animate import animate as animated_plot

def plot(x,*args,**kwargs):

    ##STYLING##
    if 'style' in kwargs:
        sns.set(style=kwargs['style'])
        del kwargs['style']
    else:
        sns.set(style="whitegrid")

    if 'palette' in kwargs:
        sns.set_palette(palette=kwargs['palette'], n_colors=len(x))
        del kwargs['palette']
    else:
        sns.set_palette(palette="hls", n_colors=len(x))

    if 'animate' in kwargs:
        animate=kwargs['animate']
        del kwargs['animate']
    else:
        animate=False

    if animate:
        animated_plot(x)
    else:
        static_plot(x,*args,**kwargs)
