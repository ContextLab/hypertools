# -*- coding: utf-8 -*-
"""
=============================
Chemtrails
=============================

The past trajectory of an animated plot can be visualized with the chemtrails
argument.  This displays a low opacity version of the trace behind the
current points being plotted.  This can be used in conjunction with the
precog argument to plot a low-opacity trace of the entire timeseries.
"""

# Code source: Andrew Heusser
# License: MIT

# import
import hypertools as hyp

# load example data
geo = hyp.load('weights_avg')

# plot
geo.plot(animate=True, chemtrails=True)
