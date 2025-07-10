# -*- coding: utf-8 -*-
"""
=============================
Precognition
=============================

The future trajectory of an animated plot can be visualized with the precog
argument.  This displays a low opacity version of the trace ahead of the
current points being plotted.  This can be used in conjunction with the
chemtrails argument to plot a low-opacity trace of the entire timeseries.
"""

# sphinx_gallery_thumbnail_path = '_static/thumbnails/sphx_glr_precog_thumb.gif'

# Code source: Andrew Heusser
# License: MIT

# import
import hypertools as hyp
import numpy as np

# load example data
geo = hyp.load('weights_avg')

# plot
geo.plot(animate=True, precog=True)
