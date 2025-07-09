# -*- coding: utf-8 -*-
"""
=============================
Plotting State of the Union Addresses from 1989-2017
=============================

To plot text, simply pass the text data to the plot function.  Here, we are
ploting each SOTU address fit to a topic model, and then reduced to visualize.
By default, hypertools transforms the text data using a model fit to a selected
set of wikipedia pages.

"""

# Code source: Andrew Heusser
# License: MIT

# load hypertools
import hypertools as hyp

# load the data
geo = hyp.load('sotus')

# plot it - handle both DataGeometry and Pipeline objects
if hasattr(geo, 'plot'):
    geo.plot()
else:
    # If it's a Pipeline or other object, use hyp.plot directly
    hyp.plot(geo)
