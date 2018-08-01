# -*- coding: utf-8 -*-
"""
=============================
Animated plots
=============================

Timeseries plots can be animated by simply passing `animate=True` to the geo (
or when calling hyp.plot).
"""

# Code source: Andrew Heusser
# License: MIT

# import
import hypertools as hyp

# load example data
geo = hyp.load('weights_avg')

# plot
geo.plot(animate=True, legend=['first', 'second'])
