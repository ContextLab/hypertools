# -*- coding: utf-8 -*-
"""
=============================
Create a rotating static plot
=============================

In addition to plotting dynamic timeseries data, the spin feature can be used to
visualize static data in an animated rotating plot.
"""

# Code source: Andrew Heusser
# License: MIT

# load
import hypertools as hyp

# load data
data = hyp.tools.load('weights_sample')

# plot
hyp.plot(data, '.', animate='spin')
