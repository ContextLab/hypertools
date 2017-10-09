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

# import
import hypertools as hyp

# load example data
<<<<<<< HEAD
data = hyp.load('weights_sample')
=======
data = hyp.tools.load('weights_sample')
>>>>>>> 44fe07e96e8f109b3023a70c8716b20c71f07764

# plot
hyp.plot(data, '.', animate='spin')
