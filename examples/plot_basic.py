# -*- coding: utf-8 -*-
"""
=============================
A basic example
=============================

Here is a basic example where we load in some data (a list of arrays - samples
by features), take the first two arrays in the list and plot them as points
with the 'o'.  Hypertools can handle all format strings supported by matplotlib.
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
hyp.plot(data, '.')
