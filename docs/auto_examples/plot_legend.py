# -*- coding: utf-8 -*-
"""
=============================
Generating a legend
=============================

An example of how to use the `legend` kwarg to generate a legend.
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
hyp.plot(data, '.', legend=['Group A', 'Group B', 'Group C'])
