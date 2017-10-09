# -*- coding: utf-8 -*-
"""
=============================
Explore mode!
=============================

Explore mode is an experimental feature that allows you to (not surprisingly)
explore the points in your dataset.  When you hover over the points, a label
will pop up that will help you identify the datapoint.  You can customize the
labels by passing a list of labels to the `label(s)` kwarg. Alternatively, if
you don't pass a list of labels, the labels will be the index of the datapoint,
along with the PCA coordinate.
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
hyp.plot(data, '.', explore=True)
