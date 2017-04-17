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

import hypertools as hyp
import scipy.io as sio
import numpy as np

data = hyp.tools.load('weights_sample')

hyp.plot(data, 'o', explore=True)
