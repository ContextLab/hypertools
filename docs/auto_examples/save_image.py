# -*- coding: utf-8 -*-
"""
=============================
Saving a plot
=============================

To save a plot, simply use the `save_path` kwarg, and specify where you want
the image to be saved, including the file extension (e.g. pdf)
"""

# Code source: Andrew Heusser
# License: MIT

import hypertools as hyp
import scipy.io as sio
import numpy as np

data = hyp.tools.load('weights')
w=[i for i in data[0:3]]

hyp.plot(w,'o', save_path='test-image.pdf')
