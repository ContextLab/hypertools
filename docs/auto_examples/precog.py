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

# Code source: Andrew Heusser
# License: MIT

import hypertools as hyp
import scipy.io as sio
import numpy as np

data = hyp.tools.load('weights')
aligned_w = hyp.tools.align(data)

w1 = np.mean(aligned_w[:17],0)
w2 = np.mean(aligned_w[18:],0)

hyp.plot([w1, w2], animate=True, precog=True)
