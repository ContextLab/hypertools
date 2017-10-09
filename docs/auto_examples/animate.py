# -*- coding: utf-8 -*-
"""
=============================
Hyperalign a list of arrays and create an animated plot
=============================

The sample data is a list of 2D arrays, where each array is fMRI brain activity
from one subject.  The rows are timepoints and the columns are neural
'features'.  First, the matrices are hyperaligned using hyp.tools.align.
"""

# Code source: Andrew Heusser
# License: MIT

# import
import hypertools as hyp
import numpy as np

# load example data
<<<<<<< HEAD
data = hyp.load('weights', align=True)
=======
data = hyp.tools.load('weights', align=True)
>>>>>>> 44fe07e96e8f109b3023a70c8716b20c71f07764

# average into two groups
w1 = np.mean(data[:17],0)
w2 = np.mean(data[18:],0)

# plot
hyp.plot([w1, w2], animate=True)
