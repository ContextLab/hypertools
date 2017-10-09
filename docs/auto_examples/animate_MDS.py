# -*- coding: utf-8 -*-
"""
=============================
Animated trajectory plotted with multidimensional scaling
=============================

This is a trajectory of brain data, hyperaligned and then plotted in 3D
with multidimensional scaling.
"""

# Code source: Andrew Heusser
# License: MIT

import hypertools as hyp
<<<<<<< HEAD
import numpy as np

data = hyp.load('weights')
aligned_w = hyp.align(data)
=======
import scipy.io as sio
import numpy as np

data = hyp.tools.load('weights')
aligned_w = hyp.tools.align(data)
>>>>>>> 44fe07e96e8f109b3023a70c8716b20c71f07764

w1 = np.mean(aligned_w[:17],0)
w2 = np.mean(aligned_w[18:],0)

<<<<<<< HEAD
hyp.plot([w1, w2], animate=True, reduce='MDS')
=======
hyp.plot([w1, w2], animate=True, model='MDS')
>>>>>>> 44fe07e96e8f109b3023a70c8716b20c71f07764
