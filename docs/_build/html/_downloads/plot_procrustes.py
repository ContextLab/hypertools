# -*- coding: utf-8 -*-
"""
=============================
Aligning two matrices with the procrustes function
=============================

In this example, we load in some synthetic data, rotate it, and then use the
procustes function to get the datasets back in alignment.  The procrustes
function uses linear transformations to project a source matrix into the
space of a target matrix.
"""

# Code source: Andrew Heusser
# License: MIT

import hypertools as hyp
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt

data = hyp.tools.load('spiral')
target = data

# A random rotation matrix
rot = np.random.rand(3,3)

# creating new spiral with some noise
source = np.dot(target, rot)

# Before hyperalignment
hyp.plot([target, source], title='Before alignment')

# After hyperalignment
hyp.plot([hyp.tools.procrustes(source, target), target], ['-','--'], title='After alignment')
