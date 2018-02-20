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

# import
import hypertools as hyp
import numpy as np
import scipy

# load example data
data, labels = hyp.tools.load('spiral')
target = data

# a random rotation matrix
rot = scipy.linalg.orth(np.random.rand(3,3))

# creating new spiral with some noise
source = np.dot(target, rot)

# before hyperalignment
hyp.plot([target, source], title='Before alignment')

# after hyperalignment
hyp.plot([hyp.tools.procrustes(source, target), target], ['-','--'], title='After alignment')
