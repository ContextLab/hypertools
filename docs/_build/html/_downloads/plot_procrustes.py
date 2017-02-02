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

data = sio.loadmat('sample_data/test_data.mat')
target = data['spiral']

# A random rotation matrix
rot = np.array([[-0.89433495, -0.44719485, -0.01348182],
       [-0.43426149,  0.87492975, -0.21427761],
       [-0.10761949,  0.18578133,  0.97667976]])
# creating new spiral with some noise
source = np.dot(target, rot)

# Before hyperalignment
fig,ax,data = hyp.plot([target, source], show=False, return_data=True)
ax.set_title('Before Procrustes')
plt.show()

# After hyperalignment
fig,ax,data = hyp.plot([hyp.tools.procrustes(source, target), target], ['-','--'], show=False, return_data=True)
ax.set_title('After Procrustes')
plt.show()
