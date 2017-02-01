# -*- coding: utf-8 -*-
"""
=============================
Choosing the thumbnail figure
=============================

An example to demonstrate how to choose which figure is displayed as the
thumbnail if the example generates more than one figure. This is done by
specifying the keyword-value pair ``sphinx_gallery_thumbnail_number = 2`` as a
comment somewhere below the docstring in the example file.
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
