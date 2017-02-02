# -*- coding: utf-8 -*-
"""
=============================
Customizing figures
=============================

Since Hypertools is based on matplotlib, you can use matplotlib to change plot
attributes.
"""

# Code source: Andrew Heusser
# License: MIT

import hypertools as hyp
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt

data=sio.loadmat('sample_data/weights.mat')
w=[i for i in data['weights'][0][0:2]]

fig,ax,data = hyp.plot(w,'o', ndims=2, legend=['Group A', 'Group B'], show=False, return_data=True)

ax.set_title('This is an example title', fontsize=20)
ax.set_ylabel('PCA Dimension 2', fontsize=15)
ax.set_xlabel('PCA Dimension 1', fontsize=15)
legend_text = plt.gca().get_legend().get_texts()
plt.setp(legend_text, fontsize=15)

plt.show()
