# -*- coding: utf-8 -*-
"""
=============================
Normalizing your features
=============================

Often times its useful to normalize (z-score) you features before plotting, so
that they are on the same scale.  Otherwise, some features will be weighted more
heavily than others when doing PCA, and that may or may not be what you want.
The `normalize` kwarg can be passed to the plot function.  If `normalize` is
set to 'across', the zscore will be computed for the column across all of the
lists passed.  Conversely, if `normalize` is set to 'within', the z-score will
be computed separately for each column in each list.  Finally, if `normalize` is
set to 'row', each row of the matrix will be zscored.  Alternative, you can use
the normalize function found in tools (see the third example).
"""

# Code source: Andrew Heusser
# License: MIT

import hypertools as hyp
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt

cluster1 = np.random.multivariate_normal(np.zeros(3), np.eye(3), size=100)
cluster2 = np.random.multivariate_normal(np.zeros(3)+10, np.eye(3), size=100)

data = [cluster1,cluster2]

fig,ax,data = hyp.plot(data, '.', normalize='across')
ax.set_title('z-score columns across all lists')
plt.show()

fig,ax,data = hyp.plot(data, '.', normalize='within')
ax.set_title('z-score columns within each list')
plt.show()

normalized_row = hyp.tools.normalize(data, normalize='row')
fig,ax,data = hyp.plot(normalized_row, '.')
ax.set_title('z-score each row')
plt.show()
