# -*- coding: utf-8 -*-
"""
==============
Animated plots
==============

Timeseries can be animated by passing ``animate=True`` to `hyp.plot`: each
trajectory is drawn progressively while the camera rotates around the
scene. The data here are two group-average brain-activity trajectories
(``hyp.load('weights_avg')``). The first animation uses the default (PCA)
reduction; the second reduces the same data with multidimensional scaling
instead, which changes the shape of the path the trajectories trace out.
"""

# sphinx_gallery_thumbnail_path = '_static/thumbnails/sphx_glr_animate_thumb.gif'

# Code source: Andrew Heusser
# License: MIT

import hypertools as hyp

# load example data: two trajectories, each (timepoints, features)
data = hyp.load('weights_avg')

# %%
# Animate with the default reducer (PCA); `legend` names the two groups.
fig, ani = hyp.plot(data, animate=True, legend=['first', 'second'])

# %%
# Any reducer can be combined with animation -- here multidimensional
# scaling (MDS) replaces PCA.
fig, ani_mds = hyp.plot(data, animate=True, reduce='MDS')
