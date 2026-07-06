# -*- coding: utf-8 -*-
"""
=============================
Density shading
=============================

The `density` kwarg overlays a subtle KDE (kernel density estimate) "glow"
behind the data: a 2D alpha-ramped heatmap, or a 3D volumetric cloud,
showing where each dataset's points are concentrated (GH #108, #191).
Density shading is OFF by default (`density=None`) -- pass `density=True`
for the defaults, or a dict to override `alpha`/`levels`/`grid`/`per_group`.
"""

# Code source: Contextual Dynamics Laboratory
# License: MIT

import numpy as np

import hypertools as hyp

rng = np.random.default_rng(1)

# two 2D clusters
cluster_a_2d = rng.standard_normal((200, 2)) * 0.6
cluster_b_2d = rng.standard_normal((200, 2)) * 0.6 + [4, 0]

hyp.plot([cluster_a_2d, cluster_b_2d], '.', density=True)

###############################################################################
# The same kwarg works in 3D, where it renders as a translucent volumetric
# cloud instead of a 2D heatmap.

cluster_a_3d = rng.standard_normal((200, 3)) * 0.6
cluster_b_3d = rng.standard_normal((200, 3)) * 0.6 + [4, 0, 0]

hyp.plot([cluster_a_3d, cluster_b_3d], '.', density=True)

###############################################################################
# A dict lets you turn up the glow (higher `alpha`, more contour `levels`)
# for a more prominent effect -- but by default the density stays subtle
# so the actual data points remain the dominant visual element.

hyp.plot([cluster_a_2d, cluster_b_2d], '.',
         density={'alpha': 0.4, 'levels': 5})
