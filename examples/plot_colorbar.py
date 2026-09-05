# -*- coding: utf-8 -*-
"""
=============================
Colorbars
=============================

The `colorbar` kwarg draws a colorbar reflecting whatever color mapping
is already in use. For a continuous `hue`, the colorbar is a
continuous gradient spanning the actual value range. For discrete groups
(categorical `hue`, `cluster`/`n_clusters`, or a plain list of datasets),
the colorbar is segmented into one block per group, labeled the same way
the legend would be.
"""

# Code source: Contextual Dynamics Laboratory
# License: MIT

import numpy as np

import hypertools as hyp

rng = np.random.default_rng(2)
data = rng.standard_normal((200, 3)).cumsum(axis=0)

# continuous hue: a colorbar spanning the actual value range
hue = np.linspace(0, 1, len(data))
hyp.plot(data, '-', hue=hue, colorbar=True)

###############################################################################
# Discrete groups (here, k-means clusters) get a segmented colorbar with
# one tick/block per group instead of a continuous gradient. A dict lets
# you set a label or move the colorbar to another side of the figure.

hyp.plot(data, '.', n_clusters=4,
         colorbar={'label': 'cluster', 'location': 'right'})
