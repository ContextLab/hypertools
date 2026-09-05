# -*- coding: utf-8 -*-
"""
=======================================
Trails: chemtrails and precognition
=======================================

Animated plots can show more than the current position along each
trajectory. ``chemtrails=True`` leaves a low-opacity trace of the path
*already travelled* behind the moving points; ``precog=True`` draws a
low-opacity trace of the path *still to come* ahead of them. Combining
both (or passing ``bullettime=True``) shows the entire timeseries at low
opacity with the current segment highlighted. The data are two
group-average brain-activity trajectories (``hyp.load('weights_avg')``).
See *Mixing trail styles per dataset* for choosing a different style for
each dataset in the same animation.
"""

# sphinx_gallery_thumbnail_path = '_static/thumbnails/sphx_glr_animate_trails_thumb.gif'

# Code source: Andrew Heusser
# License: MIT

import hypertools as hyp

# load example data
data = hyp.load('weights_avg')

# %%
# Chemtrails: a faint trace of the past trajectory follows the moving points.
fig, ani_past = hyp.plot(data, animate=True, chemtrails=True)

# %%
# Precognition: a faint trace of the future trajectory leads the moving
# points.
fig, ani_future = hyp.plot(data, animate=True, precog=True)
