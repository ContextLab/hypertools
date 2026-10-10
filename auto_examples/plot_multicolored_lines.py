# -*- coding: utf-8 -*-
"""
=============================
Multicolored lines
=============================

Passing continuous values (or a matrix with one row per observation) as
`hue` together with a line format string colors each trajectory
continuously along its length -- for example, coloring a trajectory by
time, by a behavioral variable, or by mixture proportions. Works on both
the matplotlib and plotly backends.
"""

# Code source: Contextual Dynamics Lab
# License: MIT

import numpy as np
import hypertools as hyp

data = np.cumsum(np.random.default_rng(42).standard_normal((300, 8)),
                 axis=0)

# color the trajectory by time
hyp.plot(data, hue=np.arange(len(data), dtype=float))

# color by any per-observation matrix (here: two smoothly varying weights)
weights = np.column_stack([np.linspace(0, 1, len(data)),
                           np.linspace(1, 0, len(data))])
hyp.plot(data, hue=weights)
