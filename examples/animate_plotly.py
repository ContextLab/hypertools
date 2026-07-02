# -*- coding: utf-8 -*-
"""
=============================
Animated interactive plots (plotly backend)
=============================

Animations work on the plotly backend too: `animate=True` reveals
trajectories through a sliding time window and `animate='spin'` rotates the
camera, each with interactive play/pause controls in notebooks. Animations
on either backend export to `.gif`, animated `.png`, or `.mp4` -- the file
extension picks the format.
"""

# Code source: Contextual Dynamics Lab
# License: MIT

import numpy as np
import hypertools as hyp

data = np.cumsum(np.random.default_rng(42).standard_normal((200, 8)),
                 axis=0)

# interactive animation (play/pause controls in notebooks)
hyp.plot(data, animate=True, duration=5, backend='plotly')

# export an animation: extension picks the format (.gif / .png / .mp4)
hyp.plot(data, animate='spin', duration=5, backend='plotly',
         save_path='spin.gif', show=False)
