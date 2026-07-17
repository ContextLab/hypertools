# -*- coding: utf-8 -*-
"""
===========================================
Animated interactive plots (plotly backend)
===========================================

Animations work on the plotly backend too: `animate=True` reveals
trajectories through a sliding time window and `animate='spin'` rotates the
camera, each with interactive play/pause controls in notebooks. Animations
export via the `save_path` kwarg -- the file extension picks the format.
On the plotly backend, interactive `.html` exports are fast (no
rasterization, no extra dependencies). Rasterized exports (`.gif`, animated
`.png`, `.mp4`) are also supported, but they render every frame through
kaleido/Chromium at roughly a few seconds per frame -- keep `duration` and
`frame_rate` small for those, or use the matplotlib backend for long
rasterized animations.
"""

# Code source: Contextual Dynamics Lab
# License: MIT

import os
import tempfile

import numpy as np
import hypertools as hyp

data = np.cumsum(np.random.default_rng(42).standard_normal((200, 8)),
                 axis=0)

# interactive animation (play/pause controls in notebooks)
hyp.plot(data, animate=True, duration=30, backend='plotly')

# export an interactive animation: .html keeps the play/pause controls and
# writes in seconds. (Write it to a temp dir so no build artifacts land in
# the source tree; a short duration keeps the file small.)
save_path = os.path.join(tempfile.mkdtemp(), 'spin.html')
hyp.plot(data, animate='spin', duration=4, frame_rate=15, backend='plotly',
         save_path=save_path, show=False)
print(f'wrote {os.path.getsize(save_path)} bytes to spin.html')
