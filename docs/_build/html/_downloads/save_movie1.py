# -*- coding: utf-8 -*-
"""
=============================
Saving an animation
=============================

To save an animation, simply add the `save_path` kwarg and specify the path
where you want to save the movie, including the extension.  NOTE: this
depends on having ffmpeg installed on your computer.
"""

# Code source: Andrew Heusser
# License: MIT

import hypertools as hyp
import scipy.io as sio
import numpy as np

data = hyp.tools.load('weights')
aligned_data = hyp.tools.align(data)

group1 = np.mean(aligned_data[:17], 0)
group2 = np.mean(aligned_data[18:], 0)

hyp.plot([group1, group2], animate=True, zoom=2.5, save_path='animation.mp4')
