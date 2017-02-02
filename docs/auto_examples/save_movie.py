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

data=sio.loadmat('example_data/weights.mat')
w=data['weights'][0]
w = [i for i in w]
aligned_w = hyp.tools.align(w)

w1 = np.mean(aligned_w[:17],0)
w2 = np.mean(aligned_w[18:],0)

hyp.plot([w1,w2], animate=True, zoom=2.5, save_path='animation.mp4')
