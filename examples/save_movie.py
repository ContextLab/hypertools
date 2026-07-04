# -*- coding: utf-8 -*-
"""
=============================
Saving an animation
=============================

To save an animation, simply add the `save_path` kwarg and specify the path
where you want to save the movie, including the extension.  NOTE: this
depends on having ffmpeg installed on your computer.
"""

# sphinx_gallery_thumbnail_path = '_static/thumbnails/sphx_glr_save_movie_thumb.gif'

# Code source: Andrew Heusser
# License: MIT

import hypertools as hyp
import numpy as np

data = hyp.load('weights', align='hyper')

group1 = np.mean(data[:17], 0)
group2 = np.mean(data[18:], 0)

import os, tempfile
save_path = os.path.join(tempfile.mkdtemp(), 'animation.mp4')
fig, ani = hyp.plot([group1, group2], animate=True, save_path=save_path)
