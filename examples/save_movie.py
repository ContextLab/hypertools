# -*- coding: utf-8 -*-
"""
=============================
Saving an animation
=============================

To save an animation, simply add the `save_path` kwarg and specify the path
where you want to save the movie, including the extension.  NOTE: saving to
`.mp4` (or `.mov`/`.avi`) uses matplotlib's ffmpeg writer, so ffmpeg must be
installed and on your PATH for those formats; `.gif` and animated `.png`
exports are written with Pillow and need no external tools.
"""

# sphinx_gallery_thumbnail_path = '_static/thumbnails/sphx_glr_save_movie_thumb.gif'

# Code source: Andrew Heusser
# License: MIT

import hypertools as hyp
import numpy as np

data = hyp.load('weights', align='hyper')

# average the 36 subjects into two equal groups of 18
group1 = np.mean(data[:18], 0)
group2 = np.mean(data[18:], 0)

import os  # noqa: E402 (sphinx-gallery narrative section, not top-of-file)
import tempfile  # noqa: E402 (same)
save_path = os.path.join(tempfile.mkdtemp(), 'animation.mp4')
fig, ani = hyp.plot([group1, group2], animate=True, save_path=save_path)
