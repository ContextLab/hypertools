# -*- coding: utf-8 -*-
"""
===================
Saving an animation
===================

To save an animation, add the ``save_path`` kwarg with the path (and file
extension) you want. Saving to ``.mp4`` (or ``.mov``/``.avi``) uses
matplotlib's ffmpeg writer, so ffmpeg must be installed and on your PATH
for those formats; ``.gif`` and animated ``.png`` exports are written with
Pillow and need no external tools. The data are the 36 hyperaligned
subjects of the ``weights`` dataset, averaged into two groups of 18, so the
movie shows two group-average trajectories through the same story.
"""

# sphinx_gallery_thumbnail_path = '_static/thumbnails/sphx_glr_save_movie_thumb.gif'

# Code source: Andrew Heusser
# License: MIT

import os
import tempfile

import numpy as np

import hypertools as hyp

# `hyp.load` can run alignment on its way in: align='HyperAlign' hyperaligns
# the 36 subjects into a shared space as they are loaded (equivalent to
# calling hyp.align(hyp.load('weights'), model='HyperAlign') yourself), so
# that averaging across subjects below is meaningful
data = hyp.load('weights', align='HyperAlign')

# average the 36 aligned subjects into two equal groups of 18
group1 = np.mean(data[:18], 0)
group2 = np.mean(data[18:], 0)

# animate the two group trajectories and write the movie to disk
save_path = os.path.join(tempfile.mkdtemp(), 'animation.mp4')
fig, ani = hyp.plot([group1, group2], animate=True, save_path=save_path)
print(f'saved {os.path.getsize(save_path) // 1024} KB to {save_path}')
