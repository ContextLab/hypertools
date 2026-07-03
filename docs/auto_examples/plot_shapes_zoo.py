# -*- coding: utf-8 -*-
"""
=============================
A zoo of 3D shapes
=============================

HyperTools ships with a "shapes zoo" of classic 3D point clouds (they
download once and are then cached in `~/hypertools_data`).  This example
loads *every* shape in the zoo and displays each in its own panel,
plotted as small black dots (the ``,`` pixel marker) by passing
pre-created 3D axes to `hyp.plot` via the `ax` keyword.
"""

# Code source: Contextual Dynamics Laboratory
# License: MIT

import math

import matplotlib.pyplot as plt

import hypertools as hyp

# every point cloud in the shapes zoo
shapes = ['bunny', 'cube', 'dragon', 'sphere', 'teapot', 'vase', 'biplane']

ncols = 4
nrows = math.ceil(len(shapes) / ncols)
fig, axes = plt.subplots(nrows, ncols, figsize=(3 * ncols, 3 * nrows),
                         subplot_kw={'projection': '3d'})
axes = axes.ravel()

for shape, ax in zip(shapes, axes):
    # small black dots: ',' is matplotlib's single-pixel marker, ideal for
    # dense point clouds
    hyp.plot(hyp.load(shape), ',', color='k', ax=ax,
             title=shape.capitalize())

# hide any unused panels
for ax in axes[len(shapes):]:
    ax.set_visible(False)

plt.tight_layout()
plt.show()
