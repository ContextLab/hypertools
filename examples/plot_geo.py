# -*- coding: utf-8 -*-
"""
=============================
Working with plot outputs (figures & fitted models)
=============================

`hyp.plot` returns a plain matplotlib (or plotly) Figure -- there is no
special container object to learn. Anything you can do with a Figure
(``fig.savefig(...)``, grabbing ``fig.axes[0]`` to tweak the plot,
embedding it in a larger layout, etc.) just works.

If you also want access to the analyzed data and the fitted
reduce/align/cluster models, pass ``return_model=True``. Instead of the
bare figure, `hyp.plot` then returns a dict bundle:
``{'fig': ..., 'xform_data': ..., 'animation': ..., 'models': ...}``, where
``xform_data`` is the normalized/reduced/aligned data that was actually
plotted, ``animation`` is the ``matplotlib.animation.Animation`` handle when
``animate=True`` (``None`` otherwise, and for plotly figures), and ``models``
records the reduce/align/cluster specs used to produce it.

Note that `hyp.load` returns raw data directly (e.g. a list of arrays) --
there is nothing further to unpack.
"""

# Code source: Contextual Dynamics Lab
# License: MIT

# import
import os
import tempfile
import hypertools as hyp

# load some data -- a list of arrays, ready to plot as-is
data = hyp.load('spiral')

# plot: the return value is just a matplotlib Figure
fig = hyp.plot(data, ndims=3)

# treat it like any other Figure
png_path = os.path.join(tempfile.mkdtemp(), 'spiral.png')
fig.savefig(png_path)
ax = fig.axes[0]
print(f"axes type: {type(ax).__name__}")

# ask for the fitted models and the analyzed data alongside the figure
out = hyp.plot(data, ndims=3, reduce='PCA', return_model=True)

fig2 = out['fig']
xform_data = out['xform_data']
models = out['models']

print(f"number of arrays returned: {len(xform_data)}")
print(f"reduced shape (first array): {xform_data[0].shape}")
print(f"reduce model spec: {models['reduce']}")
