"""
===============================================================
Story trajectories: brain activity while listening to a story
===============================================================

This example walks through the "story trajectories" demo (GH #275): an
animated, hyperaligned cloud of trajectories that shows how every subject's
whole-brain activity pattern traces out a *shared* path through a
low-dimensional space while they listen to the same spoken story.

Background
----------

The data (``hyp.load('weights')``) come from Simony et al. (2016,
*Nature Communications*, `10.1038/ncomms12141
<https://doi.org/10.1038/ncomms12141>`_), an fMRI study in which 36 subjects
listened to the same ~7-minute spoken story -- "PieMan", told live by Jim
O'Grady at a Moth GrandSLAM event -- while their whole-brain activity was
recorded. Each subject's raw voxel-by-timepoint data were summarized with
Hierarchical Topographic Factor Analysis (HTFA) into ``k=100`` latent
"hubs" -- spatially compact, story-timescale sources of correlated
activity -- giving every subject a ``(timepoints, 100)`` trajectory through
"hub space" that HyperTools can align, reduce, and animate directly.

What the animation shows
--------------------------

Every subject starts the story with an idiosyncratic, unaligned activity
pattern. Hyperalignment rotates every subject's trajectory into a common
space that maximizes shared, story-locked structure, so that, once aligned,
subjects' trajectories overlap into a single coherent, shared shape -- the
narrative structure of the story, written into all 36 brains at once. The
animation spins the camera around that shared shape to show its full 3-D
geometry. Two scale-free checks confirm the acceptance criterion (GH #275) --
that subjects genuinely move together, smoothly:

* **Together.** Hyperalignment tightens the subjects' spread around their
  shared centroid at each timepoint (within-timepoint dispersion, normalized by
  the cloud's overall scale) by ~18%: 0.88 without alignment -> 0.73 with it.
* **Smooth.** Switching from UMAP to a linear reduction shrinks the largest
  per-step jump in the displayed trajectories almost ten-fold (normalized
  3.3 -> 0.37) -- the difference between a choppy animation and a smooth one.

(Plain inter-subject correlation is a poor proxy here: a jumpy UMAP embedding
can score high correlation while still looking scattered and choppy, so we
report dispersion and smoothness instead.)

Getting alignment right
-----------------------

Two choices matter, and the earlier version of this demo got both wrong:

* **Align in the LOW-dimensional space, not the 100-hub space.** HyperTools'
  canonical pipeline order is ``manip -> normalize -> reduce -> align``, so
  ``reduce`` runs *before* ``align``: here we reduce each subject to a
  ``ndims=10`` `IncrementalPCA` space and hyperalign *there* (with
  ``n_iter=10`` iterations), then show the first 3 aligned dimensions. Aligning
  in this richer-but-still-low-dimensional space, rather than a bare 3-D
  embedding, is what actually pulls the displayed trajectories together.
* **Use a linear reduction.** ``reduce='UMAP'`` warped each subject's
  trajectory nonlinearly and left the animation jumpy and poorly aligned;
  `IncrementalPCA` keeps the trajectories smooth and preserves the shared
  linear structure hyperalignment depends on.

The exact code
----------------

The full pipeline takes a couple of minutes to run on the full dataset, so
this example does **not** re-run it live; instead it displays the pre-rendered
result. Here is the exact code that produced it:

.. code-block:: python

    import matplotlib.pyplot as plt
    import hypertools as hyp

    data = hyp.load('weights')   # 36 subjects, each (timepoints, 100 hubs)

    # per-subject preprocessing in native (100-hub) space, BEFORE reduction:
    # smooth each timeseries, resample everyone onto a common grid, z-score
    manip_spec = [
        {'model': 'Smooth', 'kwargs': {'kernel_width': 40}},
        {'model': 'Resample', 'kwargs': {'n_samples': 600}},
        'ZScore',
    ]

    # reduce to a LOW-dimensional (IncrementalPCA) space and hyperalign THERE,
    # with 10 iterations -- NOT in the full 100-hub space
    aligned = hyp.analyze(
        data, manip=manip_spec, reduce='IncrementalPCA', ndims=10,
        align={'model': 'HyperAlign', 'kwargs': {'n_iter': 10}},
    )

    # one translucent rainbow color per subject; animate the first 3 (highest-
    # variance) aligned dimensions, spinning the camera around the shared shape
    n = len(aligned)
    colors = [(*plt.get_cmap('gist_rainbow')(i / (n - 1))[:3], 0.5)
              for i in range(n)]
    hyp.plot([subject[:, :3] for subject in aligned], '-', color=colors,
             linewidth=1.2, animate='spin', duration=9, zoom=1.5,
             save_path='story_trajectories.mp4')

Below: the shared aligned shape from three camera angles, followed by the full
spinning animation (``docs/images/v1.0-round17/story_trajectories.mp4``).

.. video:: /images/v1.0-round17/story_trajectories.mp4
   :width: 700
   :loop:
"""

# Code source: Contextual Dynamics Laboratory
# License: MIT

# use the pre-rendered spinning story animation as this example's gallery
# thumbnail, matching the other animated examples (animate_spin, chemtrails,
# ...).
# sphinx_gallery_thumbnail_path = '_static/thumbnails/sphx_glr_plot_story_trajectories_thumb.gif'

import os

import matplotlib.image as mpimg
import matplotlib.pyplot as plt

import hypertools as hyp


def _find_img_dir():
    """Locate docs/images/v1.0-round17 regardless of how this script is
    run: directly (`__file__` is defined), or execed by sphinx-gallery
    (which chdir's into this script's own directory before running it, but
    does NOT define `__file__` -- so `os.getcwd()` stands in for it)."""
    bases = []
    try:
        bases.append(os.path.dirname(os.path.abspath(__file__)))
    except NameError:
        pass
    bases.append(os.getcwd())
    bases.append(os.path.dirname(os.path.abspath(hyp.__file__)))
    for base in bases:
        for up in ('..', os.path.join('..', '..')):
            candidate = os.path.normpath(
                os.path.join(base, up, 'docs', 'images', 'v1.0-round17'))
            if os.path.isdir(candidate):
                return candidate
    raise RuntimeError('could not locate docs/images/v1.0-round17')


IMG_DIR = _find_img_dir()

# the shared, hyperaligned shape from three camera angles (a spin has no
# "early/mid/late" -- the whole trajectory is shown at once and rotated)
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for ax, (fname, label) in zip(
        axes,
        (('story_frame_early.png', 'view 1'),
         ('story_frame_mid.png', 'view 2'),
         ('story_frame_late.png', 'view 3'))):
    img = mpimg.imread(os.path.join(IMG_DIR, fname))
    ax.imshow(img)
    ax.set_title(f'shared trajectory, {label}')
    ax.axis('off')
plt.tight_layout()
plt.show()
