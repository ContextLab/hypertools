"""
===============================================================
Story trajectories: brain activity while listening to a story
===============================================================

This example walks through the "story trajectories" demo (GH #275): an
animated cloud of hyperaligned trajectories showing how all 36 subjects'
whole-brain activity traces out a *shared* path through a low-dimensional
space while they listen to the same spoken story.

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
subjects' trajectories move *together* through the space as a single coherent
cloud as the story unfolds. The ``animate='window'`` style slides a short
opaque trail along each aligned trajectory, so you literally watch all 36
subjects travel the shared path in lock-step.

A scale-free within-timepoint dispersion (the mean distance of the 36
subjects to their shared centroid at each timepoint, averaged over
timepoints and divided by the overall cloud scale; lower = move together
more) drops from ~0.71 to ~0.52 once the subjects are hyperaligned the
RIGHT way -- see below.

Getting alignment right
-----------------------

The critical choice is *where* to hyperalign:

* **Align in the 100-hub feature space, THEN reduce to 3-D -- not the other
  way around.** Hyperalignment needs room: it rotates each subject's
  trajectory (an orthogonal transform) to match a shared response, and with
  only 3 (or even 10) dimensions to work in it can barely align anything,
  leaving the subjects a poorly-aligned tangle. Aligning in the full 100-hub
  space and reducing to 3-D *afterwards* is what pulls the displayed
  trajectories together (dispersion ~0.88 unaligned, ~0.84 when reduced to
  3-D then aligned, ~0.71 when reduced to 10-D then aligned -> ~0.52 when
  aligned in hub space then reduced). Note the hub space *is* the "low-dimensional"
  space to align in -- it is already a 100-dim summary of hundreds of
  thousands of voxels; the mistake is over-reducing further before aligning.
* **Use a linear reduction (IncrementalPCA).** ``reduce='UMAP'`` warped each
  subject's trajectory nonlinearly and left the animation jumpy; a linear
  reduction keeps the aligned trajectories smooth.

The exact code
----------------

The full pipeline runs in well under a minute on the full dataset (roughly
ten seconds on a modern laptop, including saving the movie), but this
example does **not** re-run it live; it displays the pre-rendered result so
the gallery stays fast and deterministic. Here is the exact code that
produced it:

.. code-block:: python

    import seaborn as sns
    import hypertools as hyp

    data = hyp.load('weights')   # 36 subjects, each (timepoints, 100 hubs)

    # per-subject preprocessing in native (100-hub) space. NOTE: Smooth
    # requires an ODD kernel width (even widths are bumped up by 1 with a
    # warning), so 41 is used here
    manip_spec = [
        {'model': 'Smooth', 'kwargs': {'kernel_width': 41}},
        {'model': 'Resample', 'kwargs': {'n_samples': 300}},
        'ZScore',
    ]
    manip_data = hyp.manip(data, model=manip_spec)

    # ALIGN in the 100-hub space (n_iter=10), THEN reduce to 3-D -- reducing
    # before aligning would starve hyperalignment and leave a tangle
    aligned = hyp.align(manip_data,
                        align={'model': 'HyperAlign', 'kwargs': {'n_iter': 10}})

    # one bold color per subject from seaborn's 'husl' palette (the classic
    # HyperTools palette: evenly-spaced but tempered hues). A short sliding
    # 'window' trail (focused=1.5 s) lets you watch all 36 subjects move
    # together through the story. The lines are near-opaque and bold -- because
    # the subjects are tightly aligned the overlapping ribbons read as ONE
    # coherent shape, so opaque lines stay legible (translucent ones just blur
    # into haze).
    n = len(aligned)
    colors = [(*c, 0.85) for c in sns.color_palette('husl', n)]
    hyp.plot(aligned, '-', color=colors, linewidth=1.6, reduce='IncrementalPCA',
             ndims=3, animate='window', focused=1.5, zoom=1.5, duration=9,
             save_path='story_trajectories.mp4')

Below: three moments as the story unfolds (the window sliding from the start
to the end), followed by the full animation
(``docs/images/v1.0-round17/story_trajectories.mp4``).

.. video:: /images/v1.0-round17/story_trajectories.mp4
   :width: 700
   :loop:
"""

# Code source: Contextual Dynamics Laboratory
# License: MIT

# use the pre-rendered story animation as this example's gallery thumbnail,
# matching the other animated examples (animate_spin, chemtrails, ...).
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

# three moments as the story unfolds -- the sliding window early, midway, and
# late -- from the pre-rendered animation. (constrained layout keeps the
# subplot titles from being clipped at the top of the figure.)
fig, axes = plt.subplots(1, 3, figsize=(15, 5), layout='constrained')
for ax, label in zip(axes, ('early', 'mid', 'late')):
    img = mpimg.imread(os.path.join(IMG_DIR, f'story_frame_{label}.png'))
    ax.imshow(img)
    ax.set_title(f'{label} in the story')
    ax.axis('off')
