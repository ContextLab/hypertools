"""
=============================================================
Story trajectories: brain activity while listening to a story
=============================================================

An animated cloud of hyperaligned trajectories showing how 36 subjects'
whole-brain activity traces out a *shared* path through a low-dimensional
space while they listen to the same spoken story.

The data (``hyp.load('weights')``) come from Simony et al. (2016, *Nature
Communications*, `10.1038/ncomms12141 <https://doi.org/10.1038/ncomms12141>`_),
an fMRI study in which 36 subjects listened to the same ~7-minute story
("PieMan", told by Jim O'Grady at a Moth GrandSLAM) while their brain
activity was recorded. Each subject's voxel data were summarized with
Hierarchical Topographic Factor Analysis into ``k=100`` latent "hubs",
giving every subject a ``(300 timepoints, 100 hubs)`` trajectory that
HyperTools can manipulate, align, reduce, and animate directly.

Every subject starts the story with an idiosyncratic activity pattern.
Hyperalignment rotates each subject's trajectory into a common space that
maximizes the shared, story-locked structure, so that once aligned the
subjects move *together* through the space as a single cloud as the story
unfolds. The ``animate='window'`` style slides a short opaque trail along
each aligned trajectory, so you watch all 36 subjects travel the shared
path in lock-step.

Two choices matter for getting a coherent cloud:

* **Align in the 100-hub space, then reduce to 3-D** (not the other way
  around). Hyperalignment finds an orthogonal transform per subject, and
  with only 3 dimensions to work in it can barely align anything; the
  100-hub space gives it room. The printed dispersion (the mean distance of
  the 36 subjects to their centroid at each timepoint, relative to the
  overall cloud scale; lower means the subjects move together more) drops
  substantially after alignment.
* **Use a linear reducer** (``IncrementalPCA``). A nonlinear reducer such
  as UMAP warps each trajectory differently and makes the animation jumpy.

The whole pipeline (load, smooth, z-score, hyperalign, reduce, animate)
runs on the full dataset in about ten seconds, so the figure below is
computed live.
"""

# Code source: Contextual Dynamics Laboratory
# License: MIT

# sphinx_gallery_thumbnail_path = '_static/thumbnails/sphx_glr_plot_story_trajectories_thumb.gif'

import numpy as np

import hypertools as hyp

data = hyp.load('weights')   # 36 subjects, each (300 timepoints, 100 hubs)

# per-subject preprocessing in the native 100-hub space: a 41-timepoint
# boxcar smooth (Smooth requires an odd kernel width) and z-scoring
manip_spec = [{'model': 'Smooth', 'kwargs': {'kernel_width': 41}}, 'ZScore']
manip_data = hyp.manip(data, model=manip_spec)

# ALIGN in the 100-hub space (10 hyperalignment iterations), THEN reduce
aligned = hyp.align(manip_data,
                    model='HyperAlign', n_iter=10)


def dispersion(trajectories):
    """Mean distance of the subjects to their shared centroid at each
    timepoint, averaged over timepoints and divided by the overall cloud
    scale (so it is comparable before and after alignment)."""
    stack = np.stack([np.asarray(t) for t in trajectories])   # (subj, t, d)
    centroid = stack.mean(axis=0, keepdims=True)
    spread = np.linalg.norm(stack - centroid, axis=2).mean()
    scale = np.linalg.norm(stack - stack.mean(axis=(0, 1)), axis=2).mean()
    return spread / scale


print(f'dispersion before alignment: {dispersion(manip_data):.2f}')
print(f'dispersion after alignment:  {dispersion(aligned):.2f}')

# one bold color per subject from the 'husl' palette; a short sliding
# 'window' trail (focused=1.5 s) lets you watch all 36 subjects move
# together through the story. Because the subjects are tightly aligned, the
# overlapping near-opaque ribbons read as ONE coherent shape.
fig, ani = hyp.plot(aligned, '-', palette='husl', alpha=0.85, linewidth=1.6,
                    reduce='IncrementalPCA', ndims=3, animate='window',
                    focused=1.5, zoom=1.5, duration=9)
