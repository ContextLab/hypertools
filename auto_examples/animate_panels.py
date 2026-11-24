# -*- coding: utf-8 -*-
"""
=====================================
Animating a grid of panels together
=====================================

``panels=`` draws each dataset in a panel of its own, and with ``animate=``
the whole grid becomes one animation: every panel runs on the same clock, and
one object is returned (and saved) for the figure. ``caption=`` adds a line of
text under the grid that is updated on every frame -- here a running
timepoint counter. The data are three participants' brain-activity
trajectories (``hyp.load('weights_sample')``), 300 timepoints each, reduced
together so the panels share one set of axes.
"""

# Code source: Contextual Dynamics Laboratory
# License: MIT

import hypertools as hyp

# load example data: a list of three (300 timepoints x 100 features) arrays
data = hyp.load('weights_sample')


# the caption is rebuilt for every frame from that frame's context; at 30
# frames per second, a 10-second clip draws one frame per timepoint
def timepoint(ctx):
    return f'timepoint {ctx.frame + 1} of {ctx.n_frames}'


# backend='matplotlib': an animated panel grid is a matplotlib animation
# (on Colab the default backend would be plotly)
fig, ani = hyp.plot(data, panels=(1, 3), animate=True, duration=10,
                    title=['Participant 1', 'Participant 2', 'Participant 3'],
                    caption=timepoint, backend='matplotlib')
