# -*- coding: utf-8 -*-
"""
============================================
Interactive plotting with the plotly backend
============================================

HyperTools 1.0 can render any plot with plotly instead of matplotlib by
passing `backend='plotly'` -- handy for rotating and zooming 3D plots
interactively. With the default `backend='auto'`, hypertools automatically
uses plotly on Google Colab and Kaggle notebooks (where plotly is
preinstalled and interactivity works best) and matplotlib everywhere else,
so existing workflows are unchanged. Both backends produce the same styling:
colors, line/marker sizes, format strings, and the signature cube frame.
"""

# Code source: Contextual Dynamics Lab
# License: MIT

import numpy as np
import hypertools as hyp

data = np.cumsum(np.random.default_rng(42).standard_normal((200, 10)),
                 axis=0)

# classic matplotlib render (the default outside Colab/Kaggle)
hyp.plot(data, backend='matplotlib')

# identical call, interactive plotly render (in the docs gallery this is
# captured as an image; in a notebook it is fully interactive)
hyp.plot(data, backend='plotly')
