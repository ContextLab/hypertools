# -*- coding: utf-8 -*-
"""
===================================
Nested lists and multilevel styling
===================================

hypertools 1.0 accepts arbitrarily nested lists of datasets. Every dataset
under the same outermost group shares that group's color, and each
additional nesting level renders with thinner, fainter lines -- a
summary-to-detail visual hierarchy. For example, `[[a, b], [c]]` colors
`a` and `b` alike (group 1) and `c` differently (group 2).
"""

# Code source: Contextual Dynamics Lab
# License: MIT

import numpy as np
import hypertools as hyp


def walk(seed):
    return np.cumsum(np.random.default_rng(seed).standard_normal((150, 6)),
                     axis=0)


# two groups: [a, b] vs [c] -- a and b share a color
hyp.plot([[walk(0), walk(1)], [walk(2)]])

# mixed depth: deeper leaves render thinner and fainter
hyp.plot([[walk(0), [walk(1)]], walk(2)])
