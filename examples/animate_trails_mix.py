# -*- coding: utf-8 -*-
"""
===============================
Mixing trail styles per dataset
===============================

`chemtrails`, `precog`, and `bullettime` each accept a per-dataset list of
bools instead of a single bool, so different datasets in the same animation
can show different trail styles: a low-opacity trace of the past
(chemtrails), of the future (precog), or of the entire timeseries at once
(bullettime -- equivalent to chemtrails AND precog together).
"""

# Code source: Contextual Dynamics Laboratory
# License: MIT

import numpy as np

import hypertools as hyp

rng = np.random.default_rng(4)
n_time = 40
t = np.linspace(0, 4 * np.pi, n_time)

data_a = np.column_stack([np.cos(t), np.sin(t), t / 4])
data_b = np.column_stack([np.cos(t), -np.sin(t), t / 4]) + [3, 0, 0]
data_c = np.column_stack([np.sin(t), np.cos(t), t / 4]) + [-3, 0, 0]

# dataset 0: chemtrails (past trail) only
# dataset 1: precog (future trail) only
# dataset 2: bullettime (full trail, past + future) only
fig, ani = hyp.plot(
    [data_a, data_b, data_c],
    animate=True,
    chemtrails=[True, False, False],
    precog=[False, True, False],
    bullettime=[False, False, True],
)
