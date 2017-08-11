#!/usr/bin/env python
import warnings
import matplotlib as mpl

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    mpl.use('TkAgg')
