#!/usr/bin/env python
import warnings
import matplotlib as mpl

try:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mpl.use('TkAgg')
except:
    warnings.warn('Could not switch backend to TkAgg.  This may impact performance of the plotting functions.')
