# -*- coding: utf-8 -*-
"""
=============================
Formatting a mixed list of high dimensional data
=============================

The format_data function takes a (mixed) list of Numpy Arrays, Pandas
DataFrames, strings, or lists of strings and formats and returns them as a list
of Numpy Arrays. Text columns of a DataFrame are transformed into multiple binary
columns using the Pandas get_dummies function. Strings or lists of strings are
vectorized and then transformed with a text model. If a mixed list of numerical
and text data are passed, the data are aligned (using Hyperalignment by default)
to put them in the same space (and same dimensionality).
"""

# Code source: Andrew Heusser
# License: MIT

# import
import hypertools as hyp
import numpy as np
import pandas as pd

mat = np.random.rand(1000,100)
df = pd.DataFrame(np.random.rand(1000,100))
text = ['here is some test text', 'and a little more', 'and more']

# return formatted data
formatted_data = hyp.tools.format_data([mat, df, text])
