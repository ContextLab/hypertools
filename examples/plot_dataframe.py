# -*- coding: utf-8 -*-
"""
=============================
Plotting a Pandas Dataframe
=============================

Hypertools supports single-index Pandas Dataframes as input. In this example, we
plot the mushrooms dataset from the kaggle database.  This is a dataset of text
features describing different attributes of a mushroom. Dataframes that contain
columns with text are converted into binary feature vectors representing the
presence or absences of the feature (see the top-level `pandas.get_dummies`
function for more). Because the rows of this dataset have no meaningful order,
we plot them as points (the '.' format string) rather than as a connected line.
"""

# Code source: Andrew Heusser
# License: MIT

# import
import hypertools as hyp

# load example data
data = hyp.load('mushrooms')

print(data.head())

# plot (as points -- the rows are unordered samples, not a trajectory)
hyp.plot(data, '.')
