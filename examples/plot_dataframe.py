# -*- coding: utf-8 -*-
"""
=============================
Plotting a Pandas Dataframe
=============================

Hypertools supports single-index Pandas Dataframes as input. In this example, we
plot the mushrooms dataset from the kaggle database.  This is a dataset of text
features describing different attributes of a mushroom. Dataframes that contain
columns with text are converted into binary feature vectors representing the
presence or absences of the feature (see Pandas.Dataframe.get_dummies for more).
"""

# Code source: Andrew Heusser
# License: MIT

# import
import hypertools as hyp

# load example data
data, labels = hyp.load('mushrooms')

# pop off the class (poisonousness)
hue = data.pop('class')

# plot
hyp.plot(data, '.', hue=hue)
