# -*- coding: utf-8 -*-
"""
=============================
Choosing the thumbnail figure
=============================

An example to demonstrate how to choose which figure is displayed as the
thumbnail if the example generates more than one figure. This is done by
specifying the keyword-value pair ``sphinx_gallery_thumbnail_number = 2`` as a
comment somewhere below the docstring in the example file.
"""

# Code source: Andrew Heusser
# License: MIT

import hypertools as hyp
import pandas as pd

data=pd.read_csv('sample_data/mushrooms.csv')

hyp.plot(data,'o',n_clusters=10)

# OR
# cluster_labels = hyp.tools.cluster(data, n_clusters=10)
# hyp.plot(data,'o',group=cluster_labels)
