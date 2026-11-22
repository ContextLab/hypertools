# -*- coding: utf-8 -*-
"""
=====================================
Plotting State of the Union Addresses
=====================================

`hyp.load('sotus')` returns the full text of the 29 State of the Union
addresses delivered between 1989 and 2017, grouped by president rather than
sorted by date, so this example first puts them in date order. Passing the
raw speech texts straight to `hyp.plot` runs hypertools' default text
pipeline: each address is converted to a vector of word counts, modeled with
a 50-topic Latent Dirichlet Allocation model fit to a large sample of
wikipedia pages, and reduced to 3 dimensions. Because the addresses are
plotted in chronological order, the connected line traces a "text
trajectory" through semantic space: addresses that emphasize similar themes
land near one another, and the trajectory shows how the topics presidents
discuss have drifted over three decades.
"""

# Code source: Andrew Heusser
# License: MIT

# load hypertools
import hypertools as hyp

# load the State of the Union addresses: 29 speeches (1989-2017), listed by
# president -- G. H. W. Bush (1989-92), G. W. Bush (2001-08), Clinton
# (1993-2000), Obama (2009-16), Trump (2017) -- then sorted by year
speeches = hyp.load('sotus')
years = [*range(1989, 1993), *range(2001, 2009), *range(1993, 2001), *range(2009, 2018)]
speeches = [speech for _year, speech in sorted(zip(years, speeches))]
print(f'{len(speeches)} State of the Union addresses loaded')

# plot the trajectory through semantic space
hyp.plot(speeches)
