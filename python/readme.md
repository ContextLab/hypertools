To install, navigate to this folder in Terminal and type:

pip install -e .

(this assumes you have pip installed on your system: https://pip.pypa.io/en/stable/installing/)

INPUTS:

        X: a T by D matrix of observations.  T is the number of coordinates
        and D is the dimensionality of each observation.  NaNs are
        treated as missing observations.

ARGUMENTS:

        Format strings can be passed as a string, or tuple/list of length x.

        See matplotlib API for more styling options

KEYWORD ARGUMENTS:

        palette (string): A matplotlib or seaborn color palette

        color (list): A list of colors for each line to be plotted. Can be named colors, RGB values (e.g. (.3, .4, .1)) or hex codes. If defined, overrides palette. See http://matplotlib.org/examples/color/named_colors.html for list of named colors. Note: must be the same length as X.

        point_colors (list of str, floats or ints): A list of colors for each point. Must be dimensionality of data (X). If the data type is numerical, the values will be mapped to rgb values in the specified palette.  If the data type is strings, the points will be labeled categorically.

        linestyle (list): a list of line styles

        marker (list): a list of marker types

        See matplotlib API for more styling options

        labels (list): A list of labels for each point. Must be dimensionality of data (X). If no label is wanted for a particular point, input `None`

        explore (bool): (experimental feature) Displays user defined labels or PCA coordinates on hover. When a point is clicked, the label will remain on the plot (WIP). To use, set `explore=True`.


EXAMPLE USES:

Plot with default color palette: `coords.plot_coords(w)`

Change color palette: `coords.plot_coords(w,palette='Reds')`

Specify colors using unlabeled list of format strings: `coords.plot_coords([w[0],w[1]],['r:','b--'])`

Plot data as points: `coords.plot_coords([w[0],w[1]],'o')`

Specify colors using keyword list of colors (color codes, rgb values, hex codes or a mix): `coords.plot_coords([w[0],w[1],[w[2]],color=['r', (.5,.2,.9), '#101010'])`

Specify linestyles using keyword list: `coords.plot_coords([w[0],w[1],[w[2]],linestyle=[':','--','-'])`

Specify markers using keyword list: `coords.plot_coords([w[0],w[1],[w[2]],marker=['o','*','^'])`

Specify markers with format string and colors with keyword argument: `coords.plot_coords([w[0],w[1],[w[2]], 'o', color=['r','g','b'])``

Specify labels:
```
# Label first point of each list
labels=[]
for idx,i in enumerate(w):
    tmp=[]
    for iidx,ii in enumerate(i):
        if iidx==0:
            tmp.append('Point ' + str(idx))
        else:
            tmp.append(None)
    labels.append(tmp)

coords.plot_coords(w, 'o', labels=labels)
```

Specify point_colors:
```
# Label first point of each list
point_colors=[]
for idx,i in enumerate(w):
    tmp=[]
    for iidx,ii in enumerate(i):
            tmp.append(np.random.rand())
    point_colors.append(tmp)

coords.plot_coords(w, 'o', point_colors=point_colors)
```

Turn on explore mode (experimental): `coords.plot_coords(w, 'o', explore=True)`
