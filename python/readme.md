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

        palette: A matplotlib or seaborn color palette

        color: A list of colors for each line to be plotted. Can be named colors, RGB values (e.g. (.3, .4, .1)) or hex codes. If defined, overrides palette. See http://matplotlib.org/examples/color/named_colors.html for list of named colors. Note: must be the same length as X.

        linestyle: a list of line styles

        marker: a list of marker types

        See matplotlib API for more styling options

EXAMPLE USES:

Plot with default color palette: `coords.plot_coords(w)`

Change color palette: `coords.plot_coords(w,palette='Reds')`

Specify colors using unlabeled list of format strings: `coords.plot_coords([w[0],w[1]],['r:','b--'])`

Plot data as points: `coords.plot_coords([w[0],w[1]],'o')`

Specify colors using keyword list of colors (color codes, rgb values, hex codes or a mix): `coords.plot_coords([w[0],w[1],[w[2]],color=['r', (.5,.2,.9), '#101010'])`

Specify linestyles using keyword list: `coords.plot_coords([w[0],w[1],[w[2]],linestyle=[':','--','-'])`

Specify markers using keyword list: `coords.plot_coords([w[0],w[1],[w[2]],marker=['o','*','^'])`

Specify markers with format string and colors with keyword argument: `coords.plot_coords([w[0],w[1],[w[2]], 'o', color=['r','g','b'])``
