To install, navigate to this folder in Terminal and type:

pip install -e .

(this assumes you have pip installed on your system: https://pip.pypa.io/en/stable/installing/)

INPUTS:
        X: a T by D matrix of observations.  T is the number of coordinates
        and D is the dimensionality of each observation.  NaNs are
        treated as missing observations.

KEYWORD ARGUMENTS:

        palette: A matplotlib or seaborn color palette

          example: `palette="muted"`

        color: A list of colors for each line to be plotted. Can be named colors, RGB values (e.g. (.3, .4, .1)) or hex codes. If defined, overrides palette. See http://matplotlib.org/examples/color/named_colors.html for list of named colors. Note: must be the same length as X.

          example: `color=['r','g','k']` or a mix: `color=['r',(.4,.2,.9),'#00FF00']`
