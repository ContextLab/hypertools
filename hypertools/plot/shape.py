from ..tools.format_data import format_data
from ..tools.reduce import reduce as reducer

import numpy as np
import seaborn as sns
import six
from matplotlib import pyplot as plt

lines = ['-', ':', '-.'] #TODO: complete this list
dots = ['.', 'o', 'd', 's', '^'] #TODO: complete this list
colors = ['r', 'g', 'b', 'w', 'c', 'm', 'y', 'k', 'w'] #TODO: complete this list

def parse_format(fmt):
    style = {'line': None, 'marker': None, 'color': None}
    if fmt is None:
        style['line'] = '-'
        style['color'] = 'k'

    assert isinstance(fmt, six.string_types), 'Format must be a string'
    for c in fmt:
        if c in lines:
            assert style['line'] is None, 'Cannot specify multiple line styles in the same format string'
            style['line'] = c
        elif c in dots:
            assert style['marker'] is None, 'Cannot specify multiple marker styles in the same format string'
            style['marker'] = c
        elif c in colors:
            assert style['color'] is None, 'Cannot specify multiple colors in the same format string'
            style['color'] = c
        else:
            raise Exception('Invalid format string')
    return style

def rgb(s):
    if s.lower() == 'r':
        x = [1, 0, 0]
    elif s.lower() == 'g':
        x = [0, 1, 0]
    elif s.lower() == 'b':
        x = [0, 0, 1]
    elif s.lower() == 'w':
        x = [1, 1, 1]
    elif s.lower() == 'c':
        x = [83 / 255., 209 / 255., 237 / 255.]
    elif s.lower() == 'm':
        x = [247 / 255., 0 / 255., 94 / 255.]
    elif s.lower() == 'y':
        x = [1, 1, 0]
    elif s.lower() == 'k':
        x = [0, 0, 0]
    elif isinstance(s, np.ndarray) and (len(s) == 3 or len(s) == 4):
        x = s
    else:
        raise Exception('Unknown color')
    return [float(i) for i in x]



class Shape():
    '''
    Store a single plotable shape (1d, 2d, or 3d numpy array)
    '''
    def __init__(self, data, style='-', linewidth=1, hue=None, cmap='Spectral', name=None):
        assert isinstance(data, np.ndarray), 'Plottable shape data must be a Numpy array'
        assert data.shape[1] <= 3, 'Plottable shape data must have 3 or fewer dimensions'

        self.data = data
        self.style = style
        self.linewidth = linewidth
        self.hue = hue
        self.cmap = cmap
        self.name = name

    def plot(self):
        style = parse_format(style)

        if not (hue is None):
            assert isinstance(hue, np.ndarray)
            if hue.shape[0] == 1:
                assert (hue.shape[1] == 3) or (hue.shape[1] == 4), 'Single hue must be a 3d or 4d numpy array of (r,g,b) or (r,g,b,a) values'
            else:
                assert hue.shape[0] == self.data.shape[0], 'Must specify either a single hue or a hue for each datapoint'
                nbins = 100
                colors = sns.color_palette(self.cmap, n_colors=nbins)
                hue = colors[np.digitize(reducer(hue, 1), nbins), :]
        else:
            hue = rgb(style['color'])

        #idea:
        #create a list of line segment objects (as needed) and marker objects (as needed)
        #depending on animation options, draw out the plot



