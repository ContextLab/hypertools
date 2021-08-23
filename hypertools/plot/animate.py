# noinspection PyPackageRequirements
import datawrangler as dw
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.animation import FuncAnimation

from ..core import get

from .static import static_plot


class Animator:
    def __init__(self, fig, data, mode, angles, zooms, opts):
        self.fig = fig
        self.mode = mode
        self.opts = opts
        self.angles = angles
        self.zooms = zooms
        self.data = data

        if type(data) is list:
            c = np.max([d.shape[1] for d in data])
        else:
            c = data.shape[1]
        self.proj = f'{c}d'

        # don't necessarily need to throw an error here-- "spinning" in 2d should just look like a static image that
        # plays for the requested duration
        assert not (self.proj == '2d') and (self.mode == 'spin'),\
            RuntimeError('Spin animations are not supported for 2d plots')
        if self.mode == 'spin':
            self.fig = static_plot(self.data, **self.opts)
        else:
            if dw.zoo.is_dataframe(data):
                index_vals = set(data.index.values)
            else:  # data is a list
                index_vals = {}
                for d in data:
                    index_vals = index_vals.union(set(d.index.values))
            indices = list(index_vals)
            indices.sort()

            duration = len(indices)
            window_length = np.round(duration * self.opts['focused'] / self.opts['duration'])
            self.window_starts = np.concatenate([np.zeros([window_length]), np.arange(len(indices) - window_length)])
            self.window_ends = np.arange(len(self.window_starts))

            tail_window_length = np.round(duration * self.opts['unfocused'] / self.opts['duration'])
            self.tail_window_starts = np.concatenate([np.zeros([tail_window_length]),
                                                      np.arange(len(indices) - tail_window_length)])
            self.tail_window_ends = np.arange(len(self.tail_window_starts))
            self.tail_window_precogs = self.window_ends + np.arange(len(self.tail_window_starts))

    def __call__(self, i):
        if self.proj == '3d':
            self.ax.view_init(elev=elev, azim=get(self.angles, i))
            self.ax.dist = 9 - get(self.zooms, i)

        if self.mode == 'window':
            return self.animate_window(i)
        elif self.mode == 'chemtrails':
            return self.animate_chemtrails(i)
        elif self.mode == 'precog':
            return self.animate_precog(i)
        elif self.mode == 'bullettime':
            return self.animate_bullettime(i)
        elif self.mode == 'grow':
            return self.animate_grow(i)
        elif self.mode == 'shrink':
            return self.animate_shrink(i)
        elif self.mode == 'spin':
            return self.ax
        else:
            raise ValueError(f'unknown animation mode: {self.mode}')

    @staticmethod
    @dw.decorate.list_generalizer
    def get_window(x, w_start, w_end):
        return x.loc[w_start:w_end]

    @staticmethod
    def tail_opts(opts):
        alpha = opts.pop('unfocused_alpha', opts['alpha'])
        x = opts.copy()
        x['alpha'] = alpha
        return x

    def animate_window(self, i):
        self.fig = static_plot(get_window(self.data, self.window_starts[i], self.window_ends[i]), **self.opts)
        return self.fig

    def animate_chemtrails(self, i):
        self.fig = static_plot(get_window(self.data, self.tail_window_starts[i], self.tail_window_ends[i]),
                               **tail_opts(self.opts))
        return self.animate_window(i)

    def animate_precog(self, i):
        self.fig = static_plot(get_window(self.data, self.tail_window_ends[i], self.tail_window_precogs[i]),
                               **tail_opts(self.opts))
        return self.animate_window(i)

    def animate_bullettime(self, i):
        self.fig = static_plot(self.data, **tail_opts(self.opts))
        return self.animate_window(i)

    def animate_grow(self, i):
        self.fig = static_plot(get_window(self.data, np.zeros_like(self.window_ends[i]), self.window_ends[i]),
                               **self.opts)
        return self.fig

    def animate_shrink(self, i):
        self.fig = static_plot(get_window(self.data, self.window_ends[i],
                                          (len(self.window_ends) - 1) * self.ones_like(self.window_ends[i])),
                               **self.opts)
        return self.fig
