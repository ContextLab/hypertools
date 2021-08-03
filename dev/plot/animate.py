# noinspection PyPackageRequirements
import datawrangler as dw
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.animation import FuncAnimation

from ..core import get

from .static import static_plot


class Animator:
    def __init__(self, ax, data, mode, angles, zooms, opts):
        self.ax = ax
        self.mode = mode
        self.opts = opts
        self.angles = angles
        self.zooms = zooms
        self.data = data

        if 'proj' in ax.properties().keys():
            self.proj = '3d'
        else:
            self.proj = '2d'

        assert not (self.proj == '2d') and (self.mode == 'spin'),\
            RuntimeError('Spin animations are not supported for 2d plots')
        if self.mode == 'spin':
            self.ax = static_plot(self.data, **self.opts)
        else:
            if dw.zoo.is_dataframe(data):
                index_vals = set(data.index.values)
            else:  # data is a list
                index_vals = {}
                for d in data:
                    index_vals = index_vals.union(set(d.index.values))
            indices = list(index_vals)
            indices.sort()
            raise NotImplementedError('stopped here...need to compute sliding windows from indices')

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
            return self.animate_spin(i)
        else:
            raise ValueError(f'unknown animation mode: {self.mode}')


    def animate_window(self, i):
        pass

    def animate_chemtrails(self, i):
        pass

    def animate_precog(self, i):
        pass

    def animate_bullettime(self, i):
        pass

    def animate_spin(self, i):
        pass
