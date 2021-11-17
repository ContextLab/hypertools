# noinspection PyPackageRequirements
import datawrangler as dw
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from scipy.spatial.distance import cdist

from ..manip import manip
from ..core import get, get_default_options, eval_dict

from .static import static_plot, get_bounds, flatten

defaults = eval_dict(get_default_options()['animate'])


class Animator:
    def __init__(self, data, **kwargs):
        self.data = data

        self.fig = kwargs.pop('fig', go.Figure())
        self.style = kwargs.pop('style', defaults['style'])
        self.focused = kwargs.pop('focused', defaults['focused'])
        self.focused_alpha = kwargs.pop('focused_alpha', defaults['focused_alpha'])
        self.unfocused = kwargs.pop('unfocused', defaults['unfocused'])
        self.unfocused_alpha = kwargs.pop('unfocused_alpha', defaults['unfocused_alpha'])
        self.rotations = kwargs.pop('rotations', defaults['rotations'])
        self.framerate = kwargs.pop('framerate', defaults['framerate'])
        self.duration = kwargs.pop('duration', defaults['duration'])
        self.elevation = kwargs.pop('elevation', defaults['elevation'])
        self.zooms = kwargs.pop('zoom', defaults['zoom'])
        self.opts = kwargs

        assert data is not None, ValueError('No dataset provided.')

        stacked_data = dw.stack(data)
        self.center = np.atleast_2d(stacked_data.mean(axis=0).values)
        self.proj = f'{stacked_data.shape[1]}d'

        self.zooms = np.multiply(self.zooms, np.max(cdist(self.center, stacked_data.values)))
        self.indices = None

        if self.style == 'spin':
            self.fig = static_plot(self.data, fig=self.fig, **self.opts)
            self.angles = np.linspace(0, self.rotations * 360, self.duration * self.framerate + 1)[:-1]
        else:
            if dw.zoo.is_dataframe(data):
                index_vals = set(data.index.values)
            else:  # data is a list
                index_vals = set()
                for d in data:
                    index_vals = index_vals.union(set(d.index.values))

            # union of unique indices
            indices = list(index_vals)

            # compress or stretch (repeat) indices to match the requested duration and framerate
            self.indices = np.linspace(np.min(indices), np.max(indices), self.duration * self.framerate + 1)
            n_frames = len(self.indices)

            window_length = int(np.floor(n_frames * self.focused / self.duration))
            self.window_starts = np.concatenate([np.zeros([window_length]),
                                                 np.arange(1, len(self.indices) - window_length)])
            self.window_ends = np.arange(1, self.window_starts[-1] + window_length + 1)

            tail_window_length = int(np.round(n_frames * self.unfocused / self.duration))
            self.tail_window_starts = np.concatenate([np.zeros([tail_window_length]),
                                                      np.arange(1, len(self.indices) - tail_window_length)])
            self.tail_window_ends = np.abs(np.multiply(self.window_starts - 1, self.window_starts >= 1))
            self.tail_window_precogs = np.concatenate([tail_window_length +
                                                       np.arange(1, len(self.indices) - tail_window_length),
                                                       self.indices[-1] * np.ones([tail_window_length])])

            self.angles = np.linspace(0, self.rotations * 360, len(self.window_starts) + 1)[:-1]

    def build_animation(self):
        frame_duration = 1000 * self.duration / len(self.angles)

        # set up base figure
        fig = self.fig.to_dict().copy()

        # add buttons and slider and define transitions
        fig['layout']['updatemenus'] = [{'buttons': [{
            'label': '▶',  # play button
            'args': [None, {'frame': {'duration': frame_duration, 'redraw': False},
                            'fromcurrent': True,
                            'transition': {'duration': 0}}],
            'method': 'animate'}, {
            'label': '||',  # stop/pause button
            'args': [[None], {'frame': {'duration': 0, 'redraw': True},
                              'mode': 'immediate',
                              'transition': {'duration': 0}}],
            'method': 'animate'}],
            # slider
            'direction': 'left',
            'pad': {'r': 10, 't': 87},
            'showactive': False,
            'type': 'buttons',
            'x': 0.1,
            'xanchor': 'right',
            'y': 0,
            'yanchor': 'top'}]

        bounds = get_bounds(self.data)
        fig['layout']['xaxis'] = {'range': [bounds[0, 0], bounds[1, 0]], 'autorange': False}
        fig['layout']['yaxis'] = {'range': [bounds[0, 1], bounds[1, 1]], 'autorange': False}
        if bounds.shape[1] == 3:
            fig['layout']['zaxis'] = {'range': [bounds[0, 2], bounds[1, 2]], 'autorange': False}

        # define slider behavior
        slider = {
            'active': 0,
            'yanchor': 'top',
            'xanchor': 'left',
            'currentvalue': {
                'font': {'size': 12},
                'prefix': 'Frame ',
                'visible': True,
                'xanchor': 'right'
            },
            'transition': {'duration': 0},
            'pad': {'b': 10, 't': 50},
            'len': 0.9,
            'x': 0.1,
            'y': 0,
            'steps': []
        }
        for i in range(len(self.angles)):
            slider_step = {'args': [[i],
                                    {'frame': {'duration': frame_duration, 'redraw': False},
                                     'mode': 'immediate',
                                     'transition': {'duration': 0}}],
                           'label': str(i),
                           'method': 'animate'}
            slider['steps'].append(slider_step)

        # connect slider to frames
        fig['layout']['sliders'] = [slider]

        # add frames
        fig['data'] = self.get_frame(0).data
        fig['frames'] = [self.get_frame(i, simplify=True) for i in range(len(self.angles))]

        return go.Figure(fig)

    def get_frame(self, i, simplify=False):
        if self.proj == '3d':
            center = dw.stack(data).mean(axis=0).values
            angle = np.deg2rad(get(self.angles, i))
            zoom = get(self.zooms, i)
            elevation = zoom * np.sin(np.deg2rad(self.elevation)) + self.center[0, 2]

            camera = dict(
                up=dict(x=0, y=0, z=1),
                center=center,
                eye=dict(x=center[0] + zoom * np.cos(angle), y=center[1] + zoom * np.sin(angle), z=elevation)
            )
            self.fig.update_layout(scene_camera=camera)

        if self.style == 'window':
            return self.animate_window(i, simplify=simplify)
        elif self.style == 'chemtrails':
            return self.animate_chemtrails(i, simplify=simplify)
        elif self.style == 'precog':
            return self.animate_precog(i, simplify=simplify)
        elif self.style == 'bullettime':
            return self.animate_bullettime(i, simplify=simplify)
        elif self.style == 'grow':
            return self.animate_grow(i, simplify=simplify)
        elif self.style == 'shrink':
            return self.animate_shrink(i, simplify=simplify)
        elif self.style == 'spin':
            if simplify:
                return go.Frame(data=Animator.get_datadict(data))
            return static_plot(self.data, **self.get_opts())
        else:
            raise ValueError(f'unknown animation mode: {self.mode}')

    def get_window(self, x, w_start, w_end):
        if type(x) is list:
            return [self.get_window(i, w_start, w_end) for i in x]

        return x.loc[self.indices[int(w_start)]:self.indices[int(w_end)]]

    @classmethod
    def get_datadict(cls, data):
        if type(data) is list:
            return [cls.get_datadict(d)[0] for d in data]
        elif data.shape[1] == 2:
            return [go.Scatter(x=flatten(data.values[:, 0]), y=flatten(data.values[:, 1]))]
        elif data.shape[1] == 3:
            return [go.Scatter3d(x=flatten(data.values[:, 0]), y=flatten(data.values[:, 1]),
                                 z=flatten(data.values[:, 2]))]
        else:
            raise ValueError(f'data must be either 2D or 3D; given: {data.shape[1]}D')

    def get_opts(self):
        opts = self.opts.copy()
        _ = opts.pop('opacity', None)
        return {**opts,
                'opacity': self.focused_alpha}

    def tail_opts(self):
        return dw.core.update_dict(self.get_opts(), {'opacity': self.unfocused_alpha})

    def animate_helper(self, i, starts=None, ends=None, extra_starts=None, extra_ends=None, simplify=False):
        if starts is None:
            starts = self.window_starts
        if ends is None:
            ends = self.window_ends

        window = self.get_window(self.data, starts[i], ends[i])
        if extra_starts is not None:
            extra = self.get_window(self.data, extra_starts[i], extra_ends[i])
        else:
            extra = None

        if simplify:
            if extra is not None:
                return go.Frame(data=[*Animator.get_datadict(window), *Animator.get_datadict(extra)], name=str(i))
            else:
                return go.Frame(data=Animator.get_datadict(window), name=str(i))
        else:
            fig = static_plot(window, **self.get_opts())
            if extra is not None:
                static_plot(extra, **self.tail_opts(), fig=fig, showlegend=False)
            return fig

    def animate_window(self, i, simplify=False):
        return self.animate_helper(i, self.window_starts, self.window_ends, simplify=simplify)

    def animate_chemtrails(self, i, simplify=False):
        return self.animate_helper(i, extra_starts=self.tail_window_starts,
                                   extra_ends=self.tail_window_ends,
                                   simplify=simplify)

    def animate_precog(self, i, simplify=False):
        return self.animate_helper(i, extra_starts=self.window_ends, extra_ends=self.tail_window_precogs,
                                   simplify=simplify)

    def animate_bullettime(self, i, simplify=False):
        return self.animate_helper(i, extra_starts=np.zeros_like(self.window_starts),
                                   extra_ends=np.ones_like(self.window_ends),
                                   simplify=simplify)

    def animate_grow(self, i, simplify=False):
        return self.animate_helper(i, starts=np.zeros_like(self.window_starts), simplify=simplify)

    def animate_shrink(self, i, simplify=False):
        return self.animate_helper(i, starts=self.window_ends,
                                   ends=(len(self.window_ends) - 1) * self.ones_like(self.window_ends[i]),
                                   simplify=simplify)
