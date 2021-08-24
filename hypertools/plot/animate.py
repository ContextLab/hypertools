# noinspection PyPackageRequirements
import datawrangler as dw
import numpy as np
import plotly.graph_objects as go

from scipy.spatial.distance import cdist

from ..manip import manip
from ..core import get, get_default_options, eval_dict

from .static import static_plot, get_bounds

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
            indices = list(index_vals)
            indices.sort()

            duration = len(indices)
            window_length = int(np.round(duration * self.focused / self.duration))
            self.window_starts = np.concatenate([np.zeros([window_length]), np.arange(len(indices) - window_length)])
            self.window_ends = np.arange(len(self.window_starts))

            tail_window_length = int(np.round(duration * self.unfocused / self.duration))
            self.tail_window_starts = np.concatenate([np.zeros([tail_window_length]),
                                                      np.arange(len(indices) - tail_window_length)])
            self.tail_window_ends = np.arange(len(self.tail_window_starts))
            self.tail_window_precogs = np.concatenate([tail_window_length +
                                                       np.arange(len(indices) - tail_window_length),
                                                       indices[-1] * np.ones([tail_window_length])])

            self.angles = np.linspace(0, self.rotations * 360, len(self.window_starts))[:-1]

    def build_animation(self):
        frame_duration = 1000 * self.duration / len(self.angles)

        # set up base figure
        fig = self.fig.to_dict().copy()

        bounds = get_bounds(self.data)
        layout = go.Layout(
            xaxis={'range': [bounds[0, 0], bounds[1, 0]], 'autorange': False},
            yaxis={'range': [bounds[0, 0], bounds[1, 0]], 'autorange': False},
            updatemenus=[{
                'type': 'buttons',
                'buttons': [{'label': 'Play',
                             'method': 'animate',
                             'args': [None]}]
            }]
        )

        frames = []
        for i in range(10):
            frames.append(self.get_frame(i))

        fig['layout'] = layout
        fig['frames'] = frames
        fig['data'] = frames[-1].data
        return go.Figure(fig)

        # # add buttons and slider and define transitions
        # fig['layout']['updatemenus'] = [{'buttons': [{
        #     'label': '▶',  # play button
        #     'args': [None, {'frame': {'duration': 250, 'redraw': True},
        #                     'fromcurrent': False,
        #                     'transition': {'duration': 0}}],
        #     'method': 'animate'}, {
        #     'label': '||',  # pause button
        #     'args': [[None], {'frame': {'duration': 0, 'redraw': True},
        #                       'mode': 'immediate',
        #                       'transition': {'duration': 0}}],
        #     'method': 'animate'}],
        #     # slider
        #     'direction': 'left',
        #     'pad': {'r': 10, 't': 87},
        #     'showactive': False,
        #     'type': 'buttons',
        #     'x': 0.1,
        #     'xanchor': 'right',
        #     'y': 0,
        #     'yanchor': 'top'}]
        #
        # # define slider behavior
        # slider = {
        #     'active': 0,
        #     'yanchor': 'top',
        #     'xanchor': 'left',
        #     'currentvalue': {
        #         'font': {'size': 12},
        #         'prefix': 'Frame ',
        #         'visible': True,
        #         'xanchor': 'right'
        #     },
        #     # 'transition': {'duration': frame_duration / 2, 'easing': 'cubic-in-out'},
        #     'transition': {'duration': 0},
        #     'pad': {'b': 10, 't': 50},
        #     'len': 0.9,
        #     'x': 0.1,
        #     'y': 0,
        #     'steps': []
        # }
        #
        # # add frames
        # frames = []
        # for i in range(10):  # len(self.angles)):
        #     frames.append(self.get_frame(i))
        #     slider_step = {'args': [[i],
        #                             {'frame': {'duration': 250, 'redraw': True},
        #                              'mode': 'immediate',
        #                              'transition': {'duration': 0}}],
        #                    'label': str(i),
        #                    'method': 'animate'}
        #     slider['steps'].append(slider_step)
        # fig['frames'] = frames
        #
        # # connect slider to frames
        # fig['layout']['sliders'] = [slider]
        #
        # return go.Figure(fig)

    @staticmethod
    def data2frame(data, i=''):
        return go.Frame(data=data)  # , name=str(i))

    def get_frame(self, i):
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
            return self.data2frame(self.animate_window(i), i)
        elif self.style == 'chemtrails':
            return self.data2frame(self.animate_chemtrails(i), i)
        elif self.style == 'precog':
            return self.data2frame(self.animate_precog(i), i)
        elif self.style == 'bullettime':
            return self.data2frame(self.animate_bullettime(i), i)
        elif self.style == 'grow':
            return self.data2frame(self.animate_grow(i), i)
        elif self.style == 'shrink':
            return self.data2frame(self.animate_shrink(i), i)
        elif self.style == 'spin':
            return self.data2frame(self.fig, i)
        else:
            raise ValueError(f'unknown animation mode: {self.mode}')

    @staticmethod
    @dw.decorate.list_generalizer
    def get_window(x, w_start, w_end):
        return x.loc[w_start:w_end]

    def get_opts(self):
        opts = self.opts.copy()
        _ = opts.pop('opacity', None)
        return {**opts,
                'opacity': self.focused_alpha}

    def tail_opts(self):
        return dw.core.update_dict(self.get_opts(), {'opacity': self.unfocused_alpha})

    def animate_window(self, i):
        return static_plot(self.get_window(self.data, self.window_starts[i], self.window_ends[i]), return_shapes=True,
                           **self.get_opts())

    def animate_chemtrails(self, i):
        return [*static_plot(self.get_window(self.data, self.tail_window_starts[i], self.tail_window_ends[i]),
                             return_shapes=True, **self.tail_opts()), *self.animate_window(i)]

    def animate_precog(self, i):
        return [*static_plot(self.get_window(self.data, self.tail_window_ends[i], self.tail_window_precogs[i]),
                             return_shapes=True, **self.tail_opts()), *self.animate_window(i)]

    def animate_bullettime(self, i):
        return [*static_plot(self.data, return_shapes=True, **self.tail_opts()), *self.animate_window(i)]

    def animate_grow(self, i):
        return static_plot(get_window(self.data, np.zeros_like(self.window_ends[i]), self.window_ends[i]),
                           return_shapes=True, **self.get_opts())

    def animate_shrink(self, i):
        return static_plot(get_window(self.data, self.window_ends[i],
                                      (len(self.window_ends) - 1) * self.ones_like(self.window_ends[i])),
                           return_shapes=True, **self.get_opts())
