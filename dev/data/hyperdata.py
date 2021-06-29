import numpy as np
import pandas as pd

from ..core.configurator import __version__
from ..decorate import interpolate, list_generalizer, funnel
from ..manip.manip import pipeline, pandas_unstack, pandas_stack, pandas_flatten
from ..manip.smooth import smooth

from .format import format_data


class HyperData(pd.DataFrame):
    def __init__(self, data, wrangler=None, dtype=None, index=None, columns=None, copy=False, **kwargs):
        super().__init__(data, index=index, columns=columns, dtype=dtype, copy=copy)

        for k, v in kwargs.items():
            assert k not in ['df', '__version__', 'stacked'], RuntimeError(f'Cannot set reserved property: {k}')
            setattr(self, k, v)

        self.df = None
        self.dtype = None

        if callable(wrangler):
            self.df = wrangler(data)
            self.dtype = dtype
        else:
            self.df, self.dtype = format_data(data, return_dtype=True, **kwargs)

        self.version = __version__

    def hyper_unstack(self, inplace=False):
        if inplace:
            setattr(self, 'data', pandas_unstack(self.data))
        else:
            return pandas_unstack(self.data)

    # noinspection PyUnusedLocal
    def hyper_stack(self, inplace=False, **kwargs):
        if inplace:
            setattr(self, 'data', pandas_stack(self.data, **kwargs))
        else:
            return pandas_stack(self.data, **kwargs)

    def hyper_flatten(self, inplace=False):
        if inplace:
            setattr(self, 'data', pandas_flatten(self.data))
        else:
            return pandas_flatten(self.data)

    def trajectorize(self,  window_length=100, dw=10, inplace=False):
        @list_generalizer
        def trajectory_helper(x):
            trajectory = pd.DataFrame(columns=self.columns)
            try:
                start_time = np.min(x.index.values)
                end_time = np.max(x.index.values)
            except TypeError:
                return None

            window_start = start_time
            while window_start < end_time:
                window_end = np.min([window_start + window_length - dw, end_time])
                # noinspection PyBroadException
                try:
                    trajectory.loc[np.mean([window_start, window_end])] = x.loc[window_start:window_end].mean(axis=0)
                except:
                    pass
            return trajectory

        trajectories = trajectory_helper(self.hyper_unstack(inplace=False))
        if inplace:
            setattr(self, 'data', trajectories)
        else:
            return trajectories

    def apply_pipeline(self, inplace=False, ops=None):
        if inplace:
            setattr(self, 'data', pipeline(self.data, ops))
        else:
            return pipeline(self.data, ops)

    def smooth(self, inplace=False):
        smoothed_trajectories = smooth(self.hyper_unstack(inplace=False), inplace=inplace, **kwargs)
        if inplace:
            setattr(self, 'data', smoothed_trajectories)
        else:
            return smoothed_trajectories

    def hyperalign(self, inplace=False, **kwargs):
        aligned_trajectories = align(self.data, **kwargs)
        if inplace:
            setattr(self, 'data', aligned_trajectories)
        else:
            return aligned_trajectories

    def reduce(self, inplace=False, **kwargs):
        reduced_trajectories = apply_defaults(reduce(self.data, **kwargs))
        if inplace:
            setattr(self, 'data', reduced_trajectories)
        else:
            return reduced_trajectories

    def cluster(self, **kwargs):
        pass

    def plot(self, **kwargs):
        pass

    def save(self, fname, **kwargs):
        pass
