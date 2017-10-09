import copy
import deepdish as dd
import numpy as np
from .tools.normalize import normalize as normalizer
from .tools.reduce import reduce as reducer
from .tools.align import align as aligner
from .config import __version__

class DataGeometry(object):
    """
    Hypertools data object

    A DataGeometry data object contains the data, figure handles and transform
    functions used to create a plot.

    Parameters
    ----------

    """

    def __init__(self, fig=None, ax=None, line_ani=None, data=None, xform_data=None,
                 reduce=None, align=None, normalize=None, kwargs=None,
                 version=__version__):

        # matplotlib figure handle
        self.fig = fig

        # matplotlib axis handle
        self.ax = ax

        # matplotlib line_ani handle (if its an animation)
        self.line_ani = line_ani

        # the raw data
        self.data = data

        # the transformed data
        self.xform_data = xform_data

        # dictionary of model and model_params
        self.reduce = reduce

        # 'hyper', 'SRM' or None
        self.align = align

        # 'within', 'across', 'row' or False
        self.normalize = normalize

        # dictionary of kwargs
        self.kwargs = kwargs

        # hypertools version
        self.version = version

    # a function to transform new data
    def transform(self, data=None):
        """
        Return transformed data, or transform new data
        """
        # if no new data passed,
        if data is None:
            return self.xform_data
        else:
            reduce_model = {'model' : self.reduce['model'], 'params' : self.reduce['params']}
            align_model = {'model' : self.align['model'], 'params' : self.align['params']}
            return aligner(reducer(normalizer(data, normalize=self.normalize), reduce=reduce_model, ndims=self.reduce['params']['n_components']), align=align_model)

    # a function to plot the data
    def plot(self, data=None, **kwargs):
        """
        Plot the data object
        """
        # import plot here to avoid circular imports
        from .plot.plot import plot as plotter

        if data is None:
            data = self.xform_data
            transform = False
            if any([k in kwargs for k in ['reduce', 'align', 'normalize']]):
                data = self.data
                transform = True
        else:
            transform = True

        # get kwargs and update with new kwargs
        new_kwargs = copy.copy(self.kwargs)
        for key in kwargs:
            new_kwargs.update({key : kwargs[key]})

        return plotter(data, transform=transform, **new_kwargs)

    def save(self, fname, compression='blosc'):
        # put geo vars into a dict
        geo = {
            'data' : self.data,
            'xform_data' : self.xform_data,
            'reduce' : self.reduce,
            'align' : self.align,
            'normalize' : self.normalize,
            'kwargs' : self.kwargs,
            'version' : self.version
        }

        # if extension wasn't included, add it
        if fname[-4:]!='.geo':
            fname+='.geo'

        # save
        dd.io.save(fname, geo, compression=compression)
