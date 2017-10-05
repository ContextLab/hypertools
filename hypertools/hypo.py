from .tools.normalize import normalize as normalizer
from .tools.reduce import reduce as reducer
from .tools.align import align as aligner
import copy

class HypO(object):
    """
    Hypertools data object

    A Hypo data object contains the data, figure handles and transform functions
    used to create a plot.

    Parameters
    ----------

    """

    def __init__(self, fig=None, ax=None, line_ani=None, data=None, reduce=None,
                 align=None, normalize=None, args=None, version=None):

        # matplotlib figure handle
        self.fig = fig

        # matplotlib axis handle
        self.ax = ax

        # matplotlib line_ani handle (if its an animation)
        self.line_ani = line_ani

        # the transformed data
        self.data = data

        # dictionary of model and model_params
        self.reduce = reduce

        # 'hyper', 'SRM' or None
        self.align = align

        # 'within', 'across', 'row' or False
        self.normalize = normalize

        # dictionary of non-transform args
        self.args = args

        # hypertools version
        self.version = version

    # a function to transform new data
    def transform(self, data):
        hypO = copy.copy(self)
        hypO.data =  aligner(reducer(normalizer(data, normalize=self.normalize), model=self.reduce['model'], model_params=self.reduce['params'], ndims=self.reduce['ndims']), model=self.align['model'], model_params=self.align['params'])
        return hypO

    # a function to plot the data
    def plot(self, *args, **kwargs):
        from .plot.plot import plot as plotter
        plotter(self.data, *args, **kwargs)
