# noinspection PyPackageRequirements
import datawrangler as dw

from ..core.model import apply_model


@dw.decorate.apply_stacked
def cluster(data, model='KMeans', **kwargs):
    """
    Cluster the data and return a list of cluster labels

    Parameters
    ----------
    :param data: any hypertools-compatible dataset
    :param model: a string containing the name of any of the following scikit-learn (or compatible) models (default:
      'KMeans'):
       - A discrete cluster model: https://scikit-learn.org/stable/modules/classes.html#module-sklearn.cluster
       - A Gaussian mixture model: https://scikit-learn.org/stable/modules/classes.html#module-sklearn.mixture
       Models can also be specified as any other scikit-learn compatible model, or a list of models/strings.  To
       customize the behavior of any model, models may also be passed as dictionaries with fields:
         - 'model': the model(s) to apply
         - 'args': a list of unnamed arguments (appended to data)
         - 'kwargs': a list of named keyword arguments
    :param kwargs: keyword arguments are first passed to datawrangler.decorate.funnel, and any remaining arguments
      are passed onto the model initialization function.  Keyword arguments override any model-specific parameters.

    Returns
    -------
    :return: a DataFrame (or list of DataFrames) containing the cluster labels or mixture proportions
    """
    return apply_model(data, model, search=['cluster', 'mixture'], **kwargs)
