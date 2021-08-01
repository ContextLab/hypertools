# noinspection PyPackageRequirements
import datawrangler as dw

from ..core.model import apply_model


@dw.decorate.apply_stacked
def reduce(data, model='IncrementalPCA', **kwargs):
    return apply_model(data, model, search=['decomposition', 'manifold', 'mixture', 'umap'], **kwargs)
