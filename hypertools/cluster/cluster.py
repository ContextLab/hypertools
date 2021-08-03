# noinspection PyPackageRequirements
import datawrangler as dw

from ..core.model import apply_model


@dw.decorate.apply_stacked
def cluster(data, model='KMeans', **kwargs):
    return apply_model(data, model, search=['cluster', 'mixture', 'ppca'], **kwargs)
