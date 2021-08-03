# noinspection PyPackageRequirements
import datawrangler as dw

from ..core.model import apply_model


@dw.decorate.apply_unstacked
def align(data, model='HyperAlign', **kwargs):
    return apply_model(data, model, search=['aligners'], **kwargs)
