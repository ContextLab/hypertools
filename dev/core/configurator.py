# noinspection PyPackageRequirements
import datawrangler as dw

__version__ = get_distribution('hypertools')


def get_default_options(fname='config.ini'):
    return dw.core.update_dict(dw.core.get_default_options(), dw.core.get_default_options(fname))
