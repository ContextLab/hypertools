"""Central default options for hypertools, parsed from core/config.ini.

Uses datawrangler's config machinery so hypertools and dw share one defaults
mechanism (single source of truth). Values are returned in a RobustDict so a
lookup for an unconfigured function/section yields {} instead of KeyError.
"""
import os

# datawrangler (>=0.5) evaluates ``os.getenv('HOME')`` at import time to build
# its data directory. ``HOME`` is unset on Windows (which uses ``USERPROFILE``),
# so ``os.path.join(None, ...)`` raises ``TypeError`` and importing dw -- and
# therefore hypertools -- crashes on Windows. Point ``HOME`` at the real home
# directory (``expanduser`` resolves it cross-platform) before importing dw.
# Filed upstream: dw should use ``os.path.expanduser`` instead of getenv.
os.environ.setdefault("HOME", os.path.expanduser("~"))

import datawrangler as dw

from .shared import RobustDict


def get_default_options(fname=None):
    """Parse config.ini into a RobustDict of {section: {option: value}}."""
    if fname is None:
        fname = os.path.join(os.path.dirname(__file__), "config.ini")
    merged = dw.core.update_dict(dw.core.get_default_options(),
                                 dw.core.get_default_options(fname))
    return RobustDict(merged, __default_value__={})


def apply_defaults(func_name, kwargs=None):
    """Return the config defaults for ``func_name`` overridden by ``kwargs``."""
    defaults = dict(get_default_options()[func_name])
    if kwargs:
        defaults.update(kwargs)
    return defaults
