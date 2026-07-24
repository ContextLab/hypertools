# -*- coding: utf-8 -*-
"""Controller-applied audit fixes (release-1.0 audit, wave-5 integration).

Cross-cutting items no fix batch owned: top-level exception re-exports
(F23-005), supported_models export (F21-005), vals2colors/vals2bins palette
coverage (F24-005), and config version sourcing (F23-009). Real calls only
-- no mocks.
"""
from importlib.metadata import version as _pkg_version

import matplotlib
matplotlib.use('Agg')

import numpy as np
import seaborn as sns

import hypertools as hyp
from hypertools._shared.helpers import vals2bins, vals2colors


def test_exceptions_reexported_at_top_level():
    # F23-005: the documented exception classes are importable from the
    # package root and share the HypertoolsError base
    assert issubclass(hyp.HypertoolsIOError, hyp.HypertoolsError)
    assert issubclass(hyp.HypertoolsBackendError, hyp.HypertoolsError)
    from hypertools.core.exceptions import HypertoolsIOError
    assert hyp.HypertoolsIOError is HypertoolsIOError


def test_supported_models_exported_and_lists_models():
    # F21-005: supported_models() is a public top-level export
    models = hyp.supported_models()
    assert isinstance(models, list) and len(models) > 10
    assert 'KMeans' in models


def test_vals2colors_uses_full_palette_range():
    # F24-005: the max value maps to the LAST palette color (a stray max+1
    # bin edge previously left the top of the colormap unused)
    res = 100
    palette = [tuple(c) for c in np.array(sns.color_palette('viridis', res))]
    cols = vals2colors(list(np.linspace(0.0, 10.0, 21)), cmap='viridis', res=res)
    assert cols[0] == palette[0]
    assert cols[-1] == palette[-1]


def test_vals2colors_constant_input_is_safe():
    cols = vals2colors([3.0, 3.0, 3.0], cmap='viridis', res=100)
    assert len(cols) == 3 and len(set(cols)) == 1


def test_vals2bins_spans_full_bin_range_without_overflow():
    res = 50
    bins = vals2bins(list(np.linspace(0.0, 1.0, 200)), res=res)
    assert min(bins) == 0
    assert max(bins) == res - 1  # top bin reachable, never out of range
    assert vals2bins([7, 7, 7], res=res) == [0, 0, 0]


def test_version_sourced_from_installed_metadata():
    # F23-009: __version__ comes from importlib.metadata, no py<3.8 shim
    assert hyp.__version__ == _pkg_version('hypertools')
