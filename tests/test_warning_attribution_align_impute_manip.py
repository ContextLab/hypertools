"""Warnings from align/impute/manip/core name the CALLER's line (1.1 release
review, 2026-09-11).

The review found `hyp.align`'s row-trim warning attributed to
``hypertools/align/common.py``; a sweep of the same modules found nine more
warnings that named a hypertools (or datawrangler) frame instead of the
user's call: fixed ``stacklevel=2`` calls reached through datawrangler's
funnel wrapper, and calls with no stacklevel at all. Python's default
filters only DISPLAY a DeprecationWarning attributed to ``__main__``, so the
deprecated spellings among them were invisible to scripts. Each now uses
`hypertools.core.model.external_stacklevel`, like the rest of the library.
Real calls, no mocks.
"""
import warnings

import numpy as np
import pandas as pd
import pytest

import hypertools as hyp
from hypertools.core.pipeline import Pipeline
from hypertools.impute import PPCA


def _data():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(30, 3))
    y = rng.normal(size=(30, 3))
    gappy = x.copy()
    gappy[3, 1] = np.nan
    return x, y, gappy


def _call(name):
    x, y, gappy = _data()
    if name == 'align= alias':
        return hyp.align([x, y], align='HyperAlign')
    if name == 'impute params dict':
        return hyp.impute(gappy, model={'model': 'PPCA', 'params': {}})
    if name == 'Pipeline params dict':
        return Pipeline([{'model': 'PCA', 'params': {'n_components': 2}}])
    if name == 'manip params dict':
        return hyp.manip(x, model={'model': 'ZScore', 'params': {}})
    if name == 'manip params beside kwargs':
        return hyp.manip(x, model={'model': 'ZScore', 'params': {},
                                   'kwargs': {}})
    if name == 'impute mismatched columns':
        return hyp.impute([pd.DataFrame(gappy, columns=list('abc')),
                           pd.DataFrame(y, columns=list('def'))])
    if name == 'impute dead column':
        dead = pd.DataFrame(gappy, columns=list('abc'))
        dead['c'] = np.nan
        return hyp.impute(dead)
    if name == 'impute instance kwargs':
        return hyp.impute(gappy, model=PPCA(), n_components=2)
    if name == 'Smooth rounds kernel_width':
        return hyp.manip(x, model='Smooth', kernel='boxcar', kernel_width=4.6)
    if name == 'Smooth makes kernel_width odd':
        return hyp.manip(x, model='Smooth', kernel='boxcar', kernel_width=4)
    raise AssertionError(name)


CASES = {
    'align= alias': 'align= is deprecated',
    'impute params dict': "'params': {...}} is deprecated",
    'Pipeline params dict': "'params': {...}} is deprecated",
    'manip params dict': "'params': {...}} is deprecated",
    'manip params beside kwargs': "ignoring the legacy 'params' key",
    'impute mismatched columns': 'do not share columns',
    'impute dead column': 'no observed values at all',
    'impute instance kwargs': 'ignoring keyword argument',
    'Smooth rounds kernel_width': 'Rounding smoothing kernel width',
    'Smooth makes kernel_width odd': 'Increasing smoothing kernel width',
}


@pytest.mark.parametrize('name', sorted(CASES))
def test_the_warning_names_the_callers_line(name):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        _call(name)
    hits = [w for w in caught if CASES[name] in str(w.message)]
    assert hits, [str(w.message) for w in caught]
    assert all(w.filename == __file__ for w in hits), \
        [(w.filename, w.lineno) for w in hits]


def _warn_calls_without_external_stacklevel(path):
    import ast
    with open(path, encoding='utf-8') as handle:
        tree = ast.parse(handle.read(), filename=path)
    missing = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == 'warn'
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == 'warnings'):
            continue
        levels = [kw.value for kw in node.keywords if kw.arg == 'stacklevel']
        if not (levels and isinstance(levels[0], ast.Call)
                and getattr(levels[0].func, 'id', None)
                == 'external_stacklevel'):
            missing.append(node.lineno)
    return missing


def test_every_warning_in_these_packages_uses_external_stacklevel():
    """Static gate against the next misattributed warning: in
    align/impute/manip/core (and tools/analyze.py) every
    ``warnings.warn`` passes ``stacklevel=external_stacklevel()`` -- a
    missing or fixed stacklevel is what named library frames above."""
    import os
    package = os.path.dirname(os.path.abspath(hyp.__file__))
    files = [os.path.join(package, 'tools', 'analyze.py')]
    for sub in ('align', 'impute', 'manip', 'core'):
        folder = os.path.join(package, sub)
        files += [os.path.join(folder, name)
                  for name in sorted(os.listdir(folder))
                  if name.endswith('.py')]
    offenders = {}
    for path in files:
        lines = _warn_calls_without_external_stacklevel(path)
        if lines:
            rel = os.path.relpath(path, package).replace(os.sep, '/')
            offenders[rel] = lines
    assert offenders == {}


def test_the_smooth_warnings_name_a_direct_class_call_too():
    from hypertools.manip import Smooth
    x, _, _ = _data()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        Smooth(kernel='boxcar', kernel_width=4).fit_transform(x)
    hits = [w for w in caught if 'kernel width' in str(w.message)]
    assert hits and all(w.filename == __file__ for w in hits)
