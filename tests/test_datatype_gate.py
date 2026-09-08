# -*- coding: utf-8 -*-
"""Static gate: hypertools does not classify datatypes itself.

Jeremy's rule (datatype audit, 2026-09-08): no function does its own
datatype checking; it defers to datawrangler (``dw.wrangle`` / ``dw.funnel``
/ the ``dw.zoo`` predicates, via the ``hypertools._shared.helpers``
wrappers ``is_array_dataset``/``is_frame_dataset``/``is_series_like``/
``as_pandas_dataframe`` and ``hypertools.core.shared.as_dataframe``), so
polars -- and whatever datawrangler recognises next -- works everywhere for
free.

This test parses every module under ``hypertools/`` (except the shared
coercion layer itself and the vendored third-party code) and fails on any

- ``isinstance(x, ...)`` whose class tuple names ``pd.DataFrame``,
  ``pd.Series`` or ``np.ndarray`` (qualified or bare), or
- ``hasattr(x, 'columns' | 'to_numpy' | 'values')`` (pandas duck-typing),

unless the site is in ``ALLOWLIST`` -- the sites judged legitimate because
they test something that is NOT a user dataset: a hypertools object, a
model spec, an already-wrangled internal frame, a single stream sample or
scalar. Every allowlist entry must still match a real site, so the list
cannot go stale: removing a site means removing its entry.
"""
import ast
import os

import pytest

import hypertools

PACKAGE_DIR = os.path.dirname(os.path.abspath(hypertools.__file__))

#: modules that ARE the coercion layer (they may name pandas/numpy types),
#: plus vendored third-party code
EXCLUDED = ('_shared/helpers.py', 'tools/format_data.py', 'external/',
            '_externals/')

#: names that make an ``isinstance`` second argument a datatype check
FLAGGED_TYPES = ('pd.DataFrame', 'pandas.DataFrame', 'pd.Series',
                 'pandas.Series', 'np.ndarray', 'numpy.ndarray')
FLAGGED_BARE = ('DataFrame', 'Series', 'ndarray')
FLAGGED_ATTRS = ('columns', 'to_numpy', 'values')

#: {repo-relative path: [(substring of the offending call, reason), ...]}.
#: A pattern matches when it is a substring of the unparsed call text.
ALLOWLIST = {
    'hypertools/io/sources.py': [
        ("isinstance(target, pd.DataFrame)",
         "scikit-learn Bunch.target (sklearn's own object, a Series or "
         "frame by sklearn's contract), not user data"),
    ],
    'hypertools/io/streaming.py': [
        ("isinstance(row, pd.Series)",
         "ONE stream sample (a row produced by the stream source), not a "
         "dataset; the stream itself is classified by is_stream"),
        ("isinstance(val, (list, tuple, np.ndarray))",
         "one FIELD of a stream sample (a scalar or short vector), not a "
         "dataset"),
    ],
    'hypertools/predict/predict.py': [
        ("isinstance(t, np.ndarray)",
         "the forecast horizon t= is a scalar; a 0-d numpy array is "
         "unwrapped to its Python scalar"),
    ],
    'hypertools/predict/arima.py': [
        ("isinstance(component, (list, tuple, np.ndarray))",
         "an ARIMA order= component (model spec), not data"),
    ],
    # plot OPTIONS: each of these decides whether a keyword argument is one
    # value or a per-dataset/per-point SEQUENCE of values (a format string
    # vs a list of them, one color vs a palette, one flag vs a flag per
    # dataset). They classify options, not datasets.
    'hypertools/plot/colors.py': [
        ("isinstance(value, (list, tuple, np.ndarray))",
         "a palette= value: one color name vs a sequence of colors"),
        ("isinstance(palette, (list, tuple, np.ndarray))",
         "a palette= option: a named palette vs an explicit color list"),
    ],
    'hypertools/plot/forecast.py': [
        ("isinstance(fmt, (list, tuple, np.ndarray))",
         "a fmt= option: one format string vs one per dataset"),
        ("isinstance(v, (list, tuple, np.ndarray))",
         "a per-forecast styling option: one value vs one per forecast"),
    ],
    'hypertools/plot/matplotlib_backend.py': [
        ("isinstance(fmt, (list, tuple, np.ndarray))",
         "a fmt= option: one format string vs one per dataset"),
    ],
    'hypertools/plot/plot.py': [
        ("isinstance(palette, (list, tuple, np.ndarray))",
         "a palette= option: a named palette vs an explicit color list"),
        ("isinstance(fmt, (list, tuple, np.ndarray))",
         "a fmt= option: one format string vs one per dataset"),
        ("isinstance(title_color, (list, tuple, np.ndarray))",
         "a title_color= option: one color vs one per title line"),
        ("hasattr(item, 'to_numpy')",
         "guards the to_numpy call AFTER is_series_like (which also admits "
         "dw.zoo.array_like objects that have no to_numpy); the datatype "
         "decision itself is the datawrangler-based predicate"),
        ("isinstance(legend_colors, (list, tuple, np.ndarray))",
         "a legend color option: one color vs one per legend entry"),
        ("isinstance(hue, (list, tuple, np.ndarray))",
         "forecast_hue= as a per-forecast LABEL vector vs a single value, "
         "AFTER any series-like (pandas/polars Series, Index) has been "
         "normalised to an array through the shared predicate"),
        ("isinstance(labels, np.ndarray)",
         "cluster labels (a fitted model's output) unwrapped to a list"),
        ("isinstance(legend, (bool, np.bool_, list, tuple, np.ndarray, "
         "pd.Series, pd.Index))",
         "legend= as a flag vs a sequence of legend entries"),
        ("isinstance(legend, (list, tuple, np.ndarray, pd.Series, pd.Index))",
         "legend= as a sequence of legend entries"),
        ("isinstance(animate, np.ndarray)",
         "animate= per-dataset morph tags given as an array (an option)"),
        ("isinstance(_lim, (list, tuple, np.ndarray))",
         "an axis-limit option: a (lo, hi) pair vs None/scalar"),
    ],
    'hypertools/plot/trails.py': [
        ("isinstance(flag, (list, tuple, np.ndarray))",
         "a trails= flag: one value vs one per dataset"),
        ("isinstance(value, np.ndarray)",
         "a trails= flag given as an array, unwrapped to a list"),
    ],
}


def _modules():
    for root, _dirs, files in os.walk(PACKAGE_DIR):
        for name in sorted(files):
            if not name.endswith('.py'):
                continue
            path = os.path.join(root, name)
            rel = os.path.relpath(path, os.path.dirname(PACKAGE_DIR))
            rel = rel.replace(os.sep, '/')
            if any(rel.startswith(f'hypertools/{ex}')
                   or f'/{ex}' in rel for ex in EXCLUDED):
                continue
            yield rel, path


def _names_in(node):
    """Dotted names appearing anywhere in an AST node (e.g. the class tuple
    of an isinstance call): 'pd.DataFrame', 'DataFrame', ..."""
    found = set()
    for sub in ast.walk(node):
        if isinstance(sub, ast.Attribute):
            found.add(ast.unparse(sub))
        elif isinstance(sub, ast.Name):
            found.add(sub.id)
    return found


def _is_flagged_isinstance(call):
    if not (isinstance(call.func, ast.Name) and call.func.id == 'isinstance'
            and len(call.args) == 2):
        return False
    names = _names_in(call.args[1])
    return any(n in names for n in FLAGGED_TYPES + FLAGGED_BARE)


def _is_flagged_hasattr(call):
    if not (isinstance(call.func, ast.Name) and call.func.id == 'hasattr'
            and len(call.args) == 2):
        return False
    attr = call.args[1]
    return isinstance(attr, ast.Constant) and attr.value in FLAGGED_ATTRS


def find_sites():
    """[(repo-relative path, line, unparsed call)] for every flagged site."""
    sites = []
    for rel, path in _modules():
        with open(path, encoding='utf-8') as f:
            tree = ast.parse(f.read(), filename=path)
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and (
                    _is_flagged_isinstance(node) or _is_flagged_hasattr(node)):
                sites.append((rel, node.lineno, ast.unparse(node)))
    return sites


def test_scanner_sees_the_coercion_layer_and_nothing_vendored():
    # the scanner's own plumbing: it walks the real package, and the
    # exclusions are exactly the coercion layer + vendored code
    rels = [rel for rel, _ in _modules()]
    assert 'hypertools/manip/manip.py' in rels
    assert 'hypertools/plot/plot.py' in rels
    assert 'hypertools/_shared/helpers.py' not in rels
    assert 'hypertools/tools/format_data.py' not in rels
    assert not any(rel.startswith('hypertools/external/') for rel in rels)
    assert not any(rel.startswith('hypertools/_externals/') for rel in rels)


def test_scanner_flags_the_patterns_it_is_meant_to():
    # the matcher itself, on real AST (no mocks): each shape the gate is
    # documented to catch is caught, and the shapes it must NOT catch (a
    # hypertools object, a scalar, a model spec, a list/tuple container
    # check) are not
    src = '\n'.join([
        "a = isinstance(x, pd.DataFrame)",
        "b = isinstance(x, (np.ndarray, list))",
        "c = isinstance(x, pandas.Series)",
        "d = isinstance(x, DataFrame)",
        "e = hasattr(x, 'to_numpy')",
        "f = hasattr(x, 'columns')",
        "g = hasattr(x, 'values')",
        "h = isinstance(x, (list, tuple))",
        "i = isinstance(x, DataGeometry)",
        "j = isinstance(x, (int, np.integer))",
        "k = hasattr(x, 'shape')",
        "m = isinstance(x, pd.RangeIndex)",
    ])
    tree = ast.parse(src)
    flagged = sorted(
        node.targets[0].id for node in tree.body
        if _is_flagged_isinstance(node.value) or _is_flagged_hasattr(node.value))
    assert flagged == ['a', 'b', 'c', 'd', 'e', 'f', 'g']


def test_no_module_classifies_datatypes_itself():
    sites = find_sites()
    unmatched = []
    used = set()
    for rel, line, text in sites:
        entries = ALLOWLIST.get(rel, [])
        hit = [pattern for pattern, _reason in entries if pattern in text]
        if hit:
            used.update((rel, pattern) for pattern in hit)
        else:
            unmatched.append(f'{rel}:{line}: {text}')
    assert not unmatched, (
        'datatype check(s) outside the shared coercion layer -- route them '
        'through datawrangler (hypertools._shared.helpers.is_frame_dataset / '
        'is_array_dataset / is_series_like / as_pandas_dataframe, or '
        'hypertools.core.shared.as_dataframe), or add a justified '
        'ALLOWLIST entry:\n  ' + '\n  '.join(unmatched))


def test_every_allowlist_entry_still_matches_a_real_site():
    sites = find_sites()
    stale = []
    for rel, entries in ALLOWLIST.items():
        for pattern, reason in entries:
            assert reason, f'{rel}: {pattern!r} has no reason'
            if not any(r == rel and pattern in text for r, _l, text in sites):
                stale.append(f'{rel}: {pattern!r}')
    assert not stale, (
        'ALLOWLIST entries that no longer match a site (the site was '
        'converted or moved -- drop the entry):\n  ' + '\n  '.join(stale))


@pytest.mark.parametrize('rel', sorted(ALLOWLIST))
def test_allowlisted_modules_exist(rel):
    assert os.path.exists(os.path.join(os.path.dirname(PACKAGE_DIR), rel)), rel
