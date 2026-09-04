"""Contract 9: `hypertools/predict/` never imports from `hypertools/plot/`.

Forecasting is not a rendering concern, and the 1.0 package split put the
machinery both halves need under `core/` (see
`hypertools/core/hierarchy.py`'s module docstring, which states the same
rule from the other side). The rule had been checked only by throwaway
`grep` while the hierarchy work was in flight; the plan says "a test asserts
it", so this is that test.

Parsed rather than grepped so the check sees IMPORTS specifically -- a
`grep` for 'plot' matches docstrings, kwarg names and `hyp.plot` prose,
which is why the throwaway version needed a human to read its output.
Relative spellings count: `from ..plot import x` and `from .. import plot`
are `ImportFrom` nodes whose `module` says nothing about `plot` at all.
"""
import ast
from pathlib import Path

import pytest

import hypertools

PREDICT_DIR = Path(hypertools.__file__).parent / 'predict'
FORBIDDEN = 'hypertools.plot'


def _module_package(path):
    """The package a file's relative imports resolve against.

    `hypertools/predict/arima.py` -> `hypertools.predict`; a package
    `__init__.py` is its own package, not its parent's.
    """
    parts = list(path.relative_to(Path(hypertools.__file__).parent).parts)
    if path.name == '__init__.py':
        parts = parts[:-1]
    else:
        parts[-1] = parts[-1][:-len('.py')]
        parts = parts[:-1]
    return '.'.join(['hypertools'] + parts)


def _resolve(package, level, module):
    """The absolute module name an import refers to (level 0 = absolute)."""
    if level == 0:
        return module
    base = package.split('.')
    # level 1 is "this package", so each level beyond it strips one parent
    base = base[:len(base) - (level - 1)] if level > 1 else base
    return '.'.join(base + ([module] if module else []))


def imported_modules(source, path):
    """Every absolute module name `source` imports, relative ones resolved."""
    package = _module_package(path)
    found = []
    for node in ast.walk(ast.parse(source, filename=str(path))):
        if isinstance(node, ast.Import):
            found.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            resolved = _resolve(package, node.level, node.module)
            found.append(resolved)
            # `from ..plot import x` names the module in `module`; `from ..
            # import plot` names it in `names`, so both spellings have to be
            # reconstructed or the second one walks straight through.
            found.extend(f'{resolved}.{alias.name}' for alias in node.names)
    return found


def offenders(source, path):
    return [name for name in imported_modules(source, path)
            if name == FORBIDDEN or name.startswith(FORBIDDEN + '.')]


PREDICT_FILES = sorted(PREDICT_DIR.rglob('*.py'))


def test_the_predict_package_has_files_to_check():
    """Guard against the check silently passing on an empty file list."""
    assert len(PREDICT_FILES) >= 5
    assert (PREDICT_DIR / 'predict.py') in PREDICT_FILES


@pytest.mark.parametrize('path', PREDICT_FILES,
                         ids=[p.name for p in PREDICT_FILES])
def test_predict_module_does_not_import_plot(path):
    found = offenders(path.read_text(encoding='utf-8'), path)
    assert found == [], (
        f'{path} imports {found}: hypertools/predict/ must not depend on '
        'hypertools/plot/ (Contract 9). Shared machinery belongs under '
        'hypertools/core/.')


@pytest.mark.parametrize('spelling', [
    'from hypertools.plot import plot',
    'from hypertools.plot.hierarchy import build_hierarchy_traces',
    'import hypertools.plot',
    'import hypertools.plot.hierarchy as h',
    'from ..plot import plot',
    'from ..plot.hierarchy import build_hierarchy_traces',
    'from .. import plot',
])
def test_the_check_catches_every_spelling_of_the_forbidden_import(spelling):
    """The discrimination proof: each of these, dropped into a predict
    module, must be reported. Relative forms are the ones a naive check
    misses, so they are stated one by one rather than as a single sample."""
    path = PREDICT_DIR / 'predict.py'
    assert offenders(spelling + '\n', path), f'{spelling!r} walked through'


def test_an_unrelated_import_is_not_reported():
    """...and the check must not fire on the imports predict really uses,
    or it would be satisfied by anything."""
    path = PREDICT_DIR / 'predict.py'
    source = ('import numpy as np\n'
              'from ..core.hierarchy import group_rows_for_forecast\n'
              'from .common import fit_predict\n')
    assert offenders(source, path) == []
