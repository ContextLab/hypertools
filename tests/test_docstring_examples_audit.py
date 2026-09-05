# -*- coding: utf-8 -*-
"""Release-1.0 audit (C5-docstring-residuals, D12-013): the primary public
API functions must carry RUNNABLE Examples sections, and those examples
must execute cleanly as real doctests -- no mocks, real models on real
(seeded) data.

Each module below gained an ``Examples`` block in the 2026-07 release
audit; this test runs them with the stdlib doctest runner so any drift
between the documented outputs and the actual code fails CI. Modules
whose examples require network access (e.g. ``hypertools.io.load``'s
hosted-dataset examples) are exercised separately and deliberately not
listed here.
"""
import doctest
import importlib

import pytest

# offline, deterministic Examples added/verified in the 2026-07 audit
MODULES_WITH_EXAMPLES = [
    'hypertools.tools.analyze',
    'hypertools.reduce.reduce',
    'hypertools.reduce.describe',
    'hypertools.align.align',
    'hypertools.cluster.cluster',
    'hypertools.predict.predict',
    'hypertools.impute.impute',
    'hypertools.manip.manip',
    'hypertools.tools.normalize',
    'hypertools.tools.text_windows',
    'hypertools.tools.damage',
    'hypertools.tools.stack',
]


@pytest.mark.parametrize('module_name', MODULES_WITH_EXAMPLES)
def test_docstring_examples_run_clean(module_name):
    """Every listed module has at least one doctest and all of them pass."""
    mod = importlib.import_module(module_name)
    results = doctest.testmod(mod, verbose=False)
    assert results.attempted > 0, (
        f'{module_name} has no runnable doctests -- its public function is '
        'expected to carry an Examples section (D12-013)')
    assert results.failed == 0, (
        f'{module_name}: {results.failed} of {results.attempted} docstring '
        'examples failed; run pytest --doctest-modules on the module for '
        'details')
