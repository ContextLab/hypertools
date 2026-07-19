"""The ``hypertools`` module docstring documents which import forms work
for name-shadowed submodules (final-wave audit item 1: an earlier revision
FALSELY claimed ``import hypertools.plot.backend as backend`` works).

Every import form the docstring mentions is executed here in a REAL
subprocess (fresh interpreter, no cached hypertools modules) and must
behave exactly as documented:

- ``from hypertools.plot import backend``            -> works (module)
- ``import hypertools.plot.backend as backend``      -> ImportError (the
  ``as``-binding resolves ``backend`` via attribute access on
  ``hypertools.plot``, which is the ``plot`` *function*)
- attribute access ``hypertools.plot.backend``       -> AttributeError
"""
import os
import subprocess
import sys

import hypertools

DOCSTRING = hypertools.__doc__


def _run(code):
    return subprocess.run(
        [sys.executable, '-c', code], capture_output=True, text=True,
        timeout=300, env=dict(os.environ, MPLBACKEND='Agg'))


def test_documented_working_form_works():
    snippet = 'from hypertools.plot import backend'
    # the docstring must recommend exactly this form...
    assert f'``{snippet}``' in DOCSTRING
    # ...and it must actually import the submodule in a fresh interpreter
    result = _run(
        snippet + "\n"
        "import types\n"
        "assert isinstance(backend, types.ModuleType), type(backend)\n"
        "assert hasattr(backend, 'set_interactive_backend')\n"
        "print('OK')\n")
    assert result.returncode == 0, result.stderr
    assert 'OK' in result.stdout


def test_documented_failing_as_form_raises_importerror():
    snippet = 'import hypertools.plot.backend as backend'
    # the docstring must mention this form AND document it as failing
    # with ImportError (never as a working alternative)
    assert f'``{snippet}``' in DOCSTRING
    note = DOCSTRING[DOCSTRING.index(f'``{snippet}``'):]
    assert 'raises ImportError' in note[:250], \
        'docstring must say the as-form raises ImportError'
    result = _run(snippet)
    assert result.returncode != 0, \
        'docstring documents this form as failing, but it succeeded'
    assert 'ImportError' in result.stderr, result.stderr


def test_documented_attribute_access_raises_attributeerror():
    # the docstring says ``hypertools.plot.backend`` attribute access
    # resolves against the plot function and raises AttributeError
    assert '``hypertools.plot.backend``' in DOCSTRING
    assert 'AttributeError' in DOCSTRING
    result = _run(
        "import hypertools\n"
        "assert callable(hypertools.plot)\n"
        "hypertools.plot.backend\n")
    assert result.returncode != 0
    assert 'AttributeError' in result.stderr, result.stderr
