"""Inside IPython, a matplotlib backend that cannot be switched to raises the
documented ``HypertoolsBackendError`` -- not the GUI toolkit's own error.

``set_interactive_backend`` switches via the ``%matplotlib`` magic when it
runs inside IPython/Jupyter, and the magic imports the toolkit itself. Until
1.1 its ``ModuleNotFoundError`` (no ``_tkinter``, no ``gi``) or ``TclError``
(no display) escaped unwrapped, while the plain-script path already raised
``HypertoolsBackendError`` (measured 2026-09-04 with the 1.1 feature tour).
"""

import textwrap

import pytest

_CELLS = [
    """
    import hypertools as hyp
    try:
        import gi                    # the GTK3Agg toolkit
        have_gi = True
    except Exception:
        have_gi = False
    print('have_gi', have_gi)
    """,
    """
    try:
        with hyp.set_interactive_backend('GTK3Agg'):
            pass
        print('switched to GTK3Agg')
    except hyp.HypertoolsBackendError as e:
        print('HypertoolsBackendError:', str(e)[:120])
    except Exception as e:
        print('UNWRAPPED', type(e).__name__, str(e)[:120])
    """,
    """
    import matplotlib
    print('backend after', matplotlib.get_backend())
    """,
]


def test_unswitchable_backend_raises_hypertools_backend_error_in_a_kernel(tmp_path):
    nbformat = pytest.importorskip('nbformat')
    nbclient = pytest.importorskip('nbclient')
    pytest.importorskip('ipykernel')
    nb = nbformat.v4.new_notebook()
    for src in _CELLS:
        nb.cells.append(nbformat.v4.new_code_cell(textwrap.dedent(src).strip()))
    nbclient.NotebookClient(nb, timeout=300, kernel_name='python3', allow_errors=True,
                            resources={'metadata': {'path': str(tmp_path)}}).execute()
    errors = [(o['ename'], o['evalue']) for c in nb.cells for o in c.get('outputs', [])
              if o['output_type'] == 'error']
    assert errors == [], errors
    text = '\n'.join(''.join(o.get('text', '')) for c in nb.cells for o in c.get('outputs', [])
                     if o['output_type'] == 'stream')
    if 'have_gi True' in text:
        pytest.skip('GTK3 is importable in this kernel, so the switch may succeed')
    assert 'UNWRAPPED' not in text, text
    assert 'HypertoolsBackendError:' in text, text
    assert 'backend after' in text
