# -*- coding: utf-8 -*-
"""Headless-backend safety (2026-07 release review, blocker #2).

An explicitly selected non-interactive matplotlib backend (MPLBACKEND=Agg)
must be respected: animated/interactive plotting must NOT switch to a GUI
backend (MacOSX/Tk/Qt/GTK/Wx) merely because ``animate=True``. On a
genuinely headless machine, loading a GUI backend is an uncatchable native
abort, so these run in REAL subprocesses (no mocks) and assert that no GUI
toolkit is ever imported.
"""
import subprocess
import sys
import textwrap



_GUI_MODULE_MARKERS = (
    '_macosx',           # matplotlib MacOSX (cocoa)
    'matplotlib.backends.backend_macosx',
    'tkinter', '_tkinter',
    'PyQt5', 'PyQt6', 'PySide2', 'PySide6',
    'gi.repository.Gtk',
    'wx',
)


def _run_headless(body):
    """Run `body` in a subprocess with MPLBACKEND=Agg and return the
    reported (backend, loaded_gui_modules, error)."""
    script = textwrap.dedent('''
        import os, sys, json
        os.environ["MPLBACKEND"] = "Agg"
        import numpy as np
        import hypertools as hyp
        import matplotlib
        err = ""
        try:
        {body}
        except BaseException as e:  # BaseException: catch would-be aborts too
            err = f"{{type(e).__name__}}: {{e}}"
        gui = [m for m in {markers!r} if m in sys.modules]
        print("RESULT" + json.dumps({{
            "backend": matplotlib.get_backend().lower(),
            "gui": gui,
            "err": err,
        }}))
    ''').format(
        body=textwrap.indent(textwrap.dedent(body), ' ' * 12),
        markers=_GUI_MODULE_MARKERS,
    )
    proc = subprocess.run([sys.executable, '-c', script],
                          capture_output=True, text=True, timeout=300,
                          env={**_clean_env()})
    assert proc.returncode == 0, (
        f'subprocess crashed (rc={proc.returncode}) -- a native GUI abort '
        f'looks exactly like this.\nstdout:\n{proc.stdout}\n'
        f'stderr:\n{proc.stderr}')
    line = [ln for ln in proc.stdout.splitlines() if ln.startswith('RESULT')]
    assert line, f'no RESULT line.\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}'
    import json
    return json.loads(line[-1][len('RESULT'):])


def _clean_env():
    import os
    env = dict(os.environ)
    env['MPLBACKEND'] = 'Agg'
    env.pop('HYPERTOOLS_BACKEND', None)  # don't let a stray override interfere
    return env


def test_import_under_mplbackend_agg_stays_agg():
    r = _run_headless('pass')
    assert r['backend'] == 'agg', r
    assert r['gui'] == [], f'GUI backend loaded on plain import: {r["gui"]}'


def test_animate_to_file_loads_no_gui_backend(tmp_path):
    out = tmp_path / 'a.gif'
    r = _run_headless(f'''
        x = np.cumsum(np.random.default_rng(0).standard_normal((60, 3)), 0)
        hyp.plot(x, animate=True, duration=1, frame_rate=5,
                 save_path={str(out)!r}, show=False)
    ''')
    assert r['err'] == '', f'animate-to-file raised: {r["err"]}'
    assert r['backend'] == 'agg', f'backend switched away from Agg: {r}'
    assert r['gui'] == [], f'GUI backend loaded for a file export: {r["gui"]}'
    assert out.exists() and out.stat().st_size > 0


def test_animate_2d_to_file_loads_no_gui_backend(tmp_path):
    # the reviewer's exact repro: a 2-D animated figure under MPLBACKEND=Agg
    out = tmp_path / 'a2d.gif'
    r = _run_headless(f'''
        x = np.cumsum(np.random.default_rng(1).standard_normal((60, 2)), 0)
        hyp.plot(x, animate=True, duration=1, frame_rate=5,
                 save_path={str(out)!r}, show=False)
    ''')
    assert r['err'] == '', f'2-D animate raised: {r["err"]}'
    assert r['backend'] == 'agg', r
    assert r['gui'] == [], f'GUI backend loaded for 2-D animation: {r["gui"]}'
    assert out.exists() and out.stat().st_size > 0


def test_animate_show_false_no_savepath_loads_no_gui_backend():
    r = _run_headless('''
        x = np.cumsum(np.random.default_rng(2).standard_normal((40, 3)), 0)
        hyp.plot(x, animate=True, duration=1, frame_rate=5, show=False)
    ''')
    assert r['err'] == '', f'animate show=False raised: {r["err"]}'
    assert r['backend'] == 'agg', r
    assert r['gui'] == [], f'GUI backend loaded with show=False: {r["gui"]}'
