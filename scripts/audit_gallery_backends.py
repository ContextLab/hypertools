"""Run EVERY gallery example under BOTH backends (matplotlib + plotly),
catalog pass/fail, and export a PNG per run for visual verification.

Usage (from repo root):
    .venv/bin/python scripts/audit_gallery_backends.py [out_dir]

Each example runs in a subprocess with matplotlib's Agg backend and cwd set
to a temp dir (so save_* examples write there). For the plotly pass,
hyp.plot is wrapped to inject backend='plotly' whenever the example didn't
specify a backend itself; show is forced off for both passes.
"""

import os
import subprocess
import sys
import tempfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXAMPLES = os.path.join(REPO, 'examples')

RUNNER = r'''
import os, sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

example, backend, out_png = sys.argv[1], sys.argv[2], sys.argv[3]

import hypertools as hyp
# hypertools/__init__ does `from .plot.plot import plot`, which shadows the
# `plot` SUBPACKAGE attribute on the package with the plot FUNCTION -- so
# `import hypertools.plot.plot as X` fails on attribute traversal. The
# module itself is in sys.modules after `import hypertools`; take it from
# there and patch the function attribute on its defining module (covers
# DataGeometry.plot's late `from .plot.plot import plot`).
_plotmod = sys.modules['hypertools.plot.plot']

_geos = []
_orig_plot = _plotmod.plot

def _wrapped_plot(*args, **kwargs):
    if backend == 'plotly':
        kwargs.setdefault('backend', 'plotly')
    kwargs['show'] = False
    geo = _orig_plot(*args, **kwargs)
    _geos.append(geo)
    return geo

# patch in the defining module (covers DataGeometry.plot's late import
# `from .plot.plot import plot`) and the top-level alias examples call
_plotmod.plot = _wrapped_plot
hyp.plot = _wrapped_plot

src = open(example).read()
ns = {'__name__': '__main__', '__file__': example}
exec(compile(src, example, 'exec'), ns)

# export the last plot for visual verification
if _geos:
    geo = _geos[-1]
    fig = getattr(geo, 'fig', None)
    if fig is None:
        pass
    elif type(fig).__module__.startswith('plotly'):
        fig.write_image(out_png, width=700, height=500)
    else:
        fig.savefig(out_png, dpi=72, facecolor='white')
print('OK')
'''


def _run_one(args):
    name, backend, out_dir = args
    key = f'{name[:-3]}__{backend}'
    png = os.path.join(out_dir, key + '.png')
    with tempfile.TemporaryDirectory() as tmp:
        try:
            proc = subprocess.run(
                [sys.executable, '-c', RUNNER,
                 os.path.join(EXAMPLES, name), backend, png],
                capture_output=True, text=True, timeout=900, cwd=tmp)
            ok = proc.returncode == 0 and 'OK' in proc.stdout
            err = ''
            if not ok:
                tail = (proc.stderr or proc.stdout).strip().splitlines()
                err = tail[-1] if tail else 'unknown'
        except subprocess.TimeoutExpired:
            ok, err = False, 'timeout (900s)'
    return key, ok, err


def main():
    from concurrent.futures import ThreadPoolExecutor
    out_dir = sys.argv[1] if len(sys.argv) > 1 else '/tmp/gallery_audit'
    os.makedirs(out_dir, exist_ok=True)
    examples = sorted(f for f in os.listdir(EXAMPLES) if f.endswith('.py'))
    jobs = [(n, b, out_dir) for n in examples
            for b in ('matplotlib', 'plotly')]
    results = {}
    with ThreadPoolExecutor(max_workers=6) as pool:
        for key, ok, err in pool.map(_run_one, jobs):
            results[key] = (ok, err)
            print(f'{"PASS" if ok else "FAIL"}  {key}'
                  + (f'  -- {err}' if err else ''), flush=True)

    n_fail = sum(1 for ok, _ in results.values() if not ok)
    with open(os.path.join(out_dir, 'REPORT.md'), 'w') as f:
        f.write('# Gallery dual-backend audit\n\n'
                '| example | backend | status | error |\n|-|-|-|-|\n')
        for key, (ok, err) in sorted(results.items()):
            ex, be = key.rsplit('__', 1)
            f.write(f'| {ex} | {be} | {"PASS" if ok else "FAIL"} '
                    f'| {err} |\n')
    print(f'\n{len(results)} runs, {n_fail} failures. '
          f'Report: {out_dir}/REPORT.md')
    sys.exit(1 if n_fail else 0)


if __name__ == '__main__':
    main()
