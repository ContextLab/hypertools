"""Screenshot harness for hypertools visual verification.

Every plot-producing use case gets captured as a PNG under
tests/screenshots/<tag>/<function>/<case>.png so behavior can be visually
verified (and later diffed across versions/backends).

Usage:
    from screenshot_harness import capture

    capture('baseline_v0.8.2', 'plot', 'static_3d_single_array',
            lambda: hyp.plot(data, show=False))

Design notes:
- Uses the Agg backend so it works headless (CI, SSH sessions).
- Closes all matplotlib figures after each capture to avoid the
  figure-accumulation problems reported in GH issue #264.
- Returns the capture record (path, success flag, error if any) so callers
  can assemble a summary report instead of dying on the first failure.
"""

import os
import traceback

import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCREENSHOT_ROOT = os.path.join(REPO_ROOT, 'tests', 'screenshots')


def capture(tag, function, case, plot_fn, dpi=100):
    """Run plot_fn(), save the resulting figure(s), close everything.

    Parameters
    ----------
    tag : str
        Grouping label, e.g. 'baseline_v0.8.2' or 'dev-2.0_matplotlib'.
    function : str
        Public API function being exercised, e.g. 'plot', 'describe'.
    case : str
        Short slug describing the use case, e.g. 'static_3d_list_of_arrays'.
    plot_fn : callable
        Zero-argument callable that produces the plot. Should pass
        show=False to hypertools functions where supported.
    dpi : int
        Resolution for the saved PNG (kept low to keep files small).

    Returns
    -------
    dict with keys: tag, function, case, path, ok, error
    """
    out_dir = os.path.join(SCREENSHOT_ROOT, tag, function)
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, case + '.png')
    record = {'tag': tag, 'function': function, 'case': case,
              'path': path, 'ok': False, 'error': None}
    try:
        result = plot_fn()
        # plotly figures (returned directly or as geo.fig) export via kaleido
        plotly_fig = _extract_plotly_fig(result)
        if plotly_fig is not None:
            plotly_fig.write_image(path, width=700, height=500)
            record['ok'] = True
            record['result'] = result
            return record
        fignums = plt.get_fignums()
        if not fignums:
            raise RuntimeError('plot_fn produced no matplotlib figures')
        # Save the most recently created figure; extra figures get suffixes.
        for i, num in enumerate(fignums):
            fig = plt.figure(num)
            fig_path = path if i == len(fignums) - 1 else \
                path.replace('.png', f'_fig{i}.png')
            fig.savefig(fig_path, dpi=dpi, bbox_inches='tight')
        record['ok'] = True
        record['result'] = result
    except Exception as e:  # noqa: BLE001 - harness must survive any failure
        record['error'] = f'{type(e).__name__}: {e}'
        record['traceback'] = traceback.format_exc()
    finally:
        plt.close('all')
    return record


def _extract_plotly_fig(result):
    fig = getattr(result, 'fig', result)
    if type(fig).__module__.startswith('plotly'):
        return fig
    return None


def summarize(records):
    """Print a pass/fail table for a list of capture records; return failures."""
    failures = [r for r in records if not r['ok']]
    print(f"\n{'=' * 70}")
    print(f"Screenshot summary: {len(records) - len(failures)}/{len(records)} succeeded")
    print('=' * 70)
    for r in records:
        status = 'ok ' if r['ok'] else 'FAIL'
        print(f"  [{status}] {r['function']}/{r['case']}"
              + ('' if r['ok'] else f"  -> {r['error']}"))
    return failures
