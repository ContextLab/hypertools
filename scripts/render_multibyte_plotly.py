#!/usr/bin/env python
"""Committed helper (GH #205, F2/F3): renders a small hypertools plotly plot
(legend + optional title + optional point labels, Japanese or ASCII) and
writes it to a static PNG via kaleido.

Run as a SUBPROCESS (not imported) by tests/test_multibyte.py's plotly
pixel-level anti-tofu checks, specifically so a kaleido/Chromium hang can
be killed by `subprocess.run(..., timeout=...)` from the parent test
process, rather than wedging the whole pytest run in place -- an
in-process thread timeout can't reliably interrupt a stuck Chromium
subprocess call (this is also why the repo's 6 known deadlock-prone
plotly export tests, in test_animation_export.py and test_round3.py, are
deselected rather than timeout-guarded).

Usage: python render_multibyte_plotly.py <legend_json> <title> <out_png>
       [<labels_json>]

`legend_json` is a JSON list of strings (one per dataset); `title` is a
plain string (pass '' for no title); `labels_json` (F3, optional) is a
JSON list of per-dataset label lists (each inner list's length must match
that dataset's 15 points -- see `hypertools.plot.plotly_backend.
_build_point_annotations`), or 'null'/omitted for no point labels.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import hypertools as hyp  # noqa: E402

#: Exit code meaning "the browser could not be driven HERE" -- Chrome absent,
#: failed to launch, or closed mid-render. The calling test SKIPS on this and
#: fails on every other non-zero exit, so an environment without a working
#: Chrome does not masquerade as a hypertools rendering defect (and, just as
#: importantly, a hypertools defect cannot hide behind a blanket skip).
NO_BROWSER_EXIT = 3

#: Override the browser executable. `tests/test_multibyte.py` points this at a
#: real non-browser binary to prove the NO_BROWSER_EXIT path fires, with a real
#: subprocess and a real `BrowserFailedError` rather than a stubbed one.
BROWSER_PATH_ENV = 'HYPERTOOLS_RENDER_BROWSER_PATH'


def _browser_lifecycle_errors():
    """The exception types that mean "no usable browser", from the libraries
    that define them -- never a hand-written message match.

    `plotly.io._kaleido` catches `ChromeNotFoundError` and re-raises it as a
    plain `RuntimeError` carrying `PLOTLY_GET_CHROME_ERROR_MSG`, so that one
    cannot be caught by type through `fig.write_image` and is matched on
    plotly's own constant instead.
    """
    from kaleido.errors import (BrowserClosedError, BrowserFailedError,
                                ChromeNotFoundError)
    from plotly.io._kaleido import PLOTLY_GET_CHROME_ERROR_MSG
    return ((BrowserClosedError, BrowserFailedError, ChromeNotFoundError),
            PLOTLY_GET_CHROME_ERROR_MSG)


def _write_image(fig, out_png):
    override = os.environ.get(BROWSER_PATH_ENV)
    if not override:
        fig.write_image(out_png, width=640, height=480)
        return
    import kaleido
    kaleido.write_fig_sync(fig, out_png,
                           opts={'width': 640, 'height': 480},
                           kopts={'path': override, 'timeout': 30})


def main():
    legend_json, title, out_png = sys.argv[1], sys.argv[2], sys.argv[3]
    labels_json = sys.argv[4] if len(sys.argv) > 4 else 'null'
    legend = json.loads(legend_json)
    labels = json.loads(labels_json)
    data = [np.random.default_rng(i).standard_normal((15, 3))
            for i in range(len(legend))]
    fig = hyp.plot(data, legend=legend, title=title or None, labels=labels,
                   backend='plotly', show=False)
    browser_errors, no_chrome_msg = _browser_lifecycle_errors()
    try:
        _write_image(fig, out_png)
    except browser_errors as err:
        print(f'NO_BROWSER: {type(err).__name__}: {err}', file=sys.stderr)
        sys.exit(NO_BROWSER_EXIT)
    except RuntimeError as err:
        if no_chrome_msg.strip() not in str(err):
            raise            # a real failure: let the traceback through
        print(f'NO_BROWSER: {type(err).__name__}: {err}', file=sys.stderr)
        sys.exit(NO_BROWSER_EXIT)


if __name__ == '__main__':
    main()
