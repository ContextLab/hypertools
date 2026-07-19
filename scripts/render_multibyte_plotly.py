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


def main():
    legend_json, title, out_png = sys.argv[1], sys.argv[2], sys.argv[3]
    labels_json = sys.argv[4] if len(sys.argv) > 4 else 'null'
    legend = json.loads(legend_json)
    labels = json.loads(labels_json)
    data = [np.random.default_rng(i).standard_normal((15, 3))
            for i in range(len(legend))]
    fig = hyp.plot(data, legend=legend, title=title or None, labels=labels,
                   backend='plotly', show=False)
    fig.write_image(out_png, width=640, height=480)


if __name__ == '__main__':
    main()
