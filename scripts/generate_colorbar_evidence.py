"""Regenerate the discrete-colorbar evidence images (GH #100 follow-up --
"the entries in legend and discrete colorbar entries should be in the same
order").

Drives the REAL `hyp.plot(..., colorbar=True, legend=[...])` code path for
both backends on 3 discrete groups, so the legend ("group A, group B, group
C" top-to-bottom) and the VERTICAL discrete colorbar can be visually
compared -- after the fix, both must read in the SAME top-to-bottom order
(previously the colorbar read bottom-up, the reverse of the legend).

Run from the repo root:
    .venv/bin/python scripts/generate_colorbar_evidence.py

Outputs:
    docs/images/v1.0-seven-features/colorbar_discrete_mpl.png
    docs/images/v1.0-seven-features/colorbar_discrete_plotly.png
"""
import os

import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import hypertools as hyp

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(REPO, 'docs', 'images', 'v1.0-seven-features')

LEGEND = ['group A', 'group B', 'group C']


def _three_groups(n=60):
    rng = np.random.default_rng(0)
    offsets = (-5.0, 0.0, 5.0)
    return [rng.standard_normal((n, 3)) * 0.4 + np.array([off, 0, 0])
            for off in offsets]


def generate_mpl(data, title, out_path):
    fig = hyp.plot(data, legend=LEGEND, colorbar=True, show=False)
    ax = fig.axes[0]
    ax.set_title(title)
    fig.savefig(out_path, dpi=100, bbox_inches='tight')
    plt.close(fig)
    print(f'wrote {out_path}')


def generate_plotly(data, title, out_path):
    fig = hyp.plot(data, legend=LEGEND, colorbar=True, backend='plotly',
                   show=False)
    # hyp.plot() was called without title=, so its layout kept the no-title
    # top margin (t=10) -- widen it when adding the title here, or the title
    # renders clipped off the top edge of the exported image.
    fig.update_layout(title=dict(text=title, y=0.97, yanchor='top'),
                      margin=dict(t=45))
    fig.write_image(out_path)
    print(f'wrote {out_path}')


def main():
    os.makedirs(OUT, exist_ok=True)
    data = _three_groups()
    generate_mpl(data, 'Discrete groups + colorbar + legend (matplotlib)',
                os.path.join(OUT, 'colorbar_discrete_mpl.png'))
    generate_plotly(data, 'Discrete groups + colorbar + legend (plotly)',
                   os.path.join(OUT, 'colorbar_discrete_plotly.png'))


if __name__ == '__main__':
    main()
