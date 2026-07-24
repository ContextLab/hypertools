"""Regenerate the `density=True` (3D) evidence images (maintainer request,
R2 fix -- "plotly dot sizing and transparency are different from the
matplotlib versions... we need more transparency for volumetric shading").

Drives the REAL `hyp.plot(..., density=True, backend=...)` code path for
both backends on two scenes:
  - the standard "close" 2-blob scene (auto-boost ~1, near no-op)
  - a widely separated (`sep=10`) 2-blob scene (auto-boost engaged)
so the retuned `go.Volume` opacity/opacityscale constants
(`hypertools.plot.density.resolve_plotly_volume_params`) can be checked
against matplotlib's iso-surface rendering in both regimes.

Run from the repo root:
    .venv/bin/python scripts/generate_density_evidence.py

Outputs:
    docs/images/v1.0-seven-features/density_3d_mpl.png
    docs/images/v1.0-seven-features/density_3d_plotly.png
    docs/images/v1.0-seven-features/density_3d_sep_mpl.png
    docs/images/v1.0-seven-features/density_3d_sep_plotly.png
"""
import os

import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import hypertools as hyp

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(REPO, 'docs', 'images', 'v1.0-seven-features')


def _blob_3d(n=150, seed=0, center=(0.0, 0.0, 0.0), scale=1.0):
    rng = np.random.default_rng(seed)
    return rng.normal(size=(n, 3)) * scale + np.asarray(center)


def _two_datasets_3d(sep=3.0):
    return [_blob_3d(seed=0, center=(-sep, 0, 0)),
            _blob_3d(seed=1, center=(sep, 0, 0))]


def generate_mpl(data, title, out_path):
    fig = hyp.plot(data, '.', density=True, show=False)
    ax = fig.axes[0]
    ax.set_title(title)
    fig.savefig(out_path, dpi=100)
    plt.close(fig)
    print(f'wrote {out_path}')


def generate_plotly(data, title, out_path):
    fig = hyp.plot(data, '.', density=True, backend='plotly', show=False)
    fig.update_layout(title=dict(text=title))
    fig.write_image(out_path)
    print(f'wrote {out_path}')


def main():
    os.makedirs(OUT, exist_ok=True)

    close_data = _two_datasets_3d(sep=3.0)
    generate_mpl(close_data, 'density=True (3D, matplotlib)',
                os.path.join(OUT, 'density_3d_mpl.png'))
    generate_plotly(close_data, 'density=True (3D, plotly)',
                   os.path.join(OUT, 'density_3d_plotly.png'))

    sep_data = _two_datasets_3d(sep=10.0)
    generate_mpl(sep_data, 'density=True, sep=10 (3D, matplotlib)',
                os.path.join(OUT, 'density_3d_sep_mpl.png'))
    generate_plotly(sep_data, 'density=True, sep=10 (3D, plotly)',
                   os.path.join(OUT, 'density_3d_sep_plotly.png'))


if __name__ == '__main__':
    main()
