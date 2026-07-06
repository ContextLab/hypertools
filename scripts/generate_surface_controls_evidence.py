"""Regenerate the surface= coloring/lighting/shading control evidence images
(maintainer request, R3 fix -- "for hulls, confirm we can control coloring,
lighting, and shading via parameters passed to plot").

Drives the REAL `hyp.plot(..., surface=..., backend=...)` code path for both
backends, six panels each: default, `color='crimson'`, a glossy/high-specular
look, a flat/matte high-ambient look, a translucent (`alpha=0.3`) look, and a
custom `lightdir`. Confirms every accepted `surface`/`surface['lighting']`
knob visibly, distinctly changes the render on BOTH backends.

Run from the repo root:
    MPLBACKEND=Agg .venv/bin/python scripts/generate_surface_controls_evidence.py

Outputs:
    docs/images/v1.0-seven-features/surface_controls_mpl.png
    docs/images/v1.0-seven-features/surface_controls_plotly.png
"""
import os
import tempfile

import numpy as np
from PIL import Image, ImageDraw, ImageFont

import hypertools as hyp

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(REPO, 'docs', 'images', 'v1.0-seven-features')

PANELS = [
    ('default', {}),
    ("color='crimson'", {'color': 'crimson'}),
    ('glossy: specular=0.9, shininess=128',
     {'lighting': {'specular': 0.9, 'shininess': 128}}),
    ('matte: ambient=0.9, diffuse=0.1, specular=0',
     {'lighting': {'ambient': 0.9, 'diffuse': 0.1, 'specular': 0.0}}),
    ('alpha=0.3', {'alpha': 0.3}),
    ('lightdir=(-1, -1, 1)', {'lighting': {'lightdir': (-1, -1, 1)}}),
]


def _blob_3d(n=150, seed=0):
    rng = np.random.default_rng(seed)
    return rng.normal(size=(n, 3))


def _render_panels(backend, tmpdir):
    data = [_blob_3d()]
    paths = []
    for idx, (_, spec) in enumerate(PANELS):
        surface_kw = dict(spec) if spec else True
        path = os.path.join(tmpdir, f'panel_{backend}_{idx}.png')
        hyp.plot(data, '.', surface=surface_kw, backend=backend, show=False,
                 save_path=path)
        paths.append(path)
    return paths


def _make_grid(paths, out_path, title):
    imgs = [Image.open(p).convert('RGB') for p in paths]
    w, h = imgs[0].size
    pad = 40
    grid = Image.new('RGB', (3 * w, 2 * (h + pad) + pad), 'white')
    draw = ImageDraw.Draw(grid)
    try:
        font = ImageFont.load_default(size=18)
    except TypeError:
        font = ImageFont.load_default()
    draw.text((10, 5), title, fill='black', font=font)
    for idx, (img, (label, _)) in enumerate(zip(imgs, PANELS)):
        row, col = idx // 3, idx % 3
        x, y = col * w, pad + row * (h + pad)
        grid.paste(img, (x, y))
        draw.text((x + 10, y + h + 5), label, fill='black', font=font)
    grid.save(out_path)
    print(f'wrote {out_path}')


def main():
    os.makedirs(OUT, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmpdir:
        for backend in ('matplotlib', 'plotly'):
            paths = _render_panels(backend, tmpdir)
            out_name = ('surface_controls_mpl.png' if backend == 'matplotlib'
                        else 'surface_controls_plotly.png')
            _make_grid(paths, os.path.join(OUT, out_name),
                      f'{backend} surface= coloring/lighting/shading controls')


if __name__ == '__main__':
    main()
