"""Regenerate `docs/images/v1.0-seven-features/dot_size_parity.png` (R2
fix evidence): the SAME point cloud rendered by hypertools' matplotlib and
plotly backends, same `markersize`, side by side -- demonstrates plotly's
dots now match matplotlib's size (maintainer request: "dots should be
smaller in plotly (to match matplotlib)").

Run from the repo root:
    .venv/bin/python scripts/generate_marker_parity_evidence.py
"""
import os

import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont

import hypertools as hyp

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(REPO, 'docs', 'images', 'v1.0-seven-features')


def main():
    os.makedirs(OUT, exist_ok=True)
    rng = np.random.default_rng(0)
    cloud = rng.standard_normal((60, 3))

    fig_mpl = hyp.plot([cloud], '.', markersize=10, show=False)
    fig_mpl.axes[0].set_title('matplotlib', fontsize=14)
    mpl_path = '/tmp/_parity_mpl.png'
    fig_mpl.savefig(mpl_path, dpi=100)
    plt.close(fig_mpl)

    fig_plotly = hyp.plot([cloud], '.', markersize=10, backend='plotly',
                          show=False)
    fig_plotly.update_layout(title=dict(text='plotly'))
    plotly_path = '/tmp/_parity_plotly.png'
    fig_plotly.write_image(plotly_path)

    im_mpl = Image.open(mpl_path).convert('RGB')
    im_plotly = Image.open(plotly_path).convert('RGB').resize(im_mpl.size)
    pad_top = 40
    grid = Image.new('RGB', (im_mpl.width * 2, im_mpl.height + pad_top),
                     'white')
    grid.paste(im_mpl, (0, pad_top))
    grid.paste(im_plotly, (im_mpl.width, pad_top))
    draw = ImageDraw.Draw(grid)
    try:
        font = ImageFont.truetype(
            '/System/Library/Fonts/Supplemental/Arial.ttf', 20)
    except OSError:
        font = ImageFont.load_default()
    draw.text((20, 10),
             "dot_size_parity: same cloud, same markersize=10, '.' fmt "
             "(R2 fix)", fill='black', font=font)
    out_path = os.path.join(OUT, 'dot_size_parity.png')
    grid.save(out_path)
    print(f'wrote {out_path} {grid.size}')

    os.remove(mpl_path)
    os.remove(plotly_path)


if __name__ == '__main__':
    main()
