"""Export sample animations from BOTH backends as evidence that animation
and animation export work (PR #270 review item).

Outputs:
    docs/images/v2.0-animations/<case>.gif  (small, loopable, committed)
    docs/_static/thumbnails/sphx_glr_animate_plotly_thumb.gif (gallery thumb)

Run from the repo root:
    .venv/bin/python scripts/generate_animation_evidence.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image, ImageSequence

import hypertools as hyp

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(REPO, 'docs', 'images', 'v2.0-animations')
THUMBS = os.path.join(REPO, 'docs', '_static', 'thumbnails')
SEED = 42


def main():
    os.makedirs(OUT, exist_ok=True)
    walk = np.cumsum(np.random.default_rng(SEED).standard_normal((150, 8)),
                     axis=0)
    walk2 = np.cumsum(np.random.default_rng(1).standard_normal((150, 8)),
                      axis=0)

    # 30fps renders with full-fps gifs: smooth, unhurried playback
    cases = [
        ('matplotlib_window.gif',
         dict(animate=True, duration=15, frame_rate=30)),
        ('matplotlib_spin.gif',
         dict(animate='spin', duration=15, frame_rate=30, rotations=1)),
        ('plotly_window.gif',
         dict(animate=True, duration=15, rotations=1, backend='plotly')),
        ('plotly_spin.gif',
         dict(animate='spin', duration=15, rotations=1, backend='plotly')),
    ]
    results = []
    for name, kwargs in cases:
        path = os.path.join(OUT, name)
        hyp.plot([walk, walk2], save_path=path, show=False,
                 size=[4.8, 3.6], **kwargs)
        plt.close('all')
        with Image.open(path) as im:
            n = getattr(im, 'n_frames', 1)
        results.append((name, os.path.getsize(path) // 1024, n))
        print(f'  {name}: {results[-1][1]}KB, {n} frames')

    # gallery thumbnail: downscale the plotly spin gif to 200x200
    os.makedirs(THUMBS, exist_ok=True)
    src = os.path.join(OUT, 'plotly_spin.gif')
    thumb_path = os.path.join(THUMBS, 'sphx_glr_animate_plotly_thumb.gif')
    with Image.open(src) as im:
        frames = [f.copy().convert('RGB').resize((200, 150))
                  for f in ImageSequence.Iterator(im)]
    frames[0].save(thumb_path, save_all=True, append_images=frames[1:],
                   duration=im.info.get('duration', 80), loop=0)
    print(f'  thumbnail: {os.path.getsize(thumb_path) // 1024}KB')

    index = ['# Animation export evidence (hypertools 2.0)', '',
             'Sample animations exported via `hyp.plot(..., save_path=...)`;'
             ' the extension picks the format (.gif / .png [APNG] / .mp4).',
             '', '| case | file | frames |', '|-|-|-|']
    for name, kb, n in results:
        index.append(f'| {name.rsplit(".", 1)[0]} | {name} ({kb}KB) | {n} |')
    with open(os.path.join(OUT, 'INDEX.md'), 'w') as f:
        f.write('\n'.join(index) + '\n')
    print('INDEX.md written')


if __name__ == '__main__':
    main()
