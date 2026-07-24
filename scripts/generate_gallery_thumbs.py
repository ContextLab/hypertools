"""Regenerate the animated gallery thumbnails with the CORRECT aspect
ratio: previous thumbs were squashed into 200x200 squares from 4:3
sources. Each thumb is now the source animation scaled to fit 200x150
(4:3 preserved) and letterboxed onto a 200x200 white canvas so the
gallery grid stays uniform.

Sources: the sphinx-gallery-rendered mp4s (docs/auto_examples/images/)
for matplotlib animation examples, and the plotly evidence gif for the
plotly example. Run after building the docs:

    .venv/bin/python scripts/generate_gallery_thumbs.py
"""

import os
import subprocess
import tempfile

from PIL import Image, ImageSequence

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MP4_DIR = os.path.join(REPO, 'docs', 'auto_examples', 'images')
THUMBS = os.path.join(REPO, 'docs', '_static', 'thumbnails')
PLOTLY_GIF = os.path.join(REPO, 'docs', 'images', 'v1.0-animations',
                          'plotly_spin.gif')

MPL_ANIMS = ['animate', 'animate_MDS', 'animate_spin', 'chemtrails',
             'precog', 'save_movie']
N_FRAMES = 40
FRAME_MS = 100  # 10 fps thumbs: light files, clearly animated


def letterbox(frame, size=(200, 200), inner=(200, 150)):
    frame = frame.convert('RGB').resize(inner, Image.LANCZOS)
    canvas = Image.new('RGB', size, 'white')
    canvas.paste(frame, ((size[0] - inner[0]) // 2,
                         (size[1] - inner[1]) // 2))
    return canvas


def thumb_from_mp4(stem):
    # the animation is usually output block 001, but examples with a
    # static figure first (e.g. save_movie) scrape it as 002
    src = os.path.join(MP4_DIR, f'sphx_glr_{stem}_001.mp4')
    if not os.path.exists(src):
        src = os.path.join(MP4_DIR, f'sphx_glr_{stem}_002.mp4')
    if not os.path.exists(src):
        print(f'  SKIP {stem}: no {src}')
        return
    with tempfile.TemporaryDirectory() as tmp:
        # sample N_FRAMES evenly across the animation
        subprocess.run(
            ['ffmpeg', '-y', '-i', src, '-vf',
             f'fps={N_FRAMES}/30,scale=200:150:flags=lanczos',
             os.path.join(tmp, 'f%03d.png')],
            check=True, capture_output=True)
        frames = sorted(os.listdir(tmp))[:N_FRAMES]
        imgs = [letterbox(Image.open(os.path.join(tmp, f))) for f in frames]
    out = os.path.join(THUMBS, f'sphx_glr_{stem}_thumb.gif')
    imgs[0].save(out, save_all=True, append_images=imgs[1:],
                 duration=FRAME_MS, loop=0)
    print(f'  {stem}: {len(imgs)} frames, {os.path.getsize(out)//1024}KB')


def thumb_from_gif(stem, src):
    if not os.path.exists(src):
        print(f'  SKIP {stem}: no {src}')
        return
    with Image.open(src) as im:
        n = im.n_frames
        picks = {int(i * n / N_FRAMES) for i in range(N_FRAMES)}
        imgs = [letterbox(f.copy()) for i, f in
                enumerate(ImageSequence.Iterator(im)) if i in picks]
    out = os.path.join(THUMBS, f'sphx_glr_{stem}_thumb.gif')
    imgs[0].save(out, save_all=True, append_images=imgs[1:],
                 duration=FRAME_MS, loop=0)
    print(f'  {stem}: {len(imgs)} frames, {os.path.getsize(out)//1024}KB')


def main():
    os.makedirs(THUMBS, exist_ok=True)
    for stem in MPL_ANIMS:
        thumb_from_mp4(stem)
    thumb_from_gif('animate_plotly', PLOTLY_GIF)


if __name__ == '__main__':
    main()
