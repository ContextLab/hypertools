"""Render the five shipped launch examples to mp4 for the Bluesky thread.

Usage: .venv/bin/python scripts/render_bluesky_clips.py [stem ...]
Output: notes/bluesky-launch/<stem>.mp4 (that folder is gitignored).

Each clip is the example's own construct_artifact() output (the exact
animation the tutorial shows), saved with the example's frame rate at a dpi
chosen so both pixel dimensions are even (h264 yuv420p needs that). Where the
dpi below equals the tutorial notebook's, the clip is byte-for-byte the mp4
that executing the notebook writes to docs/tutorials/, so copying that file
is equivalent to running this (the market and weather clips are 1200 and 2400
frames; do not render them twice)."""
import importlib
import os
import sys
import time

import matplotlib.pyplot as plt

os.environ.setdefault('MPLBACKEND', 'Agg')
os.environ.setdefault('HF_HUB_DISABLE_PROGRESS_BARS', '1')
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'examples'))
OUT = os.path.join(ROOT, 'notes', 'bluesky-launch')
JOBS = [  # (module, loader, output stem, dpi)
    ('animate_market_sectors', 'load_market', '20_market_sectors', 100),      # 8x8 -> 800x800
    ('animate_weather_decades', 'load_weather', '21_weather_decades', 100),   # 14x7 -> 1400x700
    ('animate_painting_embeddings', 'load_paintings', '22_painting_embeddings', 100),  # 17.5x9.24 -> 1750x924
    ('animate_conversation', 'embed_turns', '23_conversation', 110),          # 9.58x8.63 (legend+2-line title) -> 1052x948
    ('animate_morph_zoo', 'load_shapes', '24_morph_zoo', 130),                # 6x6 -> 780x780
]
only = sys.argv[1:]
for module, loader, stem, dpi in JOBS:
    if only and stem not in only:
        continue
    t0 = time.time()
    mod = importlib.import_module(module)
    data = getattr(mod, loader)()
    src = getattr(data, 'source', getattr(data, 'vectorizer', '?'))
    anim = mod.construct_artifact(data)
    n = anim.n_frames
    fps = anim._fps()
    path = os.path.join(OUT, stem + '.mp4')
    anim.save(path, dpi=dpi)
    w, h = [round(v * dpi) for v in anim.figure.get_size_inches()]
    mb = os.path.getsize(path) / 1e6
    print(f'{stem}: {n} frames @ {fps} fps = {n / fps:.1f} s, {w}x{h}, '
          f'{mb:.2f} MB, source={src!r}, {time.time() - t0:.0f} s', flush=True)
    plt.close('all')
