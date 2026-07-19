"""Guard the animated-gallery-thumbnail wiring (QC 2026-07).

Animated gallery examples show a still PNG unless `docs/post_build.py` swaps it
for the shipped GIF -- a Read-the-Docs `post_build` job (see `.readthedocs.yaml`).
Two examples (plot_story_trajectories, animate_surface_morph) each silently
regressed to a frozen thumbnail because their GIF was shipped but never
registered in `post_build.GIF_REPLACEMENTS` (or vice versa). This test keeps the
two sides in lockstep so the next one can't slip through.
"""
import importlib.util
import os

import pytest

DOCS_DIR = os.path.join(os.path.dirname(__file__), '..', 'docs')
THUMBS_DIR = os.path.join(DOCS_DIR, '_static', 'thumbnails')


def _load_post_build():
    path = os.path.join(DOCS_DIR, 'post_build.py')
    if not os.path.exists(path):
        pytest.skip('docs/post_build.py not present')
    spec = importlib.util.spec_from_file_location('_hyp_post_build', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_every_shipped_gif_thumbnail_is_registered():
    """Each shipped `_static/thumbnails/*.gif` must appear as a GIF_REPLACEMENTS
    value, and every registered gif must actually be shipped -- otherwise the
    png->gif swap silently no-ops and the gallery card stays frozen."""
    if not os.path.isdir(THUMBS_DIR):
        pytest.skip('docs/_static/thumbnails not present')
    mod = _load_post_build()
    shipped = {f for f in os.listdir(THUMBS_DIR) if f.endswith('.gif')}
    registered = set(mod.GIF_REPLACEMENTS.values())
    assert shipped == registered, (
        f"shipped-but-unregistered gifs (frozen cards): {shipped - registered}; "
        f"registered-but-missing gifs (dangling): {registered - shipped}")


def test_replacements_are_png_to_same_name_gif():
    mod = _load_post_build()
    for png, gif in mod.GIF_REPLACEMENTS.items():
        assert png.endswith('.png'), f"key {png!r} is not a .png"
        assert gif == png[:-4] + '.gif', f"{png!r} -> {gif!r} is not the same stem"
