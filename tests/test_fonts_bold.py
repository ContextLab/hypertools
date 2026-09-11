"""The bundled Noto Sans now ships a Bold face alongside Regular, so
``fontweight='bold'`` resolves to a real bold face instead of silently
falling back to Regular (GH #285, quoted from morph_shapes_zoo:
"hypertools' bundled default (Noto Sans) ships only a Regular face, so
`fontweight='bold'` alone silently falls back to regular").

``NotoSans-Bold.ttf`` was downloaded from the SAME upstream source/version
as the already-vendored ``NotoSans-Regular.ttf`` (Noto Sans Version 2.008,
googlefonts/noto-fonts) -- confirmed byte-identical against a fresh fetch of
the Regular file before the Bold one was added; see the registration comment
in ``hypertools/plot/fonts.py`` and ``hypertools/external/fonts/README.md``
for the exact URLs and sha256 digests.
"""

import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.font_manager import FontProperties, findfont

from hypertools.plot import fonts as hyp_fonts
from hypertools.plot.fonts import bundled_font_files, register_bundled_fonts

_FONT_DIR = hyp_fonts._BUNDLED_FONT_DIR
_REGULAR = os.path.join(_FONT_DIR, 'NotoSans-Regular.ttf')
_BOLD = os.path.join(_FONT_DIR, 'NotoSans-Bold.ttf')


def test_bold_face_file_is_bundled_next_to_regular():
    assert os.path.isfile(_REGULAR)
    assert os.path.isfile(_BOLD), "NotoSans-Bold.ttf not found next to Regular"
    files = bundled_font_files()
    assert any(f.endswith('NotoSans-Bold.ttf') for f in files)
    assert any(f.endswith('NotoSans-Regular.ttf') for f in files)


def test_bold_and_regular_faces_are_similar_in_size():
    # keeps the wheel small: the Bold face should not be dramatically larger
    # than the Regular one it sits next to.
    regular_size = os.path.getsize(_REGULAR)
    bold_size = os.path.getsize(_BOLD)
    assert regular_size > 0 and bold_size > 0
    ratio = bold_size / regular_size
    assert 0.5 < ratio < 2.0, (
        f"NotoSans-Bold.ttf ({bold_size} bytes) is not similar in size to "
        f"NotoSans-Regular.ttf ({regular_size} bytes); ratio={ratio:.2f}")


def test_findfont_resolves_bold_weight_to_the_bold_file():
    register_bundled_fonts()
    bold_path = findfont(
        FontProperties(family='Noto Sans', weight='bold'),
        fallback_to_default=False)
    assert os.path.normpath(bold_path) == os.path.normpath(_BOLD), (
        f"fontweight='bold' resolved to {bold_path!r}, not the bundled "
        f"Bold face -- 'bold' is silently falling back to Regular")


def test_findfont_regular_resolution_is_unchanged():
    register_bundled_fonts()
    regular_path = findfont(
        FontProperties(family='Noto Sans', weight='normal'),
        fallback_to_default=False)
    assert os.path.normpath(regular_path) == os.path.normpath(_REGULAR)


def test_bundled_font_wins_over_an_already_registered_copy(tmp_path):
    # GH #285 release review: reproduce a same-family system font using
    # another real font file in a fresh interpreter, without mocks.
    import shutil
    import subprocess
    import sys
    other = tmp_path / 'system-noto.ttf'
    shutil.copyfile(_REGULAR, other)
    script = '''
import sys
from matplotlib import font_manager as fm
fm.fontManager.addfont(sys.argv[1])
from hypertools.plot.fonts import register_bundled_fonts
register_bundled_fonts()
assert fm.findfont(fm.FontProperties(family='Noto Sans', weight='normal')) == sys.argv[2]
'''
    subprocess.run([sys.executable, '-c', script, str(other), _REGULAR],
                   check=True, capture_output=True, text=True)


def _render_title_rgba(fontweight):
    register_bundled_fonts()
    fig, ax = plt.subplots(figsize=(3, 2), dpi=100)
    ax.set_title('Hypertools', fontfamily='Noto Sans', fontweight=fontweight,
                  fontsize=24)
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())
    plt.close(fig)
    return buf


def test_bold_title_renders_visibly_different_pixels_than_regular():
    regular_buf = _render_title_rgba('regular')
    bold_buf = _render_title_rgba('bold')
    assert regular_buf.shape == bold_buf.shape
    assert not np.array_equal(regular_buf, bold_buf), (
        "a title drawn with fontweight='bold' is pixel-identical to the "
        "same title drawn with fontweight='regular' -- bold is not "
        "actually being rendered with a distinct (bolder) face")


def test_bold_face_will_ship_in_the_wheel_via_existing_package_data_glob():
    """``pyproject.toml``'s ``hypertools.external`` package-data already
    lists the glob ``fonts/*.ttf`` (covers the pre-existing Regular face);
    confirm that glob also matches the new Bold file by name, so no
    packaging-line change was needed to ship it."""
    import fnmatch
    assert fnmatch.fnmatch(os.path.basename(_BOLD), '*.ttf')
    assert os.path.isfile(_BOLD)


def test_explicit_bundled_family_resolves_in_a_fresh_process():
    """``hyp.plot(x, font='Noto Sans')`` -- the BUNDLED family -- raised
    "not a recognized installed font family" in a fresh interpreter on a
    machine without a system Noto Sans, because the bundled faces were only
    registered as a side effect of an earlier plot (review 2026-09-11).
    In-process state masks this (any earlier test has registered them), so
    it runs as a real ``python -c`` subprocess. The subprocess reports which
    FILE the family resolved to, so the assertion holds whether or not the
    machine also has a system copy (the bundled face wins either way)."""
    import subprocess
    import sys
    code = (
        "import matplotlib; matplotlib.use('Agg')\n"
        "import numpy as np, hypertools as hyp\n"
        "from matplotlib import font_manager\n"
        "from hypertools.plot.fonts import resolve_font\n"
        "fp = resolve_font('Noto Sans', ['hello'])\n"
        "print('RESOLVED', font_manager.findfont(fp, "
        "fallback_to_default=False))\n"
        "fig = hyp.plot(np.random.default_rng(0).normal(size=(10, 3)), "
        "font='Noto Sans', show=False)\n"
        "print('PLOTTED', type(fig).__name__)\n")
    out = subprocess.run([sys.executable, '-c', code], capture_output=True,
                         text=True, timeout=300,
                         env=dict(os.environ, MPLBACKEND='Agg'))
    assert out.returncode == 0, out.stderr[-3000:]
    lines = dict(line.split(' ', 1) for line in out.stdout.splitlines()
                 if line.startswith(('RESOLVED ', 'PLOTTED ')))
    assert os.path.samefile(lines['RESOLVED'], _REGULAR), lines
    assert lines['PLOTTED'] == 'Figure'
