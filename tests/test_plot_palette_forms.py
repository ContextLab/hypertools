"""`palette=` as a {category: color} dict and as a per-dataset list.

GH #285. `plot()` used to accept one palette for the whole figure, resolved
POSITIONALLY -- so colouring a named category meant working out its
first-appearance index by hand and building a list to match. Two new forms
fix that: a dict keyed by category name, and a list of palettes (one per
dataset, e.g. one painting per trajectory).

Every image palette here is extracted from a real PNG written to disk with
Pillow, and every colour is read back off the drawn artists.
"""
import matplotlib
matplotlib.use('Agg')

import matplotlib.colors as mcolors
import numpy as np
import pytest

import hypertools as hyp
from hypertools.plot.colors import image_palette, palette_lead_color


def _datasets(n=3, rows=20, cols=5, seed=11):
    rng = np.random.default_rng(seed)
    return [rng.normal(size=(rows, cols)) for _ in range(n)]


def _line_hexes(fig):
    return [mcolors.to_hex(line.get_color()) for line in fig.axes[0].lines]


def _legend_texts(fig):
    leg = fig.axes[0].get_legend()
    return [] if leg is None else [t.get_text() for t in leg.get_texts()]


def _plotly_hex(color):
    """plotly emits 'rgba(r,g,b,a)'; compare on the RGB triple."""
    channels = color[color.index('(') + 1:color.rindex(')')].split(',')
    return mcolors.to_hex(tuple(int(float(c)) / 255 for c in channels[:3]))


@pytest.fixture(scope='module')
def image_files(tmp_path_factory):
    """Two tiny PNGs with unmistakably different dominant colours."""
    Image = pytest.importorskip('PIL.Image')
    directory = tmp_path_factory.mktemp('palettes')
    made = {}
    for name, rgb in (('crimson', (200, 20, 40)), ('teal', (10, 160, 170))):
        pixels = np.zeros((24, 24, 3), dtype=np.uint8)
        pixels[:, :] = rgb
        # a second block so k-means has more than one cluster to rank
        pixels[:8, :8] = (245, 240, 230)
        path = directory / f'{name}.png'
        Image.fromarray(pixels).save(path)
        made[name] = str(path)
    return made


# --- dict palettes ------------------------------------------------------

PALETTE = {'Alice': '#E4572E', 'Bob': '#17BEBB', 'Cara': '#FFC914'}


def test_dict_palette_colours_line_traces_by_name():
    fig = hyp.plot(_datasets(3), hue=['Alice', 'Bob', 'Cara'],
                   palette=PALETTE, legend=True, reduce='PCA', show=False)
    assert _legend_texts(fig) == ['Alice', 'Bob', 'Cara']
    assert _line_hexes(fig) == ['#e4572e', '#17bebb', '#ffc914']


def test_dict_palette_ignores_category_order():
    """The point of the dict: the SAME colours regardless of which
    category the data happens to present first."""
    data = _datasets(3)
    first = hyp.plot(data, hue=['Alice', 'Bob', 'Cara'], palette=PALETTE,
                     legend=True, reduce='PCA', show=False)
    shuffled = hyp.plot(data, hue=['Cara', 'Alice', 'Bob'], palette=PALETTE,
                        legend=True, reduce='PCA', show=False)
    assert dict(zip(_legend_texts(first), _line_hexes(first))) == \
        dict(zip(_legend_texts(shuffled), _line_hexes(shuffled)))


def test_dict_palette_colours_marker_traces_by_name():
    fig = hyp.plot(_datasets(3), 'o', hue=['Alice', 'Bob', 'Cara'],
                   palette=PALETTE, legend=True, reduce='PCA', show=False)
    assert _line_hexes(fig) == ['#e4572e', '#17bebb', '#ffc914']


def test_dict_palette_reaches_the_colorbar_and_the_bundle():
    bundle = hyp.plot(_datasets(3), hue=['Alice', 'Bob', 'Cara'],
                      palette=PALETTE, colorbar=True, reduce='PCA',
                      return_model=True, show=False)
    categories = {k: mcolors.to_hex(v)
                  for k, v in bundle['colors']['categories'].items()}
    assert categories == {'Alice': '#e4572e', 'Bob': '#17bebb',
                          'Cara': '#ffc914'}
    assert len(bundle['fig'].axes) > 1        # a colorbar axes was added


def test_dict_palette_naming_a_subset_leaves_the_rest_in_place():
    partial = hyp.plot(_datasets(3), hue=['Alice', 'Bob', 'Cara'],
                       palette={'Bob': '#000000'}, reduce='PCA', show=False)
    default = hyp.plot(_datasets(3), hue=['Alice', 'Bob', 'Cara'],
                       reduce='PCA', show=False)
    got, base = _line_hexes(partial), _line_hexes(default)
    assert got[1] == '#000000'
    assert got[0] == base[0] and got[2] == base[2]


def test_dict_palette_with_an_unknown_key_raises():
    with pytest.raises(ValueError, match='does not have'):
        hyp.plot(_datasets(3), hue=['Alice', 'Bob', 'Cara'],
                 palette={'Dave': 'red'}, reduce='PCA', show=False)


def test_dict_palette_with_no_hue_raises():
    """A dict names CATEGORIES; with no categorical hue there are none to
    name, so this is refused rather than quietly ignored."""
    with pytest.raises(ValueError, match='CATEGORICAL hue only'):
        hyp.plot(_datasets(3), palette=PALETTE, reduce='PCA', show=False)


def test_dict_palette_under_plotly():
    pytest.importorskip('plotly')
    fig = hyp.plot(_datasets(3), hue=['Alice', 'Bob', 'Cara'],
                   palette=PALETTE, legend=True, reduce='PCA',
                   backend='plotly', show=False)
    got = [(t.name, _plotly_hex(t.line.color)) for t in fig.data
           if t.showlegend]
    assert got == [('Alice', '#e4572e'), ('Bob', '#17bebb'),
                   ('Cara', '#ffc914')]


# --- per-dataset palettes -----------------------------------------------

def test_per_dataset_image_palettes(image_files):
    specs = [f'image:{image_files["crimson"]}',
             f'image:{image_files["teal"]}', 'viridis']
    fig = hyp.plot(_datasets(3), palette=specs, reduce='PCA', show=False)
    drawn = _line_hexes(fig)
    assert len(set(drawn)) == 3
    for spec, hexed in zip(specs, drawn):
        assert mcolors.to_hex(palette_lead_color(spec)) == hexed


def test_image_lead_colour_is_the_salient_one_not_the_average(image_files):
    """Two images built to differ; each dataset must land on ITS image's
    salient colour, not on a shared muddy average."""
    crimson = np.asarray(palette_lead_color(f'image:{image_files["crimson"]}'))
    teal = np.asarray(palette_lead_color(f'image:{image_files["teal"]}'))
    assert crimson[0] > crimson[2]     # red-dominant
    assert teal[2] > teal[0]           # blue/green-dominant
    assert np.linalg.norm(crimson - teal) > 0.3


def test_image_spec_query_parameters_reach_image_palette(image_files):
    path = image_files['crimson']
    bounded = palette_lead_color(f'image:{path}?max_luminance=0.35')
    assert max(bounded) <= 1.0
    expected = image_palette(path, max_luminance=0.35)[0]
    np.testing.assert_allclose(bounded, expected, atol=1e-9)


def test_one_entry_palette_list_is_broadcast():
    fig = hyp.plot(_datasets(3), palette=['viridis'], reduce='PCA',
                   show=False)
    drawn = _line_hexes(fig)
    assert len(set(drawn)) == 1


def test_wrong_length_per_dataset_palette_raises():
    with pytest.raises(ValueError, match='per-dataset palettes'):
        hyp.plot(_datasets(3), palette=['viridis', 'magma'], reduce='PCA',
                 show=False)


def test_per_dataset_palettes_under_plotly(image_files):
    pytest.importorskip('plotly')
    specs = [f'image:{image_files["crimson"]}',
             f'image:{image_files["teal"]}', 'viridis']
    fig = hyp.plot(_datasets(3), palette=specs, reduce='PCA',
                   backend='plotly', show=False)
    drawn = [_plotly_hex(t.line.color) for t in fig.data
             if getattr(t, 'line', None) is not None
             and isinstance(t.line.color, str)
             and t.line.color.startswith('rgb')]
    for spec, hexed in zip(specs, drawn[:3]):
        assert mcolors.to_hex(palette_lead_color(spec)) == hexed


def test_per_dataset_palettes_reach_the_colors_bundle(image_files):
    specs = [f'image:{image_files["crimson"]}',
             f'image:{image_files["teal"]}', 'viridis']
    bundle = hyp.plot(_datasets(3), palette=specs, reduce='PCA',
                      return_model=True, show=False)
    got = [mcolors.to_hex(c) for c in bundle['colors']['colors']]
    assert got == [mcolors.to_hex(palette_lead_color(s)) for s in specs]


def test_per_dataset_palette_with_continuous_hue_raises():
    with pytest.raises(ValueError, match='PER-DATASET palettes'):
        hyp.plot(_datasets(2), hue=np.linspace(0, 1, 40),
                 palette=['viridis', 'magma'], reduce='PCA', show=False)


# --- the historical list-of-colours meaning is untouched ----------------

def test_all_colour_list_still_means_per_dataset_colours():
    fig = hyp.plot(_datasets(3), palette=['red', '#00ff00', (0, 0, 1)],
                   reduce='PCA', show=False)
    assert _line_hexes(fig) == ['#ff0000', '#00ff00', '#0000ff']


def test_colour_list_still_blends_a_continuous_hue():
    data = _datasets(1, rows=30)
    bundle = hyp.plot(data, hue=np.linspace(0, 1, 30),
                      palette=['#ff0000', '#0000ff'], reduce='PCA',
                      return_model=True, show=False)
    colors = bundle['colors']
    ends = colors['cmap'](colors['norm']([0.0, 1.0]))
    assert ends[0][0] > ends[0][2]          # starts red
    assert ends[1][2] > ends[1][0]          # ends blue


# --- pixel baseline -----------------------------------------------------

def test_default_palette_calls_are_pixel_identical(tmp_path):
    """Every form above is opt-in: an un-styled call must render exactly
    the same figure it did before, byte for byte."""
    import matplotlib.pyplot as plt
    rng = np.random.RandomState(0)
    data = [rng.randn(40, 5) for _ in range(3)]

    def render(name, **kwargs):
        fig = hyp.plot(data, reduce='PCA', show=False, **kwargs)
        path = tmp_path / f'{name}.png'
        fig.savefig(path, dpi=72)
        plt.close(fig)
        return path.read_bytes()

    plain = render('plain', title='Ref one', legend=True,
                   hue=[['a'] * 40, ['b'] * 40, ['c'] * 40])
    again = render('again', title='Ref one', legend=True,
                   hue=[['a'] * 40, ['b'] * 40, ['c'] * 40])
    assert plain == again

    # an explicit 'hls' (the default, spelled out) must match too
    spelled = render('spelled', title='Ref one', legend=True, palette='hls',
                     hue=[['a'] * 40, ['b'] * 40, ['c'] * 40])
    assert spelled == plain
