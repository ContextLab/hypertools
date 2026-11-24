"""A legend key looks like the dataset it stands for.

1.2.1 gallery review: `examples/animate_market_sectors.py` draws six
sector paths 1.1 points wide and a seventh, the market, 3.4 points wide,
and names them with ``legend_colors=[(label, color), ...]``. Every key of
that legend was the same 2-point line, so the heavy path had a thin key
(the library's pair-list legend has drawn one fixed swatch since 1.1.0;
the 1.2.0 example drew its own legend by hand). With one pair per drawn
dataset, each key now takes its dataset's own format and given line
width, line style, marker, marker size and alpha -- on a static plot, an
animation and a `panels=` grid, which share the one legend path.

Real figures only: the handles' properties are read back, and the keys
are measured in the rendered pixels.
"""
import importlib.util
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.colors as mcolors  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

import hypertools as hyp  # noqa: E402

PAIRS = [('thin', '#d62728'), ('medium', '#2ca02c'), ('thick', '#1f77b4')]
WIDTHS = [1.0, 2.5, 6.0]
DASHES = ['-', '--', ':']
ALPHAS = [1.0, 0.6, 0.3]


def _data(ndims, k=3):
    rng = np.random.default_rng(7)
    return [rng.normal(size=(30, ndims)).cumsum(axis=0) for _ in range(k)]


def _figure(result):
    return getattr(result, 'figure', result)


def _legend(fig):
    """The figure's one legend: on its axes, or the grid's shared one."""
    legends = [ax.get_legend() for ax in fig.axes
               if ax.get_legend() is not None] + list(fig.legends)
    assert len(legends) == 1, f'expected one legend, found {len(legends)}'
    return legends[0]


def _keys(fig):
    legend = _legend(fig)
    return {text.get_text(): handle for handle, text
            in zip(legend.legend_handles, legend.get_texts())}


def _key_pixels(fig, handle):
    """The rendered pixels of one legend key: a boolean mask (rows x
    columns of the key's own box, padded by 10 px above and below) of the
    pixels that are not the white canvas."""
    fig.canvas.draw()
    image = np.asarray(fig.canvas.buffer_rgba()).astype(int)
    box = handle.get_window_extent(fig.canvas.get_renderer())
    height = image.shape[0]
    x0, x1 = int(np.floor(box.x0)), int(np.ceil(box.x1))
    middle = height - (box.y0 + box.y1) / 2
    y0, y1 = int(np.floor(middle - 10)), int(np.ceil(middle + 10))
    crop = image[y0:y1, x0:x1, :3]
    return (255 - crop).max(axis=2) > 24


def _line_thickness_px(fig, handle):
    """How many pixel rows the key's line covers, at its densest column
    (a dashed key has gaps; the densest column is inside a dash)."""
    return int(_key_pixels(fig, handle).sum(axis=0).max())


def _modes():
    return [
        ('static', {}),
        ('animated', {'animate': True, 'duration': 2, 'frame_rate': 5}),
        ('panels', {'panels': True}),
        ('animated panels', {'panels': True, 'animate': True, 'duration': 2,
                             'frame_rate': 5}),
    ]


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close('all')


@pytest.mark.parametrize('ndims', [2, 3])
@pytest.mark.parametrize('mode', [m[0] for m in _modes()])
def test_pair_legend_keys_take_each_datasets_line_styling(mode, ndims):
    """`legend_colors=` pairs, one per dataset: every key has ITS
    dataset's width, dash pattern and alpha, in the handle and on the
    canvas."""
    extra = dict(_modes())[mode]
    fig = _figure(hyp.plot(
        _data(ndims), '-', linewidth=WIDTHS, linestyle=DASHES, alpha=ALPHAS,
        legend_colors=PAIRS, show=False, **extra))
    keys = _keys(fig)
    assert list(keys) == [label for label, _ in PAIRS]
    for (label, color), width, dash, alpha in zip(PAIRS, WIDTHS, DASHES,
                                                  ALPHAS):
        handle = keys[label]
        assert handle.get_linewidth() == pytest.approx(width), label
        assert handle.get_linestyle() == dash, label
        assert handle.get_alpha() == pytest.approx(alpha), label
        assert mcolors.to_hex(handle.get_color()) == color, label

    # the pixels: a key is as thick as its line, lw points at fig.dpi
    for label, width in zip(keys, WIDTHS):
        expected = width * fig.dpi / 72
        measured = _line_thickness_px(fig, keys[label])
        assert abs(measured - expected) <= 2.0, (
            f'{mode} {ndims}-D: the {label!r} key is {measured} px thick '
            f'on the canvas, its {width}-point line is {expected:.1f} px')
    assert (_line_thickness_px(fig, keys['thick'])
            >= 2 * _line_thickness_px(fig, keys['thin']))
    # the dashed keys have gaps along their length; the solid one has none
    covered = {label: _key_pixels(fig, keys[label]).any(axis=0)
               for label in keys}
    assert covered['thin'].mean() > 0.9
    assert covered['medium'].mean() < 0.9
    assert covered['thick'].mean() < 0.9


@pytest.mark.parametrize('ndims', [2, 3])
@pytest.mark.parametrize('animate', [False, True])
def test_pair_legend_keys_take_each_datasets_marker(animate, ndims):
    """A markers-only plot: each key is its dataset's marker, at its
    size."""
    sizes, markers = [4, 9, 16], ['o', 's', '^']
    extra = ({'animate': True, 'duration': 2, 'frame_rate': 5}
             if animate else {})
    fig = _figure(hyp.plot(
        _data(ndims), 'o', markersize=sizes, marker=markers,
        legend_colors=PAIRS, show=False, **extra))
    keys = _keys(fig)
    areas = []
    for (label, _), size, marker in zip(PAIRS, sizes, markers):
        handle = keys[label]
        assert handle.get_marker() == marker, label
        assert handle.get_markersize() == pytest.approx(size), label
        assert handle.get_linestyle() == 'None', label
        areas.append(int(_key_pixels(fig, handle).sum()))
    # the drawn glyphs grow with the marker size
    assert areas[0] < areas[1] < areas[2]
    assert areas[2] > 4 * areas[0]


def test_pair_legend_keys_follow_a_per_dataset_fmt_list():
    """`fmt=` given per dataset: a line, markers, and a line with
    markers each get the key of their own kind."""
    fig = hyp.plot(_data(2), ['-', 'o', 's--'], legend_colors=PAIRS,
                   show=False)
    keys = _keys(fig)
    assert (keys['thin'].get_linestyle(), keys['thin'].get_marker()) \
        == ('-', 'None')
    assert (keys['medium'].get_linestyle(), keys['medium'].get_marker()) \
        == ('None', 'o')
    assert (keys['thick'].get_linestyle(), keys['thick'].get_marker()) \
        == ('--', 's')


def test_pair_legend_without_styling_keeps_the_plain_swatch():
    """Nothing styled per dataset: the keys are the 2-point swatches they
    have always been."""
    fig = hyp.plot(_data(3), '-', legend_colors=PAIRS, show=False)
    for handle in _keys(fig).values():
        assert handle.get_linewidth() == 2
        assert handle.get_linestyle() == '-'
        assert handle.get_alpha() is None


def test_pairs_that_do_not_pair_up_with_the_datasets_stay_plain():
    """Two entries for three datasets name no dataset in particular: the
    keys stay plain swatches, whatever the datasets' widths."""
    fig = hyp.plot(_data(3), '-', linewidth=WIDTHS, legend_colors=PAIRS[:2],
                   show=False)
    for handle in _keys(fig).values():
        assert handle.get_linewidth() == 2
    # ...and a matrix hue's palette swatches (one per hue COLUMN) are not
    # the datasets either, even when there are as many of them
    hue = [np.tile(np.eye(3)[i], (30, 1)) for i in range(3)]
    fig = hyp.plot(_data(3), '-', linewidth=WIDTHS, hue=hue,
                   hue_mode='mixture', palette=[c for _, c in PAIRS],
                   legend=True, colorbar=False, show=False)
    for handle in _keys(fig).values():
        assert handle.get_linewidth() == 2


@pytest.mark.parametrize('panels', [False, True])
@pytest.mark.parametrize('ndims', [2, 3])
def test_plotly_pair_legend_keys_take_each_datasets_width(ndims, panels):
    pytest.importorskip('plotly')
    fig = hyp.plot(_data(ndims), '-', linewidth=WIDTHS, legend_colors=PAIRS,
                   backend='plotly', show=False,
                   **({'panels': True} if panels else {}))
    widths = {trace.name: trace.line.width for trace in fig.data
              if trace.showlegend}
    assert widths == {label: width for (label, _), width
                      in zip(PAIRS, WIDTHS)}


def _load_market_example():
    path = os.path.join(os.path.dirname(__file__), os.pardir, 'examples',
                        'animate_market_sectors.py')
    spec = importlib.util.spec_from_file_location('animate_market_sectors',
                                                  path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_market_example_key_for_the_market_path_is_as_heavy_as_the_path():
    """The gallery example the review flagged: the 'Market' key is 3.4
    points like the market path, the six sector keys 1.1 like theirs, in
    the handle and on the first and a late frame."""
    example = _load_market_example()
    anim = example.construct_artifact(example.fixture_data())
    fig = anim.figure
    keys = _keys(fig)
    assert list(keys) == list(example.SECTORS) + [
        'Market (average across sectors)']
    market = keys['Market (average across sectors)']
    assert market.get_linewidth() == pytest.approx(3.4)
    assert mcolors.to_hex(market.get_color()) == '#666666'
    for name in example.SECTORS:
        assert keys[name].get_linewidth() == pytest.approx(1.1)

    for frame in (0, anim.n_frames - 300):
        anim.draw_frame(frame)
        thick = _line_thickness_px(fig, market)
        thin = max(_line_thickness_px(fig, keys[name])
                   for name in example.SECTORS)
        assert abs(thick - 3.4 * fig.dpi / 72) <= 2.0, (frame, thick)
        assert thick >= thin + 2, (frame, thick, thin)


@pytest.mark.parametrize('ndims', [2, 3])
@pytest.mark.parametrize('fmt, style', [
    ('-', {'linestyle': ['-', '--', ':']}),
    ('o', {'marker': ['o', 's', '^']}),
    ('-o', {'marker': ['o', 's', '^'], 'linestyle': ['-', '--', ':']}),
])
def test_animated_fmt_with_style_keywords_does_not_warn(fmt, style, ndims):
    """Found while testing the keys: an animation passed matplotlib the
    format string AND the dataset's `linestyle=`/`marker=`, so matplotlib
    warned that the property "is redundantly defined" once per dataset
    (the static plot of the same call is silent). The keyword wins, as
    before, and nothing is warned."""
    import warnings
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        anim = hyp.plot(_data(ndims), fmt, animate=True, chemtrails=True,
                        duration=2, frame_rate=5, show=False, **style)
        anim.draw_frame(anim.n_frames - 1)
    redundant = [str(w.message) for w in caught
                 if 'redundantly defined' in str(w.message)]
    assert redundant == []
    lines = anim.figure.axes[0].lines
    for key, values in style.items():
        drawn = {getattr(line, f'get_{key}')() for line in lines}
        assert set(values) <= drawn, (key, drawn)
