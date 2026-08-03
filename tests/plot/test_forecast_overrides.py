# -*- coding: utf-8 -*-
"""`forecast_*=`: style the forecasts independently of the observed data.

Maintainer decision (2026-08-02): "if left unspecified, let's set forecast
colors and styles to the same as the corresponding 'observed' data, but with
50% more alpha transparency. if specified, then the forecast_hue and/or
forecast_cluster overrides the default styling for the forecasts. it could
potentially even be the case that observed vs. forecasted data have different
styles and/or different clustering and/or different colormaps/palettes"

So there are two layers, and the DEFAULT layer is unchanged: inheritance.
`forecast_hue=`, `forecast_cluster=`, `forecast_palette=` and `forecast_fmt=`
each replace one aspect of the inherited style and leave the rest alone.

`forecast_cluster=` clusters the forecast ENDPOINTS -- where each series is
predicted to END UP -- so a forecast's colour answers "which of these are
heading to the same place?". It deliberately does NOT recluster the observed
data: inheriting the observed assignment is what the default already gives,
so defining it that way would make the kwarg a no-op.
"""

import matplotlib
matplotlib.use('Agg')

import warnings

import numpy as np
import pytest

import hypertools as hyp


def _forecasts(fig):
    return [ln for ln in fig.axes[0].lines
            if getattr(ln, '_hyp_forecast_role', None) == 'static']


def _by_dataset(fig):
    """{dataset index: artist} -- pairing by identity, not drawing order."""
    return {getattr(ln, '_hyp_forecast_dataset', None): ln
            for ln in _forecasts(fig)}


def _rgb(artist):
    return np.asarray(matplotlib.colors.to_rgba(artist.get_color())[:3])


def _walks(n=3, rows=30, dims=3, seed=0):
    rng = np.random.default_rng(seed)
    return [rng.normal(size=(rows, dims)).cumsum(axis=0) for _ in range(n)]


def _converging(seed=0, rows=40):
    """Four series whose position and whose destination group them
    DIFFERENTLY -- which is the only reason the tests below can tell an
    endpoint clustering from an inherited one.

    They start in two far-apart places on x (-20 and +20) and converge
    toward x = 0 while diverging on y. So over the OBSERVED span x dominates
    and clusters them {0, 1} | {2, 3}; by the forecast ENDPOINTS x has all
    but collapsed and y dominates, clustering them {0, 2} | {1, 3}.

    `test_forecast_cluster_disagrees_with_the_observed_clustering` asserts
    that disagreement directly: an earlier fixture (four corners, two
    destinations) looked like it separated the two and did not, so every
    endpoint assertion here would have been satisfied by plain inheritance.
    """
    rng = np.random.default_rng(seed)
    starts = np.array([[-20., 0., 0.], [-20., 0., 0.],
                       [20., 0., 0.], [20., 0., 0.]])
    ends = np.array([[-5., 8., 0.], [-5., -8., 0.],
                     [5., 8., 0.], [5., -8., 0.]])
    steps = np.linspace(0, 1, rows)[:, None]
    return [starts[i] + (ends[i] - starts[i]) * steps
            + rng.normal(scale=0.05, size=(rows, 3)) for i in range(4)]


# --------------------------------------------------------------------------
# the default layer is untouched
# --------------------------------------------------------------------------

def test_with_no_override_a_forecast_still_inherits_its_traces_style():
    """The overrides are additive: passing none of them must leave the
    inherited-style-at-half-alpha default exactly as it was."""
    data = _walks(2)
    fig = hyp.plot(data, '--', predict='Kalman', t=4, show=False)
    obs = [ln for ln in fig.axes[0].lines
           if getattr(ln, '_hyp_forecast_role', None) is None]
    fc = _by_dataset(fig)
    assert len(fc) == 2
    for i, o in enumerate(obs):
        assert np.allclose(_rgb(fc[i]), _rgb(o))
        assert fc[i].get_linestyle() == o.get_linestyle()
        assert fc[i].get_alpha() == pytest.approx(0.5)


# --------------------------------------------------------------------------
# forecast_fmt=
# --------------------------------------------------------------------------

def test_forecast_fmt_restyles_the_forecast_and_leaves_the_data_alone():
    data = _walks(2)
    fig = hyp.plot(data, '-', predict='Kalman', t=4, forecast_fmt=':',
                   show=False)
    obs = [ln for ln in fig.axes[0].lines
           if getattr(ln, '_hyp_forecast_role', None) is None]
    assert all(o.get_linestyle() == '-' for o in obs), (
        'forecast_fmt= must not restyle the observed data')
    for f in _forecasts(fig):
        assert f.get_linestyle() in (':', 'dotted')


def test_forecast_fmt_does_not_change_the_inherited_COLOUR():
    """Only the aspect named is overridden. A dashed forecast of a red trace
    is still red."""
    data = _walks(2)
    fig = hyp.plot(data, '-', predict='Kalman', t=4, forecast_fmt='--',
                   show=False)
    obs = [ln for ln in fig.axes[0].lines
           if getattr(ln, '_hyp_forecast_role', None) is None]
    fc = _by_dataset(fig)
    for i, o in enumerate(obs):
        assert np.allclose(_rgb(fc[i]), _rgb(o)), (
            'forecast_fmt= changed the colour as well as the style')


def test_forecast_fmt_accepts_one_style_per_dataset():
    data = _walks(3)
    fig = hyp.plot(data, '-', predict='Kalman', t=4,
                   forecast_fmt=['-', '--', ':'], show=False)
    fc = _by_dataset(fig)
    assert fc[0].get_linestyle() in ('-', 'solid')
    assert fc[1].get_linestyle() in ('--', 'dashed')
    assert fc[2].get_linestyle() in (':', 'dotted')


# --------------------------------------------------------------------------
# forecast_palette=
# --------------------------------------------------------------------------

def test_forecast_palette_alone_recolours_the_forecasts_by_dataset():
    """With no forecast grouping given there is nothing to colour BY except
    the series itself, so the palette is spent one colour per dataset."""
    data = _walks(3)
    fig = hyp.plot(data, '-', predict='Kalman', t=4,
                   forecast_palette=['#ff0000', '#00ff00', '#0000ff'],
                   show=False)
    fc = _by_dataset(fig)
    assert np.allclose(_rgb(fc[0]), [1, 0, 0])
    assert np.allclose(_rgb(fc[1]), [0, 1, 0])
    assert np.allclose(_rgb(fc[2]), [0, 0, 1])


def test_forecast_palette_leaves_the_observed_data_alone():
    """"different colormaps/palettes" for observed vs forecast -- so the
    observed traces must keep the palette they were drawn with."""
    data = _walks(3)
    plain = hyp.plot(data, '-', predict='Kalman', t=4, show=False)
    recoloured = hyp.plot(data, '-', predict='Kalman', t=4,
                          forecast_palette=['#ff0000', '#00ff00', '#0000ff'],
                          show=False)
    for a, b in zip(
            [ln for ln in plain.axes[0].lines
             if getattr(ln, '_hyp_forecast_role', None) is None],
            [ln for ln in recoloured.axes[0].lines
             if getattr(ln, '_hyp_forecast_role', None) is None]):
        assert np.allclose(_rgb(a), _rgb(b))


# --------------------------------------------------------------------------
# forecast_hue=
# --------------------------------------------------------------------------

def test_forecast_hue_groups_the_forecasts_by_the_values_given():
    """One value per DATASET (a forecast is one trace, not one per row).
    Datasets sharing a value must share a colour, and differing values must
    differ."""
    data = _walks(4)
    fig = hyp.plot(data, '-', predict='Kalman', t=4,
                   forecast_hue=['up', 'down', 'up', 'down'], show=False)
    fc = _by_dataset(fig)
    assert np.allclose(_rgb(fc[0]), _rgb(fc[2]))
    assert np.allclose(_rgb(fc[1]), _rgb(fc[3]))
    assert not np.allclose(_rgb(fc[0]), _rgb(fc[1]))


def test_forecast_hue_takes_its_colours_from_forecast_palette():
    data = _walks(4)
    fig = hyp.plot(data, '-', predict='Kalman', t=4,
                   forecast_hue=['up', 'down', 'up', 'down'],
                   forecast_palette=['#ff0000', '#0000ff'], show=False)
    fc = _by_dataset(fig)
    drawn = {tuple(np.round(_rgb(fc[i]), 3)) for i in range(4)}
    assert drawn == {(1.0, 0.0, 0.0), (0.0, 0.0, 1.0)}


def test_a_per_observation_forecast_hue_is_rejected_clearly():
    """Passing `hue=`'s shape by mistake is the obvious error to make here.
    Grouping by `==` on arrays raises numpy's "truth value of an array is
    ambiguous" from inside the colour code, which names neither the kwarg
    nor the mistake."""
    data = _walks(2)
    with pytest.raises(ValueError, match='forecast_hue'):
        hyp.plot(data, '-', predict='Kalman', t=4,
                 forecast_hue=[np.zeros(30), np.ones(30)], show=False)


def test_forecast_hue_must_have_one_value_per_dataset():
    data = _walks(3)
    with pytest.raises(ValueError, match='forecast_hue'):
        hyp.plot(data, '-', predict='Kalman', t=4,
                 forecast_hue=['a', 'b'], show=False)


# --------------------------------------------------------------------------
# forecast_cluster= -- clusters the ENDPOINTS
# --------------------------------------------------------------------------

def test_forecast_cluster_groups_by_where_the_forecasts_END_UP():
    """The defining test. The expected grouping is re-derived through the
    PUBLIC clustering API from the bundle's forecasts (model output, which
    the styling code does not produce), so this is not the implementation
    checking itself."""
    data = _converging()
    bundle = hyp.plot(data, '-', predict='Kalman', t=6, show=False,
                      return_model=True, forecast_cluster='KMeans',
                      forecast_n_clusters=2)
    endpoints = np.vstack([np.asarray(f)[-1]
                           for f in bundle['predict']['forecasts']])
    expected = np.asarray(hyp.cluster(endpoints, cluster='KMeans',
                                      n_clusters=2)).ravel()
    fc = _by_dataset(bundle['fig'])
    assert len(fc) == len(data)
    for i in range(len(data)):
        for j in range(len(data)):
            same_cluster = expected[i] == expected[j]
            same_colour = np.allclose(_rgb(fc[i]), _rgb(fc[j]), atol=1e-6)
            assert same_colour == same_cluster, (
                f'datasets {i} and {j}: endpoint clusters '
                f'{expected[i]}/{expected[j]} but colours '
                f'{_rgb(fc[i])}/{_rgb(fc[j])}')


def test_forecast_cluster_disagrees_with_the_observed_clustering():
    """Without this the test above proves nothing: an implementation that
    simply inherited the observed cluster colour would satisfy it whenever
    the two groupings happened to coincide. They must not coincide here."""
    data = _converging()
    observed = np.asarray(hyp.cluster(data, cluster='KMeans',
                                      n_clusters=2)).ravel()
    # one observed label per OBSERVATION; take each dataset's majority
    rows = len(data[0])
    per_dataset_observed = [
        np.bincount(observed[i * rows:(i + 1) * rows]).argmax()
        for i in range(len(data))]
    bundle = hyp.plot(data, '-', predict='Kalman', t=6, show=False,
                      return_model=True)
    endpoints = np.vstack([np.asarray(f)[-1]
                           for f in bundle['predict']['forecasts']])
    per_dataset_endpoint = np.asarray(
        hyp.cluster(endpoints, cluster='KMeans', n_clusters=2)).ravel()

    def _pairs(labels):
        return {(i, j) for i in range(len(labels))
                for j in range(len(labels)) if labels[i] == labels[j]}

    assert _pairs(per_dataset_observed) != _pairs(per_dataset_endpoint), (
        'the fixture no longer separates "where they are" from "where they '
        'are heading", so the endpoint-clustering tests cannot discriminate')


def test_forecast_cluster_and_forecast_hue_are_mutually_exclusive():
    """Same relationship `hue=` and `cluster=` have: both decide the same
    thing, so accepting both would mean silently picking a winner."""
    data = _walks(4)
    with pytest.raises(ValueError, match='forecast_hue.*forecast_cluster'):
        hyp.plot(data, '-', predict='Kalman', t=4,
                 forecast_hue=['a', 'b', 'a', 'b'],
                 forecast_cluster='KMeans', show=False)


def test_forecast_cluster_on_a_single_forecast_warns_and_inherits():
    """Clustering one endpoint carries no information -- every partition of
    one point is the same partition."""
    data = _walks(1)
    with pytest.warns(UserWarning, match='forecast_cluster'):
        fig = hyp.plot(data, '-', predict='Kalman', t=4,
                       forecast_cluster='KMeans', show=False)
    obs = [ln for ln in fig.axes[0].lines
           if getattr(ln, '_hyp_forecast_role', None) is None]
    assert np.allclose(_rgb(_forecasts(fig)[0]), _rgb(obs[0])), (
        'having declined to cluster, it must fall back to inheritance'
    )


def test_asking_for_more_clusters_than_forecasts_says_what_the_counts_are():
    """The clusterer's own validator decides what is legal -- but its message
    ("n_samples=3 should be >= n_clusters=5") never says that `n_samples` is
    the number of datasets, which is the one thing the user needs to know."""
    data = _walks(3)
    with pytest.raises(ValueError) as exc:
        hyp.plot(data, '-', predict='Kalman', t=4,
                 forecast_cluster='KMeans', forecast_n_clusters=5, show=False)
    message = str(exc.value)
    assert 'forecast_cluster' in message
    assert '3' in message and 'forecast_n_clusters=5' in message


def test_forecast_n_clusters_without_forecast_cluster_warns():
    data = _walks(3)
    with pytest.warns(UserWarning, match='forecast_n_clusters'):
        hyp.plot(data, '-', predict='Kalman', t=4, forecast_n_clusters=2,
                 show=False)


# --------------------------------------------------------------------------
# the overrides need predict=, and compose with hue=/cluster=
# --------------------------------------------------------------------------

@pytest.mark.parametrize('kw', [
    dict(forecast_hue=['a', 'b']),
    dict(forecast_cluster='KMeans'),
    dict(forecast_palette='husl'),
    dict(forecast_fmt=':'),
])
def test_a_forecast_override_without_predict_is_an_error(kw):
    """There is nothing to style. Silently ignoring it would leave the user
    staring at a plot wondering why their setting did nothing."""
    with pytest.raises(ValueError, match='predict='):
        hyp.plot(_walks(2), '-', show=False, **kw)


def test_forecast_styling_survives_a_regrouping_hue():
    """The observed data is regrouped into runs; the forecasts are styled by
    their OWN grouping, which is the point of having a separate one."""
    data = _walks(2)
    fig = hyp.plot(data, '-', predict='Kalman', t=4,
                   hue=(['a'] * 5 + ['b'] * 5) * 6,
                   forecast_palette=['#ff0000', '#0000ff'], show=False)
    fc = _by_dataset(fig)
    assert np.allclose(_rgb(fc[0]), [1, 0, 0])
    assert np.allclose(_rgb(fc[1]), [0, 0, 1])


# --------------------------------------------------------------------------
# input validation
#
# `resolve_forecast_overrides` is internal, but it is independently
# importable and every one of these inputs is a plausible typo, so each has
# to fail by NAMING the kwarg rather than surfacing as an internal error from
# inside numpy, matplotlib or sklearn.
# --------------------------------------------------------------------------

def test_a_non_iterable_forecast_fmt_names_the_kwarg():
    """`list(3)` raises "'int' object is not iterable", which names neither
    the kwarg nor what it should have been given."""
    with pytest.raises(TypeError, match='forecast_fmt'):
        hyp.plot(_walks(2), '-', predict='Kalman', t=4, forecast_fmt=3,
                 show=False)


def test_a_forecast_fmt_entry_that_is_not_a_format_string_is_rejected():
    """One bad entry in an otherwise fine list. Left alone it reaches a
    different layer on each backend, so the two would not even fail alike."""
    with pytest.raises(TypeError, match='forecast_fmt'):
        hyp.plot(_walks(2), '-', predict='Kalman', t=4,
                 forecast_fmt=['-', 3], show=False)


def test_a_bytes_forecast_fmt_is_decoded_rather_than_iterated():
    """`list(b'--')` is `[45, 45]` -- two ints, silently taken as one format
    per dataset. Decoding is the only reading that is not nonsense."""
    fig = hyp.plot(_walks(2), '-', predict='Kalman', t=4, forecast_fmt=b':',
                   show=False)
    assert all(f.get_linestyle() in (':', 'dotted') for f in _forecasts(fig))


def test_an_unparseable_forecast_fmt_is_rejected_before_anything_is_drawn():
    """`fmt=` and `forecast_fmt=` share matplotlib's grammar, so they must
    share its verdict on what is a legal string -- said once, here, rather
    than differently by each backend at drawing time."""
    with pytest.raises(ValueError, match='forecast_fmt'):
        hyp.plot(_walks(2), '-', predict='Kalman', t=4, forecast_fmt='zz',
                 show=False)


def test_a_bare_string_forecast_hue_is_rejected():
    """'ab' is a sequence of two characters, so a two-dataset plot would
    silently accept it as one label per dataset -- and a three-dataset plot
    would reject it with a length message that reads as nonsense."""
    with pytest.raises(TypeError, match='forecast_hue'):
        hyp.plot(_walks(2), '-', predict='Kalman', t=4, forecast_hue='ab',
                 show=False)


def test_missing_forecast_hue_values_form_ONE_unlabeled_group():
    """`nan != nan`, so two missing labels are not equal and would come out
    as two separate groups in two separate colours (and, since `float('nan')`
    is a fresh object each time but `np.nan` is a singleton, WHICH of those
    happened would depend on how the caller spelled it).

    They are normalized to the `None` the observed categorical hue already
    uses for "unlabeled", so a missing forecast label means the same thing as
    a missing observed one: one group, drawn in neutral gray."""
    from hypertools.plot.plot import _UNLABELED_HUE_COLOR
    data = _walks(4)
    fig = hyp.plot(data, '-', predict='Kalman', t=4,
                   forecast_hue=['a', float('nan'), 'b', float('nan')],
                   show=False)
    fc = _by_dataset(fig)
    assert np.allclose(_rgb(fc[1]), _rgb(fc[3])), (
        'two missing labels came out as two different groups')
    assert np.allclose(_rgb(fc[1]), _UNLABELED_HUE_COLOR)
    assert not np.allclose(_rgb(fc[0]), _rgb(fc[2]))


def test_a_missing_forecast_hue_label_does_not_consume_a_palette_slot():
    """The named categories keep the first palette slots, exactly as they do
    for an observed `hue=` containing `None` (release-1.0 audit F02-013)."""
    data = _walks(3)
    named = hyp.plot(data, '-', predict='Kalman', t=4,
                     forecast_hue=['a', 'b', 'b'],
                     forecast_palette=['#ff0000', '#0000ff'], show=False)
    partial = hyp.plot(data, '-', predict='Kalman', t=4,
                       forecast_hue=['a', None, 'b'],
                       forecast_palette=['#ff0000', '#0000ff'], show=False)
    assert np.allclose(_rgb(_by_dataset(partial)[0]), [1, 0, 0])
    assert np.allclose(_rgb(_by_dataset(partial)[2]), [0, 0, 1])
    assert np.allclose(_rgb(_by_dataset(named)[0]), [1, 0, 0])


# -- resolver-level guards. These inputs cannot come through `plot()`, which
# always hands over real forecasts -- but the resolver is importable on its
# own, and an internal function that raises IndexError from inside numpy
# tells its next caller nothing.

def test_forecast_cluster_without_forecasts_says_what_is_missing():
    from hypertools.plot.forecast import resolve_forecast_overrides
    with pytest.raises(ValueError, match='forecast_cluster'):
        resolve_forecast_overrides(3, None, cluster='KMeans')


def test_forecast_cluster_on_an_empty_forecast_says_which_one():
    from hypertools.plot.forecast import resolve_forecast_overrides
    forecasts = [np.zeros((4, 3)), np.zeros((0, 3)), np.zeros((4, 3))]
    with pytest.raises(ValueError, match='no rows'):
        resolve_forecast_overrides(3, forecasts, cluster='KMeans')


def test_forecast_cluster_on_ragged_forecasts_says_so():
    """`np.vstack` reports "all the input array dimensions except for the
    concatenation axis must match exactly", which never mentions forecasts."""
    from hypertools.plot.forecast import resolve_forecast_overrides
    forecasts = [np.zeros((4, 3)), np.zeros((4, 2))]
    with pytest.raises(ValueError, match='same number of dimensions'):
        resolve_forecast_overrides(2, forecasts, cluster='KMeans')


def test_forecast_cluster_on_non_finite_endpoints_says_which_datasets():
    from hypertools.plot.forecast import resolve_forecast_overrides
    forecasts = [np.zeros((4, 3)), np.full((4, 3), np.nan),
                 np.zeros((4, 3))]
    with pytest.raises(ValueError, match='non-finite'):
        resolve_forecast_overrides(3, forecasts, cluster='KMeans')


# --------------------------------------------------------------------------
# plotly parity
# --------------------------------------------------------------------------

def _ply_by_dataset(fig):
    return {(tr.meta or {})['hyp_dataset']: tr for tr in fig.data
            if (tr.meta or {}).get('hyp_forecast_role') == 'static'}


def _ply_rgb(trace):
    body = trace.line.color.split('(', 1)[1].rsplit(')', 1)[0]
    return np.asarray([float(v) for v in body.split(',')[:3]]) / 255.0


def test_plotly_applies_the_same_forecast_palette():
    data = _walks(3)
    with hyp.set_interactive_backend('plotly'):
        fig = hyp.plot(data, '-', predict='Kalman', t=4,
                       forecast_palette=['#ff0000', '#00ff00', '#0000ff'],
                       show=False)
    fc = _ply_by_dataset(fig)
    assert np.allclose(_ply_rgb(fc[0]), [1, 0, 0], atol=0.01)
    assert np.allclose(_ply_rgb(fc[1]), [0, 1, 0], atol=0.01)
    assert np.allclose(_ply_rgb(fc[2]), [0, 0, 1], atol=0.01)


def test_plotly_applies_the_same_forecast_fmt():
    data = _walks(2)
    with hyp.set_interactive_backend('plotly'):
        fig = hyp.plot(data, '-', predict='Kalman', t=4, forecast_fmt=':',
                       show=False)
    for tr in _ply_by_dataset(fig).values():
        assert tr.line.dash == 'dot', (
            f'plotly forecast dash is {tr.line.dash!r}, not dotted')


def test_forecast_fmt_beats_a_linestyle_kwarg_on_BOTH_backends():
    """`linestyle=` styles the observed data; `forecast_fmt=` is aimed at the
    forecasts specifically, so it wins for them.

    The backends disagreed here: matplotlib applies the override last, but
    plotly resolves fmt strings through `_resolve_fmt`, where an explicit
    style kwarg beats the fmt -- correct for the OBSERVED trace's own fmt,
    wrong for an override that exists to overrule exactly that."""
    data = _walks(2)
    fig = hyp.plot(data, predict='Kalman', t=4, linestyle='--',
                   forecast_fmt=':', show=False)
    assert all(f.get_linestyle() in (':', 'dotted') for f in _forecasts(fig))

    with hyp.set_interactive_backend('plotly'):
        ply = hyp.plot(data, predict='Kalman', t=4, linestyle='--',
                       forecast_fmt=':', show=False)
    dashes = [tr.line.dash for tr in _ply_by_dataset(ply).values()]
    assert dashes == ['dot'] * len(data), (
        f'plotly forecast dashes {dashes}; forecast_fmt=":" was overruled by '
        f'the observed linestyle= kwarg')


def test_the_ANIMATED_live_and_trail_artists_take_the_override_too():
    """Six construction sites draw forecasts -- static, live and trail, on
    each backend -- and a style that reached only some of them would make a
    paused animation look unlike the static plot of the same data."""
    data = _walks(3)
    kw = dict(predict='Kalman', t=4, animate=True, duration=2, frame_rate=4,
              forecast_palette=['#ff0000', '#00ff00', '#0000ff'],
              forecast_fmt=':', forecast_trail=2, show=False)
    expected = [(1., 0., 0.), (0., 1., 0.), (0., 0., 1.)]

    fig, ani = hyp.plot(data, '-', **kw)
    ani._func(6, *ani._args)
    for role in ('live', 'trail'):
        artists = [ln for ln in fig.axes[0].lines
                   if getattr(ln, '_hyp_forecast_role', None) == role]
        assert artists, f'no {role} artists to check'
        for ln in artists:
            ds = ln._hyp_forecast_dataset
            assert np.allclose(_rgb(ln), expected[ds]), f'{role} {ds}'
            assert ln.get_linestyle() in (':', 'dotted')

    with hyp.set_interactive_backend('plotly'):
        ply = hyp.plot(data, '-', **kw)
    traces = [tr for tr in ply.data
              if (tr.meta or {}).get('hyp_forecast_role') in ('live', 'trail')]
    assert traces
    for tr in traces:
        ds = tr.meta['hyp_dataset']
        assert np.allclose(_ply_rgb(tr), expected[ds], atol=0.01)
        assert tr.line.dash == 'dot'


def test_both_backends_agree_on_forecast_cluster_colours():
    data = _converging()
    mpl_fig = hyp.plot(data, '-', predict='Kalman', t=6,
                       forecast_cluster='KMeans', forecast_n_clusters=2,
                       show=False)
    with hyp.set_interactive_backend('plotly'):
        ply_fig = hyp.plot(data, '-', predict='Kalman', t=6,
                           forecast_cluster='KMeans', forecast_n_clusters=2,
                           show=False)
    mpl = _by_dataset(mpl_fig)
    ply = _ply_by_dataset(ply_fig)
    assert set(mpl) == set(ply) == set(range(len(data)))
    for i in mpl:
        assert np.allclose(_rgb(mpl[i]), _ply_rgb(ply[i]), atol=0.01), (
            f'dataset {i}: matplotlib {_rgb(mpl[i])} vs plotly '
            f'{_ply_rgb(ply[i])}')
