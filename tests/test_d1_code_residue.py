# -*- coding: utf-8 -*-
"""Regression tests for the release-1.0 audit's final code-residue wave
(batch D1-code-residue, 2026-07).

Covers: streaming save-format routing and exact stream_max consumption, the
plot(reduce=None) ndims false-positive warning, lsl_stream input validation,
reduce/describe dimensionality guardrails, autoencoder hyperparameter
validation, predict/impute case-insensitive model names, the all-NaN-column
impute warning, text-input misuse errors, missing_inds' empty-array return,
the model= own-stage kwarg alias, morph_samples validation, the explore=True
headless warning, and the divide-by-zero rescale guard.

All tests use real function calls (no mocks), per the project testing
policy.
"""
import shutil
import warnings

import matplotlib

matplotlib.use('Agg')

import numpy as np
import pandas as pd
import pytest

import hypertools as hyp


RNG = np.random.default_rng(17)


# ------------------------------------------------------------ streaming

def _walk(n, d=4):
    rng = np.random.default_rng(3)
    p = np.zeros(d)
    for _ in range(n):
        p = p + 0.1 * rng.standard_normal(d)
        yield p


def test_stream_max_consumes_exactly_stream_max_samples():
    """D09-tutorials-applied-006: the truncation peek must not pull a
    sample beyond stream_max."""
    pulls = [0]

    def counting():
        rng = np.random.default_rng(0)
        while True:
            pulls[0] += 1
            yield rng.standard_normal(4)

    fig = hyp.plot(counting(), stream_init=20, stream_chunk=10,
                   stream_max=40, show=False)
    assert pulls[0] == 40
    assert fig.stream_info['n_samples'] == 40
    # stopping at the cap counts as truncated (no extra peek is made)
    assert fig.stream_info['truncated']


@pytest.mark.skipif(shutil.which('ffmpeg') is None,
                    reason='ffmpeg not installed')
def test_stream_save_path_mp4_routes_to_ffmpeg(tmp_path):
    """D09-tutorials-applied-009: streaming save_path='...mp4' must write a
    video via the ffmpeg writer instead of crashing inside PIL."""
    out = tmp_path / 'stream.mp4'
    # the random walk deliberately drifts past the head-fitted display box,
    # provoking the clamped-samples notice
    with pytest.warns(RuntimeWarning, match='outside the display box'):
        hyp.plot(_walk(200), stream_init=50, stream_chunk=50, stream_max=200,
                 save_path=str(out), show=False)
    assert out.exists() and out.stat().st_size > 0


def test_stream_save_path_bad_extension_fails_before_consuming():
    """D09-tutorials-applied-009 (companion): unknown extensions raise a
    clear ValueError BEFORE any samples are pulled from the stream."""
    pulls = [0]

    def counting():
        while True:
            pulls[0] += 1
            yield RNG.standard_normal(4)

    with pytest.raises(ValueError, match='unsupported streaming save'):
        hyp.plot(counting(), stream_init=20, stream_max=40,
                 save_path='stream.xyz', show=False)
    assert pulls[0] == 0


def test_stream_and_reduce_none_plot_emit_no_ndims_warning():
    """D1 regression: plot(reduce=None) -- including the internal streaming
    redraw -- must not trip analyze's 'ndims= was passed but reduce= is
    None' warning (plot's own ndims default is 3)."""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        hyp.plot(RNG.standard_normal((20, 3)), reduce=None, show=False)
        hyp.plot(_walk(60), stream_init=30, stream_chunk=10, stream_max=60,
                 show=False)
    assert not any('reduce= is None' in str(x.message) for x in w)


def test_lsl_stream_validates_name_and_type():
    """D10-tutorials-embeddings-lsl-013: non-string name=/type= raise a
    TypeError naming the parameter instead of leaking pylsl internals."""
    pytest.importorskip('pylsl')
    with pytest.raises(TypeError, match='name='):
        hyp.io.lsl_stream(name=123, timeout=0.5)
    with pytest.raises(TypeError, match='type='):
        hyp.io.lsl_stream(type=5, timeout=0.5)
    with pytest.raises(ValueError, match='timeout='):
        hyp.io.lsl_stream(timeout=0)


# ------------------------------------------------------- reduce/describe

def test_reduce_single_row_warning_is_grammatical_and_no_divide_warning():
    """D01-readme-013 / D08-tutorials-analysis-015 + C2: single-row input
    warns with readable text, and the display rescale emits no
    divide-by-zero RuntimeWarning."""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        hyp.plot(np.ones((1, 5)), show=False)
    msgs = [str(x.message) for x in w]
    assert any('Cannot reduce a single observation' in m for m in msgs)
    assert not any('Return zeros length of ndims' in m for m in msgs)
    assert not any(issubclass(x.category, RuntimeWarning) for x in w)


def test_reduce_warns_when_ndims_exceeds_features():
    """D04-gallery-models-008: requesting more dimensions than the data has
    features warns instead of silently returning the input unchanged."""
    data = RNG.standard_normal((30, 5))
    with pytest.warns(UserWarning, match='no reduction was performed'):
        out = np.asarray(hyp.reduce(data, reduce='TSNE', ndims=50))
    assert np.allclose(out, data)


def test_reduce_ndims_none_returns_none_model_documented_noop():
    """D14-docs-drift-009: ndims=None is a documented no-op; the model slot
    is None (and the data comes back unchanged) without warnings."""
    x = RNG.standard_normal((50, 8))
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        r, m = hyp.reduce(x, 'PCA', return_model=True)
    assert m is None
    assert np.allclose(np.asarray(r), x)
    assert not w


def test_describe_clamps_max_dims_beyond_data_dimensionality():
    """D14-docs-drift-015: max_dims beyond min(n_obs, n_features) warns and
    the evaluated component range is clamped."""
    d = [RNG.standard_normal((60, 8)) + i for i in range(3)]
    with pytest.warns(UserWarning, match='exceeds the data dimensionality'):
        result = hyp.describe(d, reduce='PCA', max_dims=14, show=False)
    # components 2..8 evaluated (cap = min(180, 8) + 1 = 9) -> 7 values
    assert len(result['average']) == 7


def test_describe_show_true_no_agg_warning():
    """X4-warnings-006 (C2): describe(show=True) under Agg must not emit
    matplotlib's 'FigureCanvasAgg is non-interactive' warning."""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        result = hyp.describe([RNG.standard_normal((40, 5))], reduce='PCA',
                              max_dims=4, show=True)
    assert result['fig'] is not None
    assert not any('non-interactive' in str(x.message) for x in w)


def test_autoencoder_rejects_invalid_hyperparameters():
    """D04-gallery-models-010: invalid epochs/lr raise instead of returning
    an untrained embedding (epochs=0 stays allowed as the documented
    untrained baseline)."""
    pytest.importorskip('torch')
    data = RNG.standard_normal((30, 5))
    with pytest.raises(ValueError, match='epochs'):
        hyp.reduce(data, reduce={'model': 'Autoencoder',
                                 'kwargs': {'epochs': -5}}, ndims=2)
    with pytest.raises(ValueError, match='lr'):
        hyp.reduce(data, reduce={'model': 'Autoencoder',
                                 'kwargs': {'lr': -1.0}}, ndims=2)
    with pytest.raises(ValueError, match='batch_size'):
        hyp.reduce(data, reduce={'model': 'Autoencoder',
                                 'kwargs': {'batch_size': 0}}, ndims=2)


# -------------------------------------------------------- predict/impute

def test_predict_and_impute_model_names_case_insensitive():
    """D09-tutorials-applied-014: documented model names resolve
    case-insensitively; unknown names still error clearly."""
    fc = hyp.predict(pd.DataFrame({'a': np.arange(30.)}), model='kalman',
                     t=2)
    assert fc.shape == (2, 1)
    df = pd.DataFrame(RNG.standard_normal((40, 5)))
    df.iloc[3, 1] = np.nan
    out = hyp.impute(df, model='ppca')
    assert not np.isnan(out.to_numpy()).any()
    with pytest.raises(ValueError, match="unknown predict model 'kalmann'"):
        hyp.predict(pd.DataFrame({'a': np.arange(30.)}), model='kalmann',
                    t=2)


def test_impute_warns_on_all_nan_column():
    """D09-tutorials-applied-012: a never-observed (all-NaN) column warns
    that its imputed values are uninformed."""
    df = pd.DataFrame(RNG.standard_normal((50, 3)), columns=list('abc'))
    df['c'] = np.nan
    with pytest.warns(UserWarning, match='no observed values'):
        out = hyp.impute(df, model='Kalman')
    assert float(out['c'].abs().max()) == 0.0


# ----------------------------------------------------------------- tools

def test_corpus_scalar_raises_and_typo_warns():
    """D05-gallery-data-text-013(a) + D08-tutorials-analysis-011: corpus=
    validation and hosted-corpus-typo warning."""
    with pytest.raises(ValueError, match='corpus='):
        hyp.plot(['some text', 'more text'], 'o', corpus=12345, show=False)
    with pytest.warns(UserWarning, match='not one of the hosted corpora'):
        hyp.tools.format_data(['some document here', 'another one'],
                              corpus='bogus-name')


def test_mixed_text_numeric_mismatch_names_the_real_problem():
    """D08-tutorials-analysis-012 / D05-gallery-data-text-013(b): the
    format_data warning + plot error explain the text/numeric sample-count
    mismatch instead of a bare numpy concatenation error."""
    with pytest.warns(UserWarning, match='DIFFERENT sample counts'):
        with pytest.raises(ValueError, match='text datasets are embedded'):
            hyp.plot([['text one', 'text two'],
                      RNG.standard_normal((5, 3))], 'o', show=False)


def test_missing_inds_returns_empty_array_when_complete():
    """D05-gallery-data-text-017: no missing data -> empty integer array
    (not None), so x[inds, :] yields an empty selection."""
    x = RNG.standard_normal((40, 6))
    inds = hyp.tools.missing_inds(x)
    assert isinstance(inds, np.ndarray)
    assert inds.shape == (0,)
    assert x[inds, :].shape == (0, 6)
    x2 = x.copy()
    x2[3, 2] = np.nan
    x2[7, 0] = np.nan
    inds2 = hyp.tools.missing_inds(x2)
    assert sorted(inds2.tolist()) == [3, 7]


def test_model_kwarg_alias_for_reduce_cluster_normalize():
    """D05-gallery-data-text-020: model= works as the own-stage spec alias
    in reduce/cluster/normalize (matching manip/impute/predict/align), and
    passing both spellings with different values raises."""
    x = RNG.standard_normal((40, 8))
    r = np.asarray(hyp.reduce(x, model='PCA', ndims=3))
    assert r.shape == (40, 3)
    labels = hyp.cluster(x, model='KMeans', n_clusters=2, random_state=0)
    assert len(labels) == 40
    z = np.asarray(hyp.normalize(x, model='within'))
    assert np.allclose(z.mean(axis=0), 0.0, atol=1e-10)
    with pytest.raises(ValueError, match='cannot pass both'):
        hyp.reduce(x, reduce='UMAP', model='PCA', ndims=3)
    with pytest.raises(ValueError, match='cannot pass both'):
        hyp.cluster(x, cluster='HDBSCAN', model='KMeans')


# ------------------------------------------------------------------ plot

def test_morph_samples_validation():
    """D03-gallery-basics-007: morph_samples=-5 raises a named ValueError
    instead of numpy's 'negative dimensions are not allowed'."""
    a = RNG.standard_normal((20, 4))
    b = RNG.standard_normal((20, 4))
    with pytest.raises(ValueError, match='morph_samples'):
        hyp.plot([a, b], animate='morph', morph_samples=-5, duration=1,
                 frame_rate=5, show=False)


def test_explore_warns_under_non_interactive_backend():
    """D05-gallery-data-text-012(b): explore=True under Agg warns that
    hover labels are unavailable (instead of a silent no-op)."""
    with pytest.warns(UserWarning, match='non-interactive'):
        fig = hyp.plot(RNG.standard_normal((30, 5)), '.', explore=True,
                       show=False)
    assert fig is not None


def test_density_2d_clipped_to_frame_box(tmp_path):
    """D05-gallery-data-text-009: the 2-D KDE glow is clipped to the frame
    square (no haze outside the black frame)."""
    fig = hyp.plot(RNG.standard_normal((3, 2)), '.', density=True,
                   show=False)
    fig.canvas.draw()
    ax = fig.axes[0]
    images = ax.get_images()
    assert images, 'density imshow layer missing'
    # matplotlib special-cases a Rectangle clip patch into `clipbox`
    # (a TransformedBbox), not `clippath`: assert the KDE layer's clipbox
    # matches the [-1, 1] frame square in data coordinates -- narrower
    # than the default axes bbox (limits are [-1.1, 1.1]).
    frame_disp = ax.transData.transform([(-1.0, -1.0), (1.0, 1.0)])
    for im in images:
        assert im.clipbox is not None
        np.testing.assert_allclose(
            np.asarray(im.clipbox.get_points()), frame_disp, atol=1.0)
