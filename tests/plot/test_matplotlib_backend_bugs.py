import numpy as np
import matplotlib as mpl
import matplotlib.backend_bases as bb
import pytest

import hypertools as hyp
from hypertools.plot import plot
from hypertools.reduce.reduce import reduce as reducer


def _clusterable_datasets(n=2, rows=20, dims=4, seed=0, shift=12.0):
    # two well-separated halves per dataset, so KMeans(n_clusters=2) (or an
    # equivalent hue=) always splits each dataset into exactly 2 contiguous
    # same-category runs -- the shape needed to exercise the
    # `_regroup_categorical_lines` run-bridging path.
    rng = np.random.default_rng(seed)
    half = rows // 2
    out = []
    for _ in range(n):
        first = rng.normal(loc=0.0, scale=0.5, size=(half, dims))
        second = rng.normal(loc=shift, scale=0.5, size=(rows - half, dims))
        out.append(np.vstack([first, second]))
    return out


def test_import_hypertools_does_not_mutate_pdf_fonttype():
    # regression test for GH #259: importing hypertools must not mutate
    # matplotlib's global rcParams as a side effect of import
    before = mpl.rcParams["pdf.fonttype"]
    import hypertools  # noqa: F401
    assert mpl.rcParams["pdf.fonttype"] == before


def test_plot_call_does_not_leak_pdf_fonttype():
    # hypertools sets pdf/ps fonttype=42 (editable vector text) for its OWN
    # saves inside the manage_backend scope, but that scope must restore the
    # user's global rcParams afterward so the setting never leaks (GH #259).
    import hypertools as hyp
    before_pdf = mpl.rcParams["pdf.fonttype"]
    before_ps = mpl.rcParams["ps.fonttype"]
    hyp.plot(np.random.rand(10, 3), show=False)
    assert mpl.rcParams["pdf.fonttype"] == before_pdf
    assert mpl.rcParams["ps.fonttype"] == before_ps


def test_show_false_does_not_leave_figure_registered_in_pyplot():
    # regression test for GH #148: show=False must remove the figure from
    # pyplot's global manager, else Jupyter's flush_figures() displays it
    # anyway. The returned Figure stays valid/savable after close().
    import matplotlib.pyplot as plt
    import hypertools as hyp
    plt.close("all")
    fig = hyp.plot(np.random.rand(10, 3), show=False)
    assert fig.number not in plt.get_fignums()
    # the returned figure is still usable (has its axes/content)
    assert len(fig.axes) >= 1


def test_2d_labeled_plot_button_release_callback_runs_without_error():
    # regression test for GH #223: update_position() used to unconditionally
    # call ax.get_proj() (3D-only) and unpack 4-tuples, crashing on 2D
    # labeled plots after any button_release_event
    data = [np.random.multivariate_normal(np.zeros(4), np.eye(4), size=5)]
    data2d = reducer(data, ndims=2)
    # one label per OBSERVATION (release-1.0 audit: plot() now validates the
    # labels= count; `data2d` is a bare (5, 2) array, so `data2d[0]` is a ROW
    # -- the old expression accidentally made 2 labels for 5 points, which
    # used to be silently accepted)
    labels = [[f"p{i}" for i in range(len(data2d))]]

    fig = plot.plot(data2d, labels=labels, show=False)
    ax = fig.axes[0]
    assert not hasattr(ax, "get_proj")

    event = bb.MouseEvent("button_release_event", fig.canvas, x=10, y=10)
    # should not raise AttributeError (missing get_proj) or ValueError
    # (unpacking a 3-tuple as a 4-tuple)
    fig.canvas.callbacks.process("button_release_event", event)


def test_3d_labeled_plot_button_release_callback_still_works():
    # 3D labeled plots must keep working after the 2D fix
    data = [np.random.multivariate_normal(np.zeros(4), np.eye(4), size=5)]
    data3d = reducer(data, ndims=3)
    labels = [[f"p{i}" for i in range(len(data3d))]]

    fig = plot.plot(data3d, labels=labels, show=False)
    ax = fig.axes[0]
    assert hasattr(ax, "get_proj")

    event = bb.MouseEvent("button_release_event", fig.canvas, x=10, y=10)
    fig.canvas.callbacks.process("button_release_event", event)


# ---------------------------------------------------------------------------
# cluster=/hue= line-format run-bridging vs. per-point labels (1.1
# animation-core follow-up, incidental finding while verifying Task 6):
# `_regroup_categorical_lines` (plot.py) segments a line-format dataset into
# contiguous same-category runs, then bridges consecutive runs of the SAME
# source dataset by duplicating each run's first point onto the PREVIOUS
# run's end (`patch_lines`, hypertools/_shared/helpers.py) so the drawn line
# stays continuous across a colour change. That duplication grew the
# segment's DATA array by one row without growing its parallel LABELS list
# to match, so `labels` permanently under-counted `xform` by one entry per
# bridge point after any cluster=/hue= line-format regrouping. This was
# masked whenever a LATER step happened to rebuild `labels` from scratch for
# an unrelated reason (`_expand_labels`, called only when animation frame-
# gridding or static antialiasing interpolation changes the point count) --
# but animate='morph' never resamples `xform` in plot.py (its own resampling
# happens later, downstream, without touching `xform`/`labels`), and a
# static plot with antialias=False never interpolates either, so both left
# the mismatched, un-rebuilt `labels` to reach `annotate_plot`
# (matplotlib_backend.py), which indexes it once per DATA point and crashed
# with a bare "IndexError: list index out of range" partway through.
# ---------------------------------------------------------------------------

def test_cluster_line_animate_morph_does_not_crash_on_labels():
    # the exact reported repro: cluster= + animate='morph' on a plain (non-
    # nested, no hue, no explicit labels=) line-format dataset list.
    hyp.plot(_clusterable_datasets(n=2, rows=20), '-', cluster='KMeans',
             n_clusters=2, animate='morph', duration=1, show=False)


def test_cluster_line_static_no_antialias_does_not_crash_on_labels():
    # sibling crash found while isolating the bug: a STATIC plot (no
    # animate= at all) with antialias=False takes the same "labels never
    # gets rebuilt" path as animate='morph' does, so it hits the identical
    # IndexError -- this is not actually morph-specific at the root.
    hyp.plot(_clusterable_datasets(n=2, rows=20), '-', cluster='KMeans',
             n_clusters=2, antialias=False, show=False)


def test_hue_line_static_no_antialias_does_not_crash_on_labels():
    # hue= goes through the same `_regroup_categorical_lines` run-bridging
    # machinery as cluster= (hue= is dropped for animate='morph' itself,
    # with a warning, for an unrelated semantic reason -- point identity
    # does not survive a morph -- so this checks the non-morph crash path).
    hue = ([0] * 10 + [1] * 10) * 2
    hyp.plot(_clusterable_datasets(n=2, rows=20), '-', hue=hue,
             antialias=False, show=False)


def test_cluster_line_animate_morph_labels_all_present_and_correct():
    # beyond "does not crash": every real per-point label the caller
    # supplied must actually be drawn, exactly once each, with none lost,
    # duplicated, or silently reassigned to the wrong point as a side
    # effect of the run-bridging fix (mirrors the existing
    # TestAnimationLabels convention in tests/test_plot_audit_b1.py).
    datasets = _clusterable_datasets(n=2, rows=20)
    n_obs = sum(len(d) for d in datasets)
    labels = [f"p{i}" for i in range(n_obs)]
    result = hyp.plot(datasets, '-', cluster='KMeans', n_clusters=2,
                      animate='morph', duration=1, labels=labels, show=False)
    fig = result.figure
    texts = sorted(t.get_text() for ax in fig.axes for t in ax.texts)
    assert texts == sorted(labels)


def test_cluster_line_animate_parallel_labels_land_at_correct_frame():
    # a SECOND, subtler bug uncovered alongside the crash: in the "masked"
    # (non-crashing) animate modes, `_expand_labels` DOES always rebuild
    # `labels` from scratch, so no IndexError was ever raised here -- but it
    # sliced each run's labels using the BRIDGED (one-too-long) length, so
    # every run after the first silently leaked its first real label into
    # the TAIL of the PRECEDING run's frame window instead of the HEAD of
    # its own, once run-bridging (`patch_lines`) was in play. Bridging the
    # labels in lockstep with the data (the same fix as the crash) removes
    # that leakage too. frame_rate defaults to 30 and duration=1, so each
    # of the 4 cluster runs gets exactly 30 frames: dataset 0's second run
    # (starting at its 11th point, "p10") must land at the HEAD of frames
    # [30, 60), and dataset 1's first point ("p20") at the head of [60, 90).
    datasets = _clusterable_datasets(n=2, rows=20)
    n_obs = sum(len(d) for d in datasets)
    labels = [f"p{i}" for i in range(n_obs)]
    result = hyp.plot(datasets, '-', cluster='KMeans', n_clusters=2,
                      animate=True, duration=1, labels=labels, show=False)
    fig = result.figure
    by_text = {t.get_text(): t for ax in fig.axes for t in ax.texts}
    p0_idx = by_text["p0"]._hyp_global_idx
    p10_idx = by_text["p10"]._hyp_global_idx
    p20_idx = by_text["p20"]._hyp_global_idx
    assert p0_idx == 0
    assert p10_idx == pytest.approx(30, abs=3)
    assert p20_idx == pytest.approx(60, abs=3)
