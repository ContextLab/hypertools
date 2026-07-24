import numpy as np
import matplotlib as mpl
import matplotlib.backend_bases as bb

from hypertools.plot import plot
from hypertools.reduce.reduce import reduce as reducer


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
