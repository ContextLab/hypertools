"""A marker+line format string (``'s--'``) keeps its marker in the legend.

The matplotlib backend draws such a dataset as two artists: the smoothed
line (which carries the legend label) and a markers-only artist at the raw
sample points (``_nolegend_``). Until the 1.1 release review the legend
handle therefore showed only the dashes. Real figures, rendered pixels.
"""

import io

import matplotlib.pyplot as plt
import pytest

import hypertools as hyp


def _walks():
    return [hyp.load('random_walk', n_samples=60, n_features=8, random_state=s)
            for s in range(3)]


def _ink(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=100)
    buf.seek(0)
    im = plt.imread(buf)
    return int((im[..., :3].sum(-1) < 2.9).sum())


@pytest.mark.parametrize('ndims', [3, 2])
def test_marker_line_fmt_legend_handle_shows_the_marker(ndims):
    fig = hyp.plot(_walks(), ['-', 'o', 's--'], ndims=ndims,
                   names=['walk 0', 'walk 1', 'walk 2'], markersize=4,
                   show=False)
    leg = fig.axes[0].get_legend()
    handles = dict(zip((t.get_text() for t in leg.get_texts()),
                       leg.legend_handles))
    assert handles['walk 2'].get_marker() == 's'
    assert handles['walk 2'].get_linestyle() == '--'
    assert handles['walk 0'].get_marker() in (None, 'None', '')
    assert handles['walk 1'].get_linestyle() == 'None'
    plt.close(fig)


def test_marker_line_fmt_draws_markers_only_at_the_raw_points():
    # the smoothed line must not sprout markers along its interpolated
    # vertices: with the legend removed, switching the line's marker off
    # changes no pixel, while the separate markers-only artist does draw.
    fig = hyp.plot(_walks(), ['-', 'o', 's--'], ndims=2,
                   names=['walk 0', 'walk 1', 'walk 2'], markersize=6,
                   show=False)
    ax = fig.axes[0]
    ax.get_legend().remove()
    lines = {ln.get_label(): ln for ln in ax.get_lines()}
    line = lines['walk 2']
    markers = [ln for ln in ax.get_lines() if ln.get_label() == '_nolegend_'
               and ln.get_marker() == 's']
    assert len(markers) == 1
    assert len(markers[0].get_xdata()) == 60          # raw sample points
    before = _ink(fig)
    line.set_marker('None')
    assert _ink(fig) == before                         # line drew no markers
    markers[0].set_visible(False)
    assert _ink(fig) < before                          # the marker artist did
    plt.close(fig)
