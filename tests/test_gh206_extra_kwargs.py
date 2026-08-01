# -*- coding: utf-8 -*-
"""GH #206 (feature, partial): (a) `plot()` accepts arbitrary extra
matplotlib-style kwargs (merged into `mpl_kwargs` AFTER the named ones, so
an explicit named kwarg always wins; the plotly backend applies the subset
it can map and WARNS listing anything it can't); (b) mismatched-length
list kwargs (`color=`, `marker=`, `linestyle=`, or any extra `**kwargs`
list) now RAISE a clear ``ValueError`` naming the kwarg, its length, and
the dataset count, instead of silently degrading to `None` for every
dataset.

Real `hyp.plot()` calls only (no mocks): every assertion inspects actual
matplotlib artists / plotly figures, or an actually-raised exception/
warning.
"""
import warnings

import numpy as np
import pytest
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import hypertools as hyp
from hypertools._shared.helpers import parse_kwargs


def _two_datasets(seed=0):
    rng = np.random.default_rng(seed)
    return [rng.standard_normal((20, 3)) for _ in range(2)]


class TestExtraKwargsPassthroughMatplotlib:
    def test_zorder_reaches_line_artists(self):
        data = _two_datasets(0)
        fig = hyp.plot(data, zorder=3, show=False)
        lines = fig.axes[0].get_lines()
        plt.close('all')

        assert len(lines) == 2
        assert all(l.get_zorder() == 3 for l in lines)

    def test_dashes_reaches_line_artists(self):
        # NOTE: a 4-element dash pattern (not 2) is used deliberately --
        # `parse_kwargs`' list/tuple broadcasting (shared by EVERY
        # list/tuple-valued kwarg hypertools supports: color=/marker=/
        # linestyle=/chemtrails=/surface=/etc.) can only distinguish "one
        # value per dataset" from "a single tuple-shaped VALUE" by length:
        # a tuple whose length happens to equal the dataset count is
        # (correctly, per that shared, pre-existing, and heavily-relied-
        # upon convention) treated as one-value-per-dataset. Since this
        # test uses 2 datasets, a 2-element dashes tuple would collide
        # with that convention; a 4-element one does not.
        data = _two_datasets(1)
        fig = hyp.plot(data, dashes=(4, 2, 4, 2), show=False)
        lines = fig.axes[0].get_lines()
        plt.close('all')

        assert len(lines) == 2
        for l in lines:
            # matplotlib has no public dashes GETTER (only `set_dashes`);
            # `_dash_pattern` is `(offset, [on, off, on, off, ...])`,
            # scaled by the artist's linewidth -- checking its ratio
            # (rather than absolute values) confirms the (4, 2, 4, 2)
            # pattern reached the artist without depending on matplotlib's
            # internal scaling factor.
            on1, off1, on2, off2 = l._dash_pattern[1]
            assert on1 == pytest.approx(on2)
            assert off1 == pytest.approx(off2)
            assert on1 == pytest.approx(2 * off1)

    def test_alpha_kwarg_reaches_line_artists(self):
        # alpha= became a first-class named parameter in 1.1 (see
        # plot()'s `alpha` and `_validate_alpha` in hypertools/plot/
        # plot.py) rather than a generic **kwargs passthrough value; this
        # test now guards the scalar case via that named-parameter path
        # (see tests/plot/test_per_dataset_alpha.py for the per-dataset
        # list form and its precedence rules).
        data = _two_datasets(2)
        fig = hyp.plot(data, alpha=0.42, show=False)
        lines = fig.axes[0].get_lines()
        plt.close('all')

        assert all(l.get_alpha() == pytest.approx(0.42) for l in lines)

    def test_named_kwarg_wins_over_extra_kwarg_collision(self):
        """linewidth= (a named parameter) must win over anything a
        pathological extra kwarg might otherwise imply -- sanity check
        that the merge order (kwargs first, named mpl_kwargs second)
        actually gives named params priority."""
        data = _two_datasets(3)
        fig = hyp.plot(data, linewidth=5, show=False)
        lines = fig.axes[0].get_lines()
        plt.close('all')
        assert all(l.get_linewidth() == 5 for l in lines)

    def test_unknown_kwarg_surfaces_matplotlib_error(self):
        """Unknown kwargs that matplotlib itself rejects surface
        matplotlib's OWN error (no separate hypertools-side whitelist)."""
        data = _two_datasets(4)
        with pytest.raises(Exception):
            hyp.plot(data, not_a_real_kwarg_xyz=1, show=False)
        plt.close('all')


class TestExtraKwargsPlotlyBackend:
    def test_mappable_extra_kwarg_applied_no_warning(self):
        data = _two_datasets(5)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            fig = hyp.plot(data, alpha=0.3, backend='plotly', show=False)
        plotly_warnings = [
            x for x in w if 'cannot map' in str(x.message)]
        assert not plotly_warnings
        assert fig is not None

    def test_unmappable_extra_kwarg_warns_and_names_it(self):
        data = _two_datasets(6)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            hyp.plot(data, zorder=5, backend='plotly', show=False)
        messages = [str(x.message) for x in w]
        matches = [m for m in messages if 'cannot map' in m]
        assert len(matches) == 1
        assert 'zorder' in matches[0]

    def test_multiple_unmappable_kwargs_all_named_in_one_warning(self):
        data = _two_datasets(7)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            hyp.plot(data, zorder=5, gid='abc', backend='plotly',
                     show=False)
        messages = [str(x.message) for x in w]
        matches = [m for m in messages if 'cannot map' in m]
        assert len(matches) == 1
        assert 'zorder' in matches[0]
        assert 'gid' in matches[0]


class TestMismatchedLengthListKwargsRaise:
    def test_color_list_wrong_length_raises_value_error(self):
        data = _two_datasets(8)
        with pytest.raises(ValueError, match=r'color.*3.*2'):
            hyp.plot(data, color=['red', 'blue', 'green'], show=False)
        plt.close('all')

    def test_marker_list_wrong_length_raises_value_error(self):
        data = _two_datasets(9)
        with pytest.raises(ValueError, match=r'marker'):
            hyp.plot(data, marker=['o', 's', '^'], show=False)
        plt.close('all')

    def test_linestyle_list_wrong_length_raises_value_error(self):
        data = _two_datasets(10)
        with pytest.raises(ValueError, match=r'linestyle'):
            hyp.plot(data, linestyle=['-', '--', ':'], show=False)
        plt.close('all')

    def test_extra_kwarg_is_never_broadcast_per_dataset(self):
        """Generic `**kwargs` passthrough (GH #206) is verbatim/scalar
        ONLY -- unlike `color=`/`marker=`/`linestyle=`, an extra kwarg's
        list/tuple value is never treated as "one entry per dataset" (see
        `_apply_extra_kwargs`'s docstring), so a mismatched-length list
        here does NOT raise the length-mismatch error the dedicated
        per-dataset kwargs above do; the identical list is instead applied
        as one verbatim value to every dataset's artist."""
        data = _two_datasets(11)
        fig = hyp.plot(data, gid=['a', 'b', 'c'], show=False)
        lines = fig.axes[0].get_lines()
        plt.close('all')
        assert len(lines) == 2
        assert all(l.get_gid() == ['a', 'b', 'c'] for l in lines)

    def test_correct_length_list_still_works(self):
        """Matching-length lists must be unaffected by the fix."""
        data = _two_datasets(12)
        fig = hyp.plot(data, color=['red', 'blue'], show=False)
        lines = fig.axes[0].get_lines()
        plt.close('all')
        assert len(lines) == 2

    def test_error_names_kwarg_length_and_dataset_count(self):
        data = _two_datasets(13)
        with pytest.raises(ValueError) as excinfo:
            hyp.plot(data, color=['red', 'blue', 'green', 'purple'],
                     show=False)
        msg = str(excinfo.value)
        assert 'color' in msg
        assert '4' in msg  # length given
        assert '2' in msg  # dataset count required
        plt.close('all')


class TestParseKwargsHelperDirectly:
    """Unit tests on the fixed `parse_kwargs` helper itself."""

    def test_matching_length_list_broadcasts_per_dataset(self):
        x = [np.zeros((3, 3)), np.zeros((3, 3))]
        result = parse_kwargs(x, {'color': ['red', 'blue']})
        assert result == [{'color': 'red'}, {'color': 'blue'}]

    def test_scalar_broadcasts_to_every_dataset(self):
        x = [np.zeros((3, 3)), np.zeros((3, 3))]
        result = parse_kwargs(x, {'linewidth': 2})
        assert result == [{'linewidth': 2}, {'linewidth': 2}]

    def test_mismatched_length_raises_value_error(self):
        x = [np.zeros((3, 3)), np.zeros((3, 3))]
        with pytest.raises(ValueError, match=r'markersize.*3.*2'):
            parse_kwargs(x, {'markersize': [1, 2, 3]})
