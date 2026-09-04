# -*- coding: utf-8 -*-
"""Does the aligned-composition gate DISCRIMINATE?

`test_examples_are_native.py` gained `_assert_aligned_composition` in round 2
(2026-09-03), replacing the tiled-panel gate when the Market example was
rebuilt as six sectors reduced separately, hyperaligned into one space and
drawn with their mean. The gate's own docstring records what happened the
last time an unexercised gate shipped: `_save_count >= 1` and
`'morph' in 'morph'` were both tautologies that could not fail.

So the gate is checked here against a composition built to satisfy it, and
against mutations that each break exactly one of its claims: the seventh
path is not the mean, the mean is not drawn heavier, the colouring is not a
mixture, the title is not a date, and the title never advances.
"""
import re

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt                                # noqa: E402
import numpy as np                                             # noqa: E402
import pytest                                                  # noqa: E402

import hypertools as hyp                                       # noqa: E402
# `tests/` IS a package (it has an `__init__.py`), so the gate module is
# imported by its dotted name rather than as a bare top-level module.
from tests.test_examples_are_native import (                   # noqa: E402
    _assert_aligned_composition)

N_SECTORS, N_MONTHS = 6, 90
COLOURS = ['#c1272d', '#f2a900', '#1b7f4f', '#2d5fa8', '#7d3c98', '#00808a']
SPEC = dict(sectors=N_SECTORS)
DATE = r'^[A-Z][a-z]+ \d{1,2}, \d{4}$'


def _sectors(seed=0):
    """Six random-walk sector matrices with 4-5 'stocks' each."""
    rng = np.random.default_rng(seed)
    return [np.cumsum(rng.standard_normal((N_MONTHS, 4 + i % 2)), axis=0)
            for i in range(N_SECTORS)]


def _compose(mean_is_mean=True, heavier=True, mixture=True,
             date_title=True, advancing=True):
    """The composition the gate describes, with one claim optionally broken."""
    reduced = [hyp.reduce(s, reduce='PCA', ndims=3) for s in _sectors()]
    aligned = hyp.align(reduced, align='hyper')
    market = np.mean(aligned, axis=0)
    if not mean_is_mean:
        market = market + 0.5 * np.std(market)       # a shifted 'mean'
    weights = np.random.default_rng(1).dirichlet(np.ones(N_SECTORS), N_MONTHS)
    hue = [np.tile(np.eye(N_SECTORS)[i], (N_MONTHS, 1))
           for i in range(N_SECTORS)] + [weights]
    if not mixture:                                  # constant market colour
        hue[-1] = np.tile(np.eye(N_SECTORS)[0], (N_MONTHS, 1))
    widths = [1.0] * N_SECTORS + ([3.0] if heavier else [1.0])
    anim = hyp.plot(aligned + [market], '-', hue=hue, palette=COLOURS,
                    hue_mode='mixture', linewidth=widths, animate=True,
                    chemtrails=True, tail_duration=1, duration=2,
                    frame_rate=15, colorbar=False, show=False,
                    title='January 1, 2000')
    ax = anim.figure.axes[0]

    def title(ctx):
        frac = ctx.frame / max(1, ctx.n_frames - 1)
        month = int(round(frac * (N_MONTHS - 1))) if advancing else 0
        text = (f'{["January", "July"][month % 2]} {1 + month % 28}, '
                f'{2000 + month // 12}' if date_title else f'month {month}')
        ax.set_title(text, color=plt.get_cmap('RdYlGn')(frac))

    anim.on_frame(title)
    return anim


def test_gate_accepts_the_composition_it_describes():
    anim = _compose()
    try:
        _assert_aligned_composition('probe', anim, SPEC)
    finally:
        plt.close(anim.figure)


@pytest.mark.parametrize('mutation, message', [
    (dict(mean_is_mean=False), 'not the mean'),
    (dict(heavier=False), 'not drawn heavier'),
    (dict(mixture=False), 'VARYING colour'),
    (dict(date_title=False), 'are not dates'),
    (dict(advancing=False), 'never advanced'),
])
def test_gate_rejects_each_broken_claim(mutation, message):
    anim = _compose(**mutation)
    try:
        with pytest.raises(AssertionError, match=re.escape(message)):
            _assert_aligned_composition('probe', anim, SPEC)
    finally:
        plt.close(anim.figure)


def test_gate_date_pattern_matches_the_example_format():
    """The pattern the gate uses is the example's own title format."""
    assert re.match(DATE, 'February 3, 2016')
    assert re.match(DATE, 'September 30, 2026')
    assert not re.match(DATE, 'month 3')
    assert not re.match(DATE, '2016-02-03')
