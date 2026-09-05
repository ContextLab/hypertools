# -*- coding: utf-8 -*-
"""hyp.load() gains three explicit-prefix WEB sources (GH #285):

- ``'wikipedia:<Title>'`` -- an article's plain-text extract from the
  MediaWiki API (``'wikipedia:A|B'`` and a list of names both give a list)
- ``'yahoo:<TICKER>'`` -- daily OHLCV bars from Yahoo Finance's v8 chart
  endpoint, always requested with EXPLICIT period1/period2 epoch bounds
  (``range=max&interval=1d`` silently degrades to 3-month bars)
- ``'sec:<TICKER>'`` -- one XBRL concept's reported values from the SEC,
  with the ``companyfacts`` fallback for filers whose ``companyconcept``
  endpoint comes back empty

Like ``'fivethirtyeight/'`` and ``'kaggle/'``, these prefixes are
unambiguous: a matching-but-failing name raises instead of falling through
the rest of the resolution chain.

All tests are REAL network calls to the real endpoints -- no mocks, no
recorded fixtures -- wrapped in `skip_on_transient_network` so an outage
skips while a genuine regression (a moved endpoint, a changed payload
shape, a rejected User-Agent) still fails.
"""

import numpy as np
import pandas as pd
import pytest

import matplotlib
matplotlib.use('Agg')

import hypertools as hyp                                       # noqa: E402
from hypertools._shared.exceptions import HypertoolsIOError     # noqa: E402
from hypertools.io.sources import (WEB_SOURCE_PREFIXES,         # noqa: E402
                                   _contact_user_agent,
                                   is_loadable_string,
                                   sec_source, web_source,
                                   wikipedia_source, yahoo_source)
from tests._netskip import skip_on_transient_network            # noqa: E402


# --------------------------------------------------------------- plumbing

def test_contact_user_agent_carries_the_project_email_and_no_url():
    ua = _contact_user_agent()['User-Agent']
    assert ua.startswith('hypertools/')
    assert '@' in ua                      # the SEC requires a contact
    # measured 2026-09-05: sec.gov's WAF answers 403 to an otherwise
    # identical User-Agent containing a URL, so the string must stay
    # contact-only. This assertion is the regression guard.
    assert 'http' not in ua


def test_prefixes_are_recognized_without_touching_the_network():
    assert WEB_SOURCE_PREFIXES == ('wikipedia:', 'yahoo:', 'sec:')
    assert web_source('not-a-web-source') is None       # "not mine"
    assert is_loadable_string('wikipedia:Dartmouth College')
    assert is_loadable_string('yahoo:AAPL')
    assert is_loadable_string('sec:AAPL')
    # ordinary prose is still not a source name
    assert not is_loadable_string('wikipedia: the free encyclopedia')


@pytest.mark.parametrize('name', ['wikipedia:', 'yahoo:  ', 'sec: '])
def test_prefix_without_a_target_raises(name):
    with pytest.raises(HypertoolsIOError):
        hyp.load(name)


# -------------------------------------------------------------- wikipedia

def test_wikipedia_returns_the_article_text():
    with skip_on_transient_network('loading wikipedia:Dartmouth College'):
        text = hyp.load('wikipedia:Dartmouth College')
    assert isinstance(text, str)
    assert len(text) > 2000
    assert 'Dartmouth' in text
    assert '<' not in text[:200]          # plain text, not wikitext/HTML


def test_wikipedia_underscores_and_redirects_resolve():
    with skip_on_transient_network('loading wikipedia:Dartmouth_College'):
        underscored = hyp.load('wikipedia:Dartmouth_College')
    assert 'Dartmouth' in underscored


def test_wikipedia_intro_is_a_prefix_of_the_whole_article():
    with skip_on_transient_network('loading wikipedia:Python (programming '
                                   'language)'):
        full = wikipedia_source('wikipedia:Python (programming language)')
        intro = wikipedia_source('wikipedia:Python (programming language)',
                                 intro=True)
    assert 0 < len(intro) < len(full)
    assert full.startswith(intro[:200])


def test_wikipedia_multiple_titles_return_a_list():
    with skip_on_transient_network('loading wikipedia:Physics|Chemistry'):
        piped = hyp.load('wikipedia:Physics|Chemistry')
    assert isinstance(piped, list) and len(piped) == 2
    assert all(isinstance(t, str) and len(t) > 1000 for t in piped)
    assert 'physics' in piped[0].lower()
    assert 'chemistry' in piped[1].lower()

    # a list of names resolves element-wise to the same thing
    with skip_on_transient_network('loading a list of wikipedia names'):
        listed = hyp.load(['wikipedia:Physics', 'wikipedia:Chemistry'])
    assert [len(t) for t in listed] == [len(t) for t in piped]


def test_wikipedia_missing_page_raises_immediately():
    with pytest.raises(HypertoolsIOError) as excinfo:
        with skip_on_transient_network('loading a missing wikipedia page'):
            hyp.load('wikipedia:Zzzznotarealpagexyz123')
    message = str(excinfo.value)
    assert 'Zzzznotarealpagexyz123' in message
    # explicit prefix -> raises directly, no "tried, in order" fall-through
    assert 'Tried, in order' not in message


def test_wikipedia_text_flows_into_text2mat():
    with skip_on_transient_network('loading wikipedia:Physics'):
        text = hyp.load('wikipedia:Physics')
    reduced = hyp.tools.text2mat([text[:5000]], vectorizer='CountVectorizer',
                                 semantic=None, corpus=None)
    assert len(reduced) == 1
    assert np.asarray(reduced[0]).shape[0] == 1


# ------------------------------------------------------------------ yahoo

def test_yahoo_returns_daily_ohlcv_indexed_by_date():
    with skip_on_transient_network('loading yahoo:AAPL'):
        df = yahoo_source('yahoo:AAPL', start='2024-01-01', end='2024-02-01')
    assert isinstance(df, pd.DataFrame)
    assert isinstance(df.index, pd.DatetimeIndex)
    assert df.index.name == 'date'
    assert df.index.is_monotonic_increasing
    for col in ('open', 'high', 'low', 'close', 'volume'):
        assert col in df.columns
        assert df[col].dtype.kind == 'f'
    assert 15 <= len(df) <= 23                    # ~21 trading days
    assert (df['high'] >= df['low']).all()
    assert (df['volume'] > 0).all()


def test_yahoo_multiyear_window_returns_daily_bars_not_3_month_ones():
    # the regression this source exists to prevent: 'range=max&interval=1d'
    # silently returns 3-MONTH bars (measured 2026-09-03, AAPL: 169 rows
    # since 1984), while explicit period1/period2 bounds return daily ones
    with skip_on_transient_network('loading yahoo:MSFT over 3 years'):
        df = yahoo_source('yahoo:MSFT', start='2020-01-01',
                          end='2023-01-01')
    assert len(df) > 700                          # ~252 trading days a year
    gaps = df.index.to_series().diff().dropna().dt.days
    assert gaps.median() <= 3                     # daily bars, not quarterly


def test_yahoo_accepts_epoch_and_datetime_bounds():
    with skip_on_transient_network('loading yahoo:AAPL by epoch bounds'):
        by_epoch = yahoo_source('yahoo:AAPL', start=1704067200,
                                end=1706745600)
        by_stamp = yahoo_source('yahoo:AAPL',
                                start=pd.Timestamp('2024-01-01'),
                                end=pd.Timestamp('2024-02-01'))
    pd.testing.assert_frame_equal(by_epoch, by_stamp)


def test_yahoo_full_history_through_hyp_load():
    with skip_on_transient_network('loading yahoo:MSFT full history'):
        df = hyp.load('yahoo:MSFT')
    assert df.index[0].year < 1990                # MSFT listed in 1986
    assert len(df) > 8000


def test_yahoo_unknown_ticker_raises_immediately():
    with pytest.raises(HypertoolsIOError) as excinfo:
        with skip_on_transient_network('loading an unknown yahoo ticker'):
            hyp.load('yahoo:ZZZZNOTAREALTICKER')
    message = str(excinfo.value)
    assert 'ZZZZNOTAREALTICKER' in message
    assert 'Tried, in order' not in message


# -------------------------------------------------------------------- sec

def test_sec_returns_reported_facts_indexed_by_period_end():
    with skip_on_transient_network('loading sec:AAPL'):
        df = hyp.load('sec:AAPL')
    assert isinstance(df, pd.DataFrame)
    assert isinstance(df.index, pd.DatetimeIndex)
    assert df.index.name == 'end'
    assert df.index.is_monotonic_increasing
    assert not df.index.duplicated().any()        # dedupe=True by default
    assert df['value'].dtype.kind in 'if'         # numeric, as filed
    assert df['value'].max() > 1e9                # Apple's share count
    assert set(df['unit']) == {'shares'}
    assert 'form' in df.columns and 'filed' in df.columns


def test_sec_falls_back_to_companyfacts_when_companyconcept_is_empty():
    # measured 2026-09-03: KO's companyconcept endpoint returns an EMPTY
    # units dict for this concept while its companyfacts file carries it
    with skip_on_transient_network('loading sec:KO'):
        df = sec_source('sec:KO')
    assert len(df) > 10
    # one KO filing reports 0 shares (2009-10-23) -- real, as-filed data
    assert (df['value'] >= 0).all()
    assert df['value'].max() > 1e9


def test_sec_dedupe_false_keeps_amended_filings():
    with skip_on_transient_network('loading sec:AAPL undeduplicated'):
        deduped = sec_source('sec:AAPL')
        raw = sec_source('sec:AAPL', dedupe=False)
    assert len(raw) >= len(deduped)
    assert set(deduped.index).issubset(set(raw.index))


def test_sec_other_taxonomy_and_concept():
    with skip_on_transient_network('loading a us-gaap concept for AAPL'):
        df = sec_source('sec:AAPL', concept='Assets', taxonomy='us-gaap')
    assert len(df) > 5
    assert set(df['unit']) == {'USD'}
    assert df['value'].max() > 1e11


def test_sec_unknown_ticker_and_concept_raise_immediately():
    with pytest.raises(HypertoolsIOError) as excinfo:
        with skip_on_transient_network('loading an unknown SEC ticker'):
            hyp.load('sec:ZZZZNOTAREALTICKER')
    assert 'ZZZZNOTAREALTICKER' in str(excinfo.value)

    with pytest.raises(HypertoolsIOError) as excinfo:
        with skip_on_transient_network('loading an unknown SEC concept'):
            sec_source('sec:AAPL', concept='NotARealConceptXyz')
    assert 'NotARealConceptXyz' in str(excinfo.value)
