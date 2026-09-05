# -*- coding: utf-8 -*-
"""
=========================================================
A quarter century of the market: six sectors, one space
=========================================================

Twenty-seven large-cap stocks in six sectors, every month since 2000, drawn
as seven paths through one shared 3-D space. Each **sector** is handed to
the library as its own matrix -- months down the rows, that sector's stocks
across the columns (four or five of them; the counts differ on purpose) --
and each cell is the stock's **trailing twelve-month return**. Three library
calls turn that into the figure:

1. ``hyp.reduce`` takes every sector from its own handful of stocks to three
   dimensions **separately**, so a sector is a trajectory in a space made
   of its own stocks, not a projection shared with the others;
2. ``hyp.align(..., model='HyperAlign')`` hyperaligns the six trajectories into
   **one** common space, so that from then on a direction means the same
   thing for Energy as it does for Technology;
3. ``hyp.plot`` draws all six, plus a seventh, heavier path: the **market**,
   the point-by-point mean of the six aligned sector paths.

**Colour is the market's composition.** Every path is coloured through the
library's *mixture* hue: ``hue=`` carries one row of weights per month and
``palette=`` the six sector colours, and each point is drawn in the
weighted blend (``hue_mode='mixture'``). A sector's weights are a one-hot
row, so it keeps its own colour; the market path's weights are each
sector's **share of the basket's market capitalisation** that month
(reported share counts x price), so its colour shifts toward whichever
sectors dominate -- tech-blue-red as the 1990s bubble deflates, more
financial-gold before 2008, and back again. The title is the **current
date**, tinted by the basket's own trailing twelve-month return: red when
the market is below where it stood a year earlier, green when it is above.
The camera makes three turns over one minute, and nothing that has been
drawn fades, so the last frame is the whole quarter century.

**Zero padding, verified.** ``hyp.align`` zero-pads datasets with different
numbers of columns to a common width automatically (its ``trim_and_pad``
step), so hyperaligning sectors of unequal size is fine. ``hyp.reduce`` on a
list of unequal-width datasets deliberately refuses (it stacks them into one
shared fit), which is exactly why the reduction here is per sector -- the
sectors do not share columns, and should not share a projection.

**Data & graceful degradation.** Adjusted and unadjusted daily closes come
from Yahoo Finance's chart endpoint (full history, month-end decimated so no
future observation reaches back into a bar) and share counts from the SEC's
XBRL company-facts API (quarterly, from 2009; earlier months back-fill the
first reported capitalisation along the adjusted price). Everything is
cached on disk. If the network is unavailable the example falls back to a
seeded synthetic basket with the same sector structure and share counts, so
it always renders, and the technique is identical either way.
"""

# Code source: Contextual Dynamics Laboratory
# License: MIT

import json
import os
import tempfile
import urllib.request
from typing import NamedTuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.lines import Line2D

import hypertools as hyp

CACHE = os.path.join(tempfile.gettempdir(), 'hypertools_gallery_cache')
START = '2000-01-01'                    # first month drawn
PERIOD1, PERIOD2 = 915148800, 4102444800   # 1999-01-01 .. 2100-01-01, unix
WINDOW = 12                             # months in a trailing return
# six sectors, four or five tickers each -- unequal on purpose (see docstring)
SECTORS = {
    'Technology': ['AAPL', 'MSFT', 'ORCL', 'IBM', 'INTC'],
    'Financials': ['JPM', 'BAC', 'GS', 'AXP'],
    'Healthcare': ['JNJ', 'PFE', 'MRK', 'ABT', 'AMGN'],
    'Energy': ['XOM', 'CVX', 'COP', 'SLB'],
    'Consumer': ['KO', 'PG', 'WMT', 'MCD', 'PEP'],
    'Industrials': ['BA', 'CAT', 'GE', 'HON'],
}
SECTOR_COLORS = ['#c1272d', '#f2a900', '#1b7f4f', '#2d5fa8', '#7d3c98',
                 '#00808a']
DURATION, FPS, ROTATIONS = 60, 20, 3    # seconds, frames per second, turns
TAIL, TITLE_SIZE = 6, 17                # bright window (s), title points
# whitened PCA: without it the common growth of a sector's stocks is nearly
# all of the variance, and its 3-D path collapses onto one line
REDUCE = {'model': 'PCA', 'kwargs': {'whiten': True}}
# the SEC's fair-access policy wants a declared agent with a contact; this is
# the project's public one (pyproject.toml)
USER_AGENT = 'hypertools-gallery/1.1 (Contextual Dynamics Lab, contextualdynamics@gmail.com)'
SEC = 'https://data.sec.gov/api/xbrl/companyconcept/CIK{cik:010d}/dei/EntityCommonStockSharesOutstanding.json'
SEC_FACTS = 'https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json'
# the title's tint: the basket's trailing-12-month return, red below zero,
# green above, saturating at +/-30 % (a range the 2008 and 2020 drawdowns
# both leave)
TITLE_CMAP = LinearSegmentedColormap.from_list(
    'market', ['#b3261e', '#555555', '#1b7f4f'])
TITLE_NORM = Normalize(-0.3, 0.3, clip=True)


class Market(NamedTuple):
    sectors: dict               # sector -> DataFrame (months x its stocks): growth since START
    weights: pd.DataFrame       # months x sectors: share of basket market cap
    market: pd.Series           # months: cap-weighted trailing-12-month return
    source: str                 # which path produced it


def _cached_json(name, url):
    """Fetch `url` once into the cache (atomic write) and return the JSON."""
    dest = os.path.join(CACHE, name)
    if not os.path.exists(dest):
        req = urllib.request.Request(url, headers={'User-Agent': USER_AGENT})
        with urllib.request.urlopen(req, timeout=30) as resp:
            payload = resp.read()
        with open(dest + '.part', 'wb') as f:
            f.write(payload)
        # rename, so an interrupted download can never leave a truncated
        # cache file that every later run would trust
        os.replace(dest + '.part', dest)
    with open(dest) as f:
        return json.load(f)


# --- the data half: the ONLY code here that reaches the network -------------
def fetch_prices(sectors=SECTORS):
    """Daily ADJUSTED and unadjusted closes for every ticker, or ``None``
    if anything (network, parsing) goes wrong."""
    # outside the try, so it raises instead of being caught and quietly
    # downgraded: a test that sets HYPERTOOLS_OFFLINE is asserting that no
    # fetch happened, and a swallowed exception would hide one
    if os.environ.get('HYPERTOOLS_OFFLINE'):
        raise RuntimeError('HYPERTOOLS_OFFLINE is set: refusing to fetch')
    os.makedirs(CACHE, exist_ok=True)
    try:
        adjusted, raw = {}, {}
        for tickers in sectors.values():
            for ticker in tickers:
                # an explicit window: `range=max` silently degrades to
                # 3-month bars (measured 2026-09-03), a period does not
                result = _cached_json(
                    f'yahoo_daily_{ticker}.json',
                    'https://query1.finance.yahoo.com/v8/finance/chart/'
                    f'{ticker}?period1={PERIOD1}&period2={PERIOD2}&interval=1d'
                )['chart']['result'][0]
                stamps = pd.to_datetime(result['timestamp'], unit='s').normalize()
                quote = result['indicators']
                adjusted[ticker] = pd.Series(
                    quote['adjclose'][0]['adjclose'], index=stamps, dtype=float)
                raw[ticker] = pd.Series(
                    quote['quote'][0]['close'], index=stamps, dtype=float)
        return pd.DataFrame(adjusted).sort_index(), pd.DataFrame(raw).sort_index()
    except Exception as error:
        print(f'price history unavailable ({error!r})')
        return None


def fetch_shares(tickers):
    """Reported shares outstanding per ticker from the SEC's XBRL API, as a
    month-end series (forward-filled between filings, NaN before the first),
    or ``None``. Counts are as reported -- NOT split-adjusted -- which is why
    market cap below multiplies them by the UNADJUSTED close."""
    if os.environ.get('HYPERTOOLS_OFFLINE'):
        raise RuntimeError('HYPERTOOLS_OFFLINE is set: refusing to fetch')
    try:
        ciks = {row['ticker']: row['cik_str'] for row in _cached_json(
            'sec_company_tickers.json',
            'https://www.sec.gov/files/company_tickers.json').values()}
        shares = {}
        for ticker in tickers:
            facts = _cached_json(f'sec_shares_{ticker}.json',
                                 SEC.format(cik=ciks[ticker]))['units']['shares']
            if not facts:
                # the per-concept endpoint comes back EMPTY for a few filers
                # (ABT, KO -- measured 2026-09-03) whose complete
                # company-facts file carries the same concept; read it there
                facts = _cached_json(
                    f'sec_facts_{ticker}.json', SEC_FACTS.format(cik=ciks[ticker])
                )['facts']['dei']['EntityCommonStockSharesOutstanding']['units']['shares']
            # one value per period end: the LATEST filing wins over amendments
            frame = pd.DataFrame(facts).sort_values('filed')
            frame = frame.drop_duplicates('end', keep='last')
            series = pd.Series(frame['val'].to_numpy(float),
                               index=pd.to_datetime(frame['end']))
            shares[ticker] = series.resample('ME').last().ffill()
        return pd.DataFrame(shares)
    except Exception as error:
        print(f'share counts unavailable ({error!r})')
        return None


def synthetic_market(sectors=SECTORS, days=7000, seed=0):
    """The same sector structure, seeded, so the figure renders offline:
    daily closes (adjusted == unadjusted) and a constant share count."""
    rng = np.random.default_rng(seed)
    index = pd.date_range('1999-01-04', periods=days, freq='B')
    tickers = [t for ts in sectors.values() for t in ts]
    drift = rng.normal(0.0003, 0.0002, size=(1, len(tickers)))
    steps = rng.normal(0, 0.013, size=(days, len(tickers))) + drift
    closes = pd.DataFrame(100.0 * np.exp(steps.cumsum(axis=0)), index=index,
                          columns=tickers)
    shares = pd.DataFrame(
        np.tile(rng.uniform(1e9, 1.6e10, len(tickers)), (days, 1)),
        index=index, columns=tickers).resample('ME').last()
    return closes, closes, shares


def assemble(adjusted, raw, shares, sectors, source):
    """Month-end trailing returns per sector, market-cap weights per sector
    and the basket's own return, on one shared monthly index from START."""
    # month-end levels; a month still in progress is DROPPED (resample
    # stamps the last close so far at the month's END, which dated a clip
    # rendered on 4 September "September 30" -- measured 2026-09-04)
    levels = np.log(adjusted.resample('ME').last()).loc[:adjusted.index[-1]]
    returns = (levels - levels.shift(WINDOW)).loc[START:].dropna()
    months = returns.index
    # what is PLOTTED: each stock's cumulative log return since the first
    # month, so a sector's matrix is a set of growth curves and its 3-D path
    # is a journey rather than a tangle of month-to-month noise
    paths = levels.loc[months] - levels.loc[months[0]]
    # market cap = UNADJUSTED close x reported shares; before the first
    # filing, the first known cap is carried back along the ADJUSTED price
    cap = raw.resample('ME').last().reindex(months) * shares.reindex(months).ffill()
    first = cap.apply(lambda col: col.first_valid_index())
    for ticker in cap:
        known = cap.loc[first[ticker], ticker]
        adj = np.exp(levels.reindex(months)[ticker])
        cap.loc[:first[ticker], ticker] = known * adj / adj[first[ticker]]
    caps = pd.DataFrame({s: cap[t].sum(axis=1) for s, t in sectors.items()})
    weights = caps.div(caps.sum(axis=1), axis=0)
    market = (returns * cap.div(cap.sum(axis=1), axis=0)).sum(axis=1)
    return Market({s: paths[t] for s, t in sectors.items()}, weights,
                  market, source)


def load_market(sectors=SECTORS):
    """The ONLY function here that may touch the network."""
    prices = shares = None
    try:
        prices = fetch_prices(sectors)
        if prices is not None:
            shares = fetch_shares([t for ts in sectors.values() for t in ts])
    except RuntimeError:
        pass
    if prices is None or shares is None:
        return assemble(*synthetic_market(sectors), sectors,
                        'synthetic basket (offline)')
    return assemble(*prices, shares, sectors,
                    'Yahoo Finance closes, SEC share counts')


def fixture_data():
    """The same payload from the seeded synthetic basket. No network."""
    return assemble(*synthetic_market(), SECTORS, 'synthetic basket (fixture)')


# --- the figure half: no network, deterministic given its input -------------
def construct_artifact(data):
    """`data` in, the animation out. No network, no module globals.
    Returns the HyperAnimation wrapper, never the unpacked pair."""
    names = list(data.sectors)
    n_months = len(data.weights)
    # 1. each sector to 3-D in a space of ITS OWN stocks; 2. hyperalign the
    # six trajectories into one shared space; 3. the market is their mean
    reduced = [hyp.reduce(data.sectors[s], reduce=REDUCE, ndims=3) for s in names]
    aligned = hyp.align(reduced, model='HyperAlign')
    market = np.mean(aligned, axis=0)
    # mixture hue: one row of weights per month, blended through the six
    # sector colours. One-hot rows keep a sector its own colour; the market
    # path's rows are the sectors' shares of the basket's capitalisation.
    hue = [np.tile(np.eye(len(names))[i], (n_months, 1))
           for i in range(len(names))] + [data.weights.to_numpy()]
    # THE call: seven paths, one minute, three turns of the camera, and
    # a trail as long as the clip so nothing drawn ever fades.
    months, basket = data.weights.index, data.market.to_numpy()
    # the title is restyled per frame below; the library reserves its margin
    # at build time from rcParams, so the size is declared here as well
    with plt.rc_context({'axes.titlesize': TITLE_SIZE, 'axes.titleweight': 'bold'}):
        anim = hyp.plot(aligned + [market], '-', hue=hue, palette=SECTOR_COLORS,
                        hue_mode='mixture', linewidth=[1.1] * len(names) + [3.4],
                        manip='Smooth', animate=True, chemtrails=True,
                        tail_duration=TAIL, duration=DURATION, frame_rate=FPS,
                        rotations=ROTATIONS, colorbar=False,
                        title=f'{months[0]:%B} {months[0].day}, {months[0].year}',
                        size=(8, 8), show=False)
    ax = anim.figure.axes[0]
    ax.legend(handles=[Line2D([], [], color=c, lw=2, label=s)
                       for s, c in zip(names, SECTOR_COLORS)]
              + [Line2D([], [], color='#666666', lw=3.4,
                        label='Market (average across sectors)')],
              loc='upper left', fontsize=8, frameon=False)

    def date_title(ctx):
        """The title is the date under the head, tinted by the basket's own
        trailing return. Assigned on EVERY frame (the portable hook rule)."""
        frac = ctx.frame / max(1, ctx.n_frames - 1)
        when = months[0] + frac * (months[-1] - months[0])
        ret = np.interp(frac, np.linspace(0, 1, n_months), basket)
        ax.set_title(f'{when:%B} {when.day}, {when.year}', fontsize=TITLE_SIZE,
                     fontweight='bold', color=TITLE_CMAP(TITLE_NORM(ret)))

    anim.on_frame(date_title)
    return anim


if __name__ == '__main__':
    market = load_market()
    print(f'market data: {len(market.weights)} months x '
          f'{sum(f.shape[1] for f in market.sectors.values())} stocks in '
          f'{len(market.sectors)} sectors, {market.weights.index[0]:%b %Y} - '
          f'{market.weights.index[-1]:%b %Y} ({market.source})')
    anim = construct_artifact(market)
    fig = anim.figure
