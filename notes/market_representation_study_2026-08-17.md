# Market example: representation + forecast study (2026-08-17)

Evidence behind the decision on what the Market gallery example should plot,
and whether it may claim a forecast result. Run with
`.venv/bin/python scripts/market_representation_study.py` on live/cached Yahoo
data (24 tickers, 2513 daily closes). The maintainer's Option D asked for a
preregistered comparison; this is it, plus what it found.

## 1. The visual candidates do NOT solve legibility

Both were rendered, at 109 monthly points, and inspected:

| candidate | leaves | roughness (min/med/max) | occupancy | build |
|-|-|-|-|-|
| D1 six sector level paths | 6 | 8.2 / 11.2 / 15.1 | 1.11/1.10/1.02 | **18.0 s** |
| D2 six stocks in three sectors | 6 (+3 sector means +1 market) | 8.5 / 11.0 / 11.6 | 1.22/1.14/1.00 | **24.9 s** |
| current shipped rewrite (weekly return/vol/momentum) | 6 | -- | -- | 219.8 s |

Roughness lands where the maintainer measured it (6-11 for monthly level-like
paths, vs 26-36 for the weekly oscillatory ones), and the **cost problem is
solved** -- 18-25 s against a 45-60 s target. But **the picture is still a
knot**: both renders show a dense tangle occupying a fraction of the cube,
with sector identity and temporal direction unreadable. Level-like features
and monthly sampling made the knot smaller and cheaper, not into a path.

Measured, so this is not an impression: the observed data spans 53-64% of the
axis range while the *bulk* of it sits in a small sub-volume, i.e. the extent
is set by one or two early excursions. `predict=` is NOT to blame (0.53 with
forecasts vs 0.54 without).

**Two measurement traps hit on the way**, both worth remembering: the black
bounding cube is SIX `Line3DCollection` wireframe faces that carry
`_segments3d` exactly like the data, so an unfiltered extent measurement
reports the CUBE (fill = 1.00 always); and `get_segments()` returns the 2-D
PROJECTION once the canvas has been drawn, so it cannot be used for a 3-D
extent at all. Filtering by `_hyp_trace_index` would be the clean way -- but
the ANIMATED continuous-hue path never sets that tag (only the static path
does). That library gap is now a real obstacle to introspection, not just an
inconsistency.

## 2. The forecast claim does NOT survive a proper baseline

The first run's baselines (zero / persistence / mean change / EW continuation)
made the models look good: Laplace on `cum_return` scored 0.22-0.40 against
baselines that were all NEGATIVE, in both time blocks, at both horizons --
i.e. it passed the preregistered acceptance rule.

**That was the baselines' fault, not a result.** `cum_return` is a trailing
12-month sum, so its change is `r[t] - r[t-12]`: the second term drops out of
the window and is **known exactly** one step ahead. Measured across the six
D2 leaves, that known term carries **37-52% of the change's variance** and
correlates **0.57-0.74** with it. None of the four trivial baselines can see
it; any model that has seen the history can.

Adding the window-aware baseline (`window_dropout`, now permanent in the
script) reverses every verdict:

| spec | model | best baseline | verdict |
|-|-|-|-|
| D2 Laplace h=1 block1 | 0.266 | **0.682** | fails |
| D2 Laplace h=1 block2 | 0.219 | **0.753** | fails |
| D2 Laplace h=3 block1 | 0.369 | **0.532** | fails |
| D2 Laplace h=3 block2 | 0.364 | **0.369** | fails |

And controlling for the known term directly, Laplace's partial correlation
with the outcome collapses from 0.266 to **+0.133** (block1) and from 0.219 to
**+0.070** (block2).

**Under the preregistered rule, no specification earns a forecast claim on
`cum_return`.** Kalman on `drawdown` still beats its baselines in both blocks
(0.227 vs -0.442; 0.206 vs -0.159), but drawdown has the SAME class of
artifact by construction -- at a new high the change is `-drawdown[t-1]`,
which is known -- and it has not been audited for it. Do not claim it either
without that audit.

### 2a. Corrections, 2026-08-18 (maintainer review)

Three defects in the run above, all found by review rather than by me, and
all now fixed in the script with tests (`tests/test_market_representation_study.py`):

1. **The window baseline was not horizon-aware.** It returned the one-step
   dropout `-rets.shift(12)` at every horizon. At `h=3` three known returns
   leave the window, not one, so the baseline was scored on a third of the
   information available to it -- understated in exactly the comparison where
   a model has most room to look good. The identity is
   `-(L - L.shift(h)).shift(CUM_WINDOW + 1 - h)`, which collapses to the old
   expression at `h=1`. Correcting it RAISED the h=3 baselines (0.532 ->
   0.765, 0.369 -> 0.717): the conclusion held and got stronger.
2. **It was applied to D2 only.** A D1 sector's `cum_return` is a trailing
   sum of the sector level and carries the same dropout, so D1 was being
   scored against an incomplete baseline set. Now both.
3. **The rule was being applied by eye.** It is now applied *in code*
   (`verdict()`), including the clause I had been skipping -- "at a horizon
   the example actually draws", which is `t=1`. Reading a rule off a table
   is how a rule gets applied leniently; five of the eight apparent passes
   were at `h=3`, a horizon nothing draws.

**Retraction:** the sentence "no specification earns a forecast claim" was
too broad as first written. Applied mechanically, the preregistered rule
passes **three** specifications, all on `drawdown` at `h=1`:

| spec | block1 | block2 | best trivial baseline |
|-|-|-|-|
| D1 Kalman | +0.241 | +0.151 | -0.504 / -0.040 |
| D1 Laplace | +0.234 | +0.070 | -0.504 / -0.040 |
| D2 Kalman | +0.227 | +0.206 | -0.442 / -0.159 |

### 2b. The drawdown audit this file demanded — and its result

Section 2 said "do not claim it either without that audit". Run:
`drawdown` is bounded above at zero and recovers, so **"predict full
recovery"** (change = `-drawdown[t]`) is a parameter-free rule needing no
model at all. Deliberately kept OUT of the acceptance rule -- adding a
competitor after seeing whom it would beat is precisely what preregistration
prevents -- and reported beside it.

It beats the models in **10 of 12** cells, including **all three survivors**:

| spec | model | full-recovery rule | verdict |
|-|-|-|-|
| D1 Kalman block1 | +0.241 | **+0.324** | loses |
| D1 Laplace block1 | +0.234 | **+0.324** | loses |
| D2 Kalman block1 | +0.227 | **+0.399** | loses |
| D2 Kalman block2 | +0.206 | **+0.247** | loses |
| D1 Laplace block2 | +0.070 | **+0.072** | loses |
| D1 Kalman block2 | +0.151 | +0.072 | beats — one block only |

Every trivial baseline on `drawdown` is anti-correlated with the realised
change (down to -0.504), because persistence predicts "keep falling" exactly
when a recovery is likeliest. Clearing that bar is nearly free. The one
specification that beats the recovery rule does so in a single block, which
fails the same both-blocks clause the rule already applies.

**So the conclusion of section 2 stands, now for the right reason and with
the loophole closed: no specification earns a forecast claim on any measure
at the horizon the example draws.**

## 3. What this implies

The maintainer's editorial option is the one the evidence supports: **Market
becomes a hierarchy VISUALIZATION with no forecast-skill claim**, and the
"prediction that works" story moves to data with genuine temporal structure.
Neither D1 nor D2 is yet good enough on legibility to be the showcase, so the
representation question is still open -- what is settled is that (a) monthly
level-like features fix the COST, (b) they do not fix the KNOT, and (c) the
forecast numbers do not survive an honest baseline.
