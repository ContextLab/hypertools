# datawrangler coordination log

The 2.0 refactor adopts datawrangler (`pydata-wrangler`) for the wrangling core. A parallel
Claude Code instance maintains `/Users/jmanning/data-wrangler` (repo `ContextLab/data-wrangler`,
default branch `main`). When we hit a dw bug or missing API, we file an issue there rather than
working around it.

## Environment
- Adopted `pydata-wrangler>=0.5.0` (was 0.4.0). **Upgraded 2026-07-03** — Jeremy released dw 0.5.0
  (PyPI + local `/Users/jmanning/data-wrangler` @ v0.5.0) which fixes dw#30.
- Refactor interpreter: `.venv` (Python 3.12.10). **pandas `<3` ceiling LIFTED** → now `pandas>=2.2.0`.
  Validated on **pandas 3.0.3 / dw 0.5.0 / numpy 2.3.5** (core+manip+external = 51 passed; full suite
  re-run at upgrade time). CI adds a pinned-pandas-3 acceptance gate (ubuntu/py3.12) in `test.yml`.

## Filed issues
| # | Title | Blocking? | Status |
|-|-|-|-|
| 30 | pandas 3.0 type detection (`type(x).__module__` strings) breaks is_dataframe / is_multiindex_dataframe → stack/unstack fail | Was gating the <3 pin | **RESOLVED in dw 0.5.0** |

## New observation (not yet filed): dw 0.5.0's `decorate.py:421` emits a `Pandas4Warning`
(`pd.concat(..., copy=...)` keyword deprecated under pandas 3.0 Copy-on-Write). Harmless now
(forward-looking pandas-4 warning), surfaces via `dw.stack`. File on data-wrangler if it becomes
an error under a future pandas. Our tests still pass; not blocking.

## API notes / deltas from the fork's dw usage
- Text embedding entry: `dw.wrangle(docs, model='CountVectorizer')` (also accepts
  `text_kwargs={'model': ...}`); fits on the bundled minipedia corpus by default.
