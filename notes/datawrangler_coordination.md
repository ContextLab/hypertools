# datawrangler coordination log

The 2.0 refactor adopts datawrangler (`pydata-wrangler`) for the wrangling core. A parallel
Claude Code instance maintains `/Users/jmanning/data-wrangler` (repo `ContextLab/data-wrangler`,
default branch `main`). When we hit a dw bug or missing API, we file an issue there rather than
working around it.

## Environment
- Adopted `pydata-wrangler>=0.4.0` (PyPI latest 0.4.0, 2025-06-14).
- Refactor interpreter: `.venv` (Python 3.12.10). pandas pinned `>=2.2,<3` (see dw#30).

## Filed issues
| # | Title | Blocking? | Status |
|-|-|-|-|
| 30 | pandas 3.0 type detection (`type(x).__module__` strings) breaks is_dataframe / is_multiindex_dataframe → stack/unstack fail | Not blocking (pandas pinned <3) | open |

## Pending: lift the pandas<3 ceiling once dw#30 lands, then add pandas 3.0 to the CI matrix.

## API notes / deltas from the fork's dw usage
- Text embedding entry: `dw.wrangle(docs, model='CountVectorizer')` (also accepts
  `text_kwargs={'model': ...}`); fits on the bundled minipedia corpus by default.
