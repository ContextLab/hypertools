## Gallery-examples + tutorials audit — every example directly executed

Toward the "every example directly checked" goal, I executed **all 53 gallery example
scripts** end-to-end (headless, `MPLBACKEND=Agg`, `show()` patched so plotly can't block
on a browser, 420 s timeout each).

### Result: 51 / 53 pass; 1 real bug fixed; 1 known headless limitation

**Fixed — `plot_hue.py` raised `ValueError` (`7507a23e`).** The original author's example
passes `hue` as one sub-list per dataset (the classic list-of-lists matching a list-of-
datasets input). The 1.0 hue validation flattened the *data* to `n_obs` but read
`np.asarray(hue)` on the `(n_datasets, len)` sub-lists as a **2-D matrix hue**, so it raised
*"hue has 3 entries but the data has 900 observations."* Now a nested hue whose top level
matches the number of datasets and whose sub-sequences match each dataset's length is
flattened to one value (or matrix row) per observation before classification; genuinely flat
/ `(n_obs, k)` matrix hues are untouched, and a nested hue with mismatched sub-lengths still
raises (no silent truncation). +4 regression tests + docstring note.

**Not a bug — `animate_plotly.py` timed out.** Its interactive `animate=True` call (line 24)
runs fine; the hang is the `save_path='spin.gif'` **plotly→kaleido export**, the same
pre-existing headless-environment deadlock already documented for the 6 deselected
`test_plotly_*_export` tests. It works with a display; no code defect.

Full suite after the hue fix: **1489 passed, 0 failed, 4 skipped, 7 deselected.**

### Tutorials

Executed 10 tutorial notebooks end-to-end via `nbconvert --execute` (headless, real data
+ real models) — **all 10 pass** against the current 1.0 API:

| tutorial | | tutorial | |
|-|-|-|-|
| `plot` | ✅ | `cluster` | ✅ |
| `analyze` | ✅ | `modern_sklearn_dynamics` | ✅ |
| `normalize` | ✅ | `projectile_kalman` | ✅ |
| `reduce` | ✅ | `stock_forecasting` | ✅ |
| `align` | ✅ | `text` (HF embeddings) | ✅ |

Not executed here (environment-gated, not API checks): `lsl_streaming` and `streaming_data`
need a live Lab-Streaming-Layer source, and `hugging_face_embeddings` / `wikipedia_embeddings`
/ `conversation_trajectories` pull large models/corpora over the network — the representative
HF path is already covered by `text` passing above.

**Branch is for review only — do not merge; base `dev-1.0-refactor`, `master` untouched.**
