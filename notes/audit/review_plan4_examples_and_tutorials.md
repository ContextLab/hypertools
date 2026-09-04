# Plan 4 (examples and tutorials) — first adversarial review

Dispatched 2026-07-31. **Plan 4's first review of any kind.** Its three sibling plans each needed
1–4 rounds, and every round found defects that would have failed on first execution; this one is no
exception — two **Fatal**.

Method: every cited line number, symbol, notebook path and API opened in the real file; the plan's
own measurement script run against the plan's own proposed rewrites rather than trusting its table.

## Findings

| Sev | Location | What is wrong | Evidence (verified) |
|-|-|-|-|
| **Fatal** | Task 8 Step 2 `BUDGETS` | 4 of 5 rewrites miss their own contracted floors, so `test_file_meets_its_native_ratio_floor` + "109 passed" cannot pass | Ran the plan's own `measure_native_ratio.py` on its own Task 2/4/5/6 code: market 109/16 = **14.7%** (needs 26), paintings 112/14 = **12.5%** (needs 20), conversation **74** lines / **18.9%** (needs ≤72 / 25), morph 27/6 = **22.2%** (needs 26). Only weather (56 / 25.0%) passes |
| **Fatal** | Task 5 `recency_fade` | Iterates `ctx.artists` but indexes `ctx.revealed_counts[i]`; with `chemtrails=True` artists is ~2N while revealed_counts is N → `IndexError` | animation-core plan:2665 — `artists=list(lines) + [t for t in trail_lines if t is not None]` |
| **High** | Prereqs Task 2 row; Self-Review; Risk #1 | Cites "MultiIndex T4" for continuous hue **4×**; T4 is the flat return-bundle, hue is **T6**. Self-Review also says T1/T2/T3/T4/T6 vs the table's T1/T2/T5/T6/T8 | multiindex.md:1307 (T4), :1469 (T5), :1759 (T6), :2408 (T8) |
| **High** | Verification note; Task 8 docstring; Decisions | "all five launch notebooks ship ZERO executed outputs" is **false** | Measured 2/6, 4/7, 1/6, 2/6, 2/7; `git log 9b94d86f` 2026-07-30 "execute the five new tutorials" |
| **High** | Baseline table (plan:47-51) | All five **notebook** baselines wrong (the five `.py` rows are exact) | conversation 191/12/6.3% (claimed 186/11/5.9), market 193/12/6.2, morph 46/9/19.6, paintings 121/11/9.1, weather 207/11/5.3 |
| Med | `docs/conf.py:115` (cited 5×) | `nbsphinx_execute='never'` is at **conf.py:131**; `:115` is blank | grep |
| Med | Task 1 Step 5 / Self-Review | "17 passed" / "17 tests"; the block defines **16** → suite delta is +125, not +126 | `grep -c '^def test_'` = 16 |
| Med | Tasks 3–6 "Execute and measure" | Expected counts impossible against the plan's own cell tables: weather 4/4 (5 code cells), paintings 5/5 (6), conversation 5/5 (6), morph 4/4 (5); only market 7/8 is right |
| Med | Task 7 Step 2 grep gate | `grep -l SentenceTransformer docs/tutorials/*.ipynb examples/*.py` "no output" also needs Tasks 4+5, yet Task 7 is declared runnable early / in parallel | 7 files match today |
| Med | "Decisions still needed" | Header says "deliberately UNNUMBERED"; items 4/5/6 are numbered. README lists only 3 of the plan's 6 | plan:2198 vs :2218-2232 |
| Low | Task 7 baseline | "The five clean ones" then names **seven** notebooks |
| Low | Task 5 cites | `fig.legend` is `:171-174` (not 168-175); title `fig.text` `:175-176` (not 176-177); speaker artist `:178-179` |
| Low | Task 1 Step 4 | Fixed `IMAGE_PALETTE_N=6` makes a >6-category hue raise "supplies 6 color(s) but N are required" | colors.py:332 |

**The `recency_fade` Fatal was independently re-verified** (not taken on the reviewer's word):
plan4 `:1501-1502` iterates `for i, artist in enumerate(ctx.artists)` and then indexes
`ctx.revealed_counts[i]`, while animation-core builds
`artists=list(lines) + [t for t in trail_lines if t is not None]` alongside
`revealed_counts=_counts` (one per dataset) — so with `chemtrails=True` the artists sequence is
roughly twice the counts sequence and the index runs off the end.

The ratio-floor Fatal rests on the reviewer running the plan's own measurement script against the
plan's own proposed code. That is the right method and the numbers are specific, but it is the one
finding I have **not** re-derived myself; re-measure before acting on the exact percentages.

## Confirmed correct

colors.py 24/105/227/250/269/287/305-306/323-331; plot.py 2750-2751, 807, 895, 1013;
text2mat / smooth / normalize / predict citations; `fig, ani = hyp.plot(...)` unpacking; the weather
one-call claim (2 axes, 0 warnings, 686 colours); `hyp.predict(..., t=1)` → `(1,1)`;
`describe(show=False)` → dict; the `ax=` / 2-column-DataFrame calls; Plan 1 T5/T7/T8/T9-Step5 and
Plan 3 T3/T4/T5 citations.

**Tutorial notebooks are TRACKED, not gitignored** (only `docs/auto_examples/` is), so the plan's
execute-and-commit method is sound.

## Open decisions (README "Decisions still open / Plan 4")

| decision | status | recommendation |
|-|-|-|
| Where `image_palette` is exported | still open | **keep module-level + `palette='image:'`**. Plan 1 already grows `__all__` for `FrameContext`, so the stated objection is weak — but the curated surface is still better kept small |
| The paintings outlier trim | still open | **drop it** — confirmed `hyp.reduce` has no `vectorizer=`, and `manip` runs pre-reduce |
| The morph example's 5-line `normalize()` helper | still open | **keep** — verified 5 lines (`examples/animate_morph_zoo.py:38-42`) and genuinely aspect-preserving vs `normalize='within'` |
