# HyperTools 1.1 — Examples and Tutorials Implementation Plan (v3)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the five launch examples and the fifteen older tutorials *showcase* hypertools instead of working around it. Measured 2026-08-02 with the docstring-aware metric, **6.6% of the code in the five launch examples belongs to a hypertools call** (48 of 723 code lines; v2 said 48 of 739, which predates `d730a085` shrinking morph from 40 code lines to 26) and **37.9% is defect** — it either re-implements something native or fills a gap Plans 1–3 close. This plan rewrites every one of them against the 1.1 API, in lockstep with its notebook, and adds the one library feature the examples still need that no other 1.1 plan owns.

**Architecture:** One library task first (Task 1: palette-from-image), because two example rewrites consume it and it is the last orphaned feature from the audit. Then five example rewrites, each *paired with its notebook in the same task and the same commit* — a script fixed without its notebook leaves the defect published, because `nbsphinx_execute = 'never'` (`docs/conf.py:131`) ships the committed notebook verbatim. Then the fifteen older tutorials, grouped by the recurring fix so each step is one reviewable diff. Finally a verification task that makes the improvement **permanent**: a committed measurement script, a real pytest module that fails if any defect marker reappears, the full suite, every example executed, every notebook re-executed, and a zero-warning docs build.

**Tech Stack:** Python 3.12.10, numpy 2.3.5, pandas 3.0.3, matplotlib 3.10.8, plotly 6.8.0, scikit-learn, Pillow 12.1.0, nbconvert 7.17.1, ipykernel 7.3.0, pytest 9.0.2.

---

## Revision note (v3)

v2 was adversarially re-reviewed (`notes/audit/review_plan4_v2.md`: 4 Fatal, 4 High, 5 Med, 4 Low) and seven measurement audits were then run against the real repo (`notes/audit/plan4_metric_remeasure.md`, `plan4_landed_state.md`, `plan4_image_palette.md`, `plan4_notebook_gate.md`, `plan4_network_decoupling.md`, `plan4_citations_and_ci.md`, and `plan3_closure_audit.md`). Everything below is measured, not estimated.

**The single biggest change: four of the five example scripts were rewritten out from under this plan.** Commit `d730a085` (2026-08-01 09:46) migrated market, weather, conversation and morph off the private `ani._func`/`ani._args` monkeypatch and onto the public `anim.on_frame(...)`. Only paintings is untouched. Tasks 2, 3, 5 and 6 are therefore **rebases**, not rewrites, and Task 6's Step 1 is already done verbatim. Applying v2's prescribed text as written would overwrite newer code with older text — in Task 5's case, replacing working code with a crash.

| v2 | v3 |
|-|-|
| **Fatal.** Notebook budgets were derived with a metric that stripped docstrings for `.py` but not `.ipynb`, so identical source measured `(3,2)` vs `(11,2)`. | `_code_lines_nb` and `_code_lines_py` now share ONE docstring-stripping callee (a shared callee cannot drift from itself). Re-measured: market 193→187, weather 207→194, conversation 191→176. The refactor is proven safe — `_code_lines_py` is identical line-for-line on all five scripts. |
| **Fatal.** Two notebook budgets were unsatisfiable. | The real cause was simpler than the metric: **paintings (110) and conversation (76) were set BELOW their own script budgets (118, 90)**, and a notebook holds its script's code plus an install cell plus a display cell. Budgets are no longer written down — `notebook_budget = script_budget + NOTEBOOK_OVERHEAD` with `NOTEBOOK_OVERHEAD = 5` measured. All ten now pass, tightly (headroom 7/7/7/2/5), and the error class is structurally impossible. |
| **Fatal.** `EXPECTED_OUTPUT_CELLS` guessed five counts; 4 of 5 looked unattainable. | **All five were wrong, and so is every per-task "Execute and measure" claim in Tasks 2–6** — each assumed every non-install cell emits, when several are bare imports, bare assignments, or `fig, ani = hyp.plot(..., show=False)`. Weather's total matched *for the wrong reason*: the real emitting cells are the exact complement of the assumed ones. Replaced by an index-set gate that names the offending cell. |
| **Fatal.** The image palette patched one resolver; categorical hue reached seaborn without it. | **"ONE interception point" is false**: `_seaborn_palette_arg` plus five raw seaborn sites, one of which (`sns.set_palette`) runs on *every* matplotlib call. Measured as real `hyp.plot` calls the maintainer's six scenarios scored **0/6**, not the 2 failures v2's own tests reported. Fixed at both interception points with a dynamic colour count: **6/6**. |
| Task 6 described as unimplemented. | Task 6 Step 1 **already landed** in `d730a085`, verbatim, including the prescribed docstring. Only the notebook remains. |
| `test_examples_produce_their_stated_artifact` ran examples with `runpy`. | Network/model-download work is split out; the suite drives a fixture-fed construction boundary, and the whole-example run becomes an opt-in smoke test. |
| The morph assertion `assert 'morph' in str(ns.get('ANIMATE', 'morph'))`. | `ANIMATE` does not exist in the example; the expression reduces to `'morph' in 'morph'` and **cannot fail** (proven by execution). Replaced with driven-frame assertions. |
| `git stash && measure && git stash pop` to read the BEFORE state. | **Data-loss hazard**, demonstrated: with a clean tree `stash` saves nothing and `pop` then restores *and drops* an unrelated pre-existing stash. Replaced with `git show <base>:<path>`, which is read-only. |
| Task 5 "13 tests"; suite delta "+135". | **12** and **+178** (Task 1 **19** + Task 5 **12** + Task 8 **147**). The plan already said 12 at the step level and 13 in the revision note — it disagreed with itself. Task 8 (106) was correct; Task 1 rises 16 → **19** with the three new colour-count tests this revision adds. All three derived by AST parametrize-expansion and cross-checked against a real collection. |
| Ratio floor "removed as a v1 Fatal". | It was removed from the *gate* but **ten lines still promise it** as a budget (634, 990, 1006, 1186, 1201, 1420, 1435, 1792, 1807, 1904). All corrected. L1435 was doubly stale (`≤ 72` where the enforced dict said 90). |
| "'0 executed outputs' corrected wherever it appears." | It was not — all five BEFORE headers still said it. Real values: 4/7, 2/7, 2/6, 2/6, 1/6. |
| Baseline `2564 collected`; "the working tree is not clean". | **2782/2784 collected (2 deselected)**; the tree is clean at `065c841e`; the note claiming the five examples are untouched is false (see above). |

**One review finding REJECTED, with evidence.** The v3 adversarial review reported (M2) that the equal-feature-width citation `plot.py:3152-3153` is stale and the real location is `3164-3165`. Checked against the repo: `sed -n '3152,3153p'` gives exactly `_widths = [ri.shape[1] for ri in raw]` / `if len(set(_widths)) > 1:`, and `3164-3165` is unrelated (`return False` / `_text_hint = (`). The citation is correct as written. The likely cause is instructive and worth guarding against: that review measured inside a worktree with **Task 1's patches already applied**, and those insert lines into `plot.py` above this point, shifting everything below. **A line number verified in a patched tree is not verified.** Re-check citations against a clean checkout.

**New Fatal found during this revision, not present in any prior review:** `hyp.plot(..., animate=...)` returns a `HyperAnimation`, a `(figure, animation)` **tuple subclass**. `fig, ani = hyp.plot(...)` binds `ani` to element `[1]` — the raw `FuncAnimation` — **discarding the wrapper that carries `.on_frame()`**. v2's Task 5 notebook does exactly this and dies with `AttributeError` at the cell that calls `ani.on_frame(recency_fade)`, so `nbclient` halts and the notebook never finishes. The already-landed script avoids it by binding `anim = hyp.plot(...)` without unpacking. Contract 8 below now states the rule, and a Task 8 test enforces it. Blast radius was measured across `docs/`, `examples/`, `hypertools/`, README and CHANGELOG: **the trap existed only in plan documents** — the shipped library and `docs/animation.rst` are correct.

---

## Revision note (v2)

v1 was adversarially reviewed for the first time (`notes/audit/review_plan4_examples_and_tutorials.md`: 2 Fatal, 3 High, 5 Med/Low) and the maintainer ruled on 2026-08-01. Both Fatals are fixed here.

| v1 | v2 |
|-|-|
| **Fatal.** `recency_fade` iterated `ctx.artists` while indexing `ctx.revealed_counts[i]`. `ctx.artists` is heads-then-trails, so with `chemtrails=True` it holds 2N entries against N counts — `IndexError` on the N+1th artist, on the first frame. | Splits by role first (`heads = ctx.artists[:n]`, `trails = ctx.artists[n:]`), asserts one trail per dataset rather than assuming it, and drives head and trail together. `n` comes from `len(ctx.revealed_counts)`, which is authoritative after `hue=` reshaping. Task 0 of Plan 3 documents the same class as **Contract 9** (forecast artists deliberately stay OUT of `ctx.artists` for exactly this reason). |
| **Fatal.** Task 8's `BUDGETS` gated a per-file native-code ratio, and 4 of 5 rewrites missed their own floors when the reviewer ran the plan's own script against the plan's own proposed code (market 14.7 vs 26, paintings 12.5 vs 20, conversation 18.9 vs 25, morph 22.2 vs 26). The gate could not have gone green. | The ratio is **reported, not gated** (maintainer's call): it is easy to game by reformatting and says little about quality. The gates that remain cannot be met by formatting — defect markers, a maximum code-line budget, executable semantic checks that each example still produces the artifact it advertises, and exact notebook execution success. |
| The callback returned early when `ctx.current_index is None`, and skipped assignment on some artists. | Every head and every trail is assigned on **every** frame, including unspoken turns; the condition moved into the VALUE (`turn_alpha`). A parallel animation now raises instead of silently doing nothing. |
| No tests for the callback at all. | `tests/plot/test_recency_fade.py` — 13 tests: first/middle/last turn, repeated and out-of-order frames, the trail pairing, the cardinality guard, and the single-point case. |
| Notebook-execution gate allowed `len(code) - 2` unexecuted cells. | Exact: every code cell must carry outputs, and none may carry a committed traceback. **(v3: this was itself unattainable — an imports-only cell and a `fig, ani = …` cell emit nothing however well they run, and the install cell is exempt besides. Superseded by the three-part gate in v3's note above.)** |
| "all five launch notebooks ship ZERO executed outputs" | False when written — measured 2/6, 4/7, 1/6, 2/6, 2/7 (`git log 9b94d86f`, 2026-07-30). Corrected wherever it appears. |
| Task 1 "17 passed"; Task 8 "109 passed"; suite delta +126. | **16**, **106**, and **+135** (Task 5 now contributes 13). Each derived in a table at its step. |
| `docs/conf.py:131` for `nbsphinx_execute='never'` (cited 5×) | `docs/conf.py:131`; `:115` is blank. |

---

## Verification note (v1)

This is the first revision of this plan, so there is no prior version to correct. What follows is the equivalent discipline: **every fact this plan leans on was re-measured in this repo before the plan was written**, and the table records where the handoff brief, the two audits, or my own first guess turned out to be wrong or incomplete.

| claim as received | what I measured (`/Users/jmanning/hypertools/.venv/bin/python`, 2026-07-28) |
|-|-|
| "`nbsphinx_execute = 'never'` means committed outputs ship verbatim" | True (`docs/conf.py:131`). The five launch notebooks are **partially** executed — re-measured 2026-08-01: `conversation_shape` **2**/6 code cells carry outputs, `market_forecast` **4**/7, `morph_shapes_zoo` **1**/6, `painting_embeddings` **2**/6, `weather_decades` **2**/7. (v1 said 0 for all five; `git log 9b94d86f`, 2026-07-30, "execute the five new tutorials" had already landed.) The 15 older tutorials carry 3–15 executed cells each. So the five launch tutorial pages render **most of their code with no figure**, not none of it — the fix is the same. **(v3: "Task 8's gate is now exact (every code cell)" was wrong; see v3's note. The gate now checks execution, a measured output INDEX SET, and the committed GIF artifact separately.)** There is also **no gallery thumbnail** for any of the five (`docs/_static/thumbnails/` holds 12 files; `scripts/generate_gallery_thumbs.py:26` hard-codes `MPL_ANIMS = ['animate', 'animate_MDS', 'animate_spin', 'chemtrails', 'precog', 'save_movie']`). Task 8 fixes both. |
| Equal per-dataset feature widths required by `plot.py:2748-2756` | The comment block starts at `plot.py:2748-2756`; the **check** is `plot.py:3152-3153` (`_widths = [ri.shape[1] for ri in raw]` / `if len(set(_widths)) > 1:`). Cite 2750-2751. |
| Market panel: 24/24 tickers, 2513 trading days, 2016-07-28 → 2026-07-28 | **Confirmed exactly.** All 24 tickers fetched from `https://query1.finance.yahoo.com/v8/finance/chart/<T>?range=10y&interval=1d` with a `User-Agent` header; every one returned `len(timestamp) == 2513`; AAPL first 2016-07-28, last 2026-07-28. Six sectors × 4 tickers = equal widths, satisfying `plot.py:3152-3153`. |
| Weather: `temperatures.csv` is (1645 months, 20 cities) | The raw CSV is **(1965, 43)**: `Unnamed: 0`, `Year`, `Month`, then **both** `<City>_anomaly` and `<City>` for 20 cities. `dropna()` → **1645 complete rows, 1875–2013**. The 20 absolute-temperature columns are selected by `raw[list(locs['City'])]` → `(1645, 20)`. `temperature_locs.csv` is (20, 4): `Unnamed: 0`, `City`, `Lat`, `Long`. |
| The weather paper call is "essentially ONE native call", 516 distinct colours | Confirmed, and stronger. `hyp.plot(temps, fmt='-', hue=avg_temp, palette='RdBu_r', normalize='across', manip='Smooth', animate=True, chemtrails=True, colorbar=True, duration=8, frame_rate=20, show=False)` runs in **0.3 s**, emits **no warnings**, and produces **2 axes** (`Axes3D` + the colorbar `Axes`). After driving frame 150 the head+trail collections carry **879 distinct RGBA values**. (516 was presumably a different frame/duration; the qualitative claim holds and is now pinned to exact parameters.) |
| `hyp.reduce(list_of_strings, ndims=3)` → (8, 3) | Confirmed. Also confirmed: `hyp.reduce([[s,s,s],[s,s,s],[s,s]], ndims=3)` → `[(3,3), (3,3), (2,3)]`, so grouped text needs no manual re-split; and `hyp.plot(texts, '.', ndims=3, vectorizer='TfidfVectorizer', semantic=None, corpus=None)` → `Figure`. |
| `labels=` is per-OBSERVATION, flat or nested | Confirmed on real artists. Flat `[None]*15` with 2 non-None entries → 2 annotations; nested `[[...5], [...5], [...5]]` with 2 non-None entries → 2 annotations. |
| GIF saving is native | Confirmed end to end: `hyp.plot(..., animate=True, save_path='x.gif')` wrote a **24 832-byte** real GIF with no ffmpeg (`plot.py:1513-1520`, dispatch at `animate.py:84`). |
| Palette-from-image is ABSENT | Confirmed. `hyp.plot(..., palette='image:/tmp/nope.png')` → `ValueError: 'image:/tmp/nope.png' is not a valid palette name` (raised by seaborn through `colors.py:306`). No `PIL`/image handling anywhere in `hypertools/`. |
| The existing `image_palette()` orders k-means clusters BY SIZE and so returns the background tone | Confirmed and reproduced. On a synthetic 90%-beige / 10%-vivid-red image, `km.cluster_centers_[np.argmax(counts)]` (`examples/animate_painting_embeddings.py:138-140`) → `[0.784, 0.769, 0.737]` (the beige). Ordering by `population × chroma` → `[0.863, 0.078, 0.078]` (the red) first. Task 1 encodes this as a test. |
| Verified baseline `2564 collected`, `2551 passed, 13 skipped` | `pytest --collect-only -q` → **`2564/2566 tests collected (2 deselected)`**. Consistent. |
| *(my own first guess)* the notebooks can be re-executed with the `python3` kernel | **False.** `~/Library/Jupyter/kernels/python3/kernel.json` points at an unrelated project's `.venv/bin/python`, not this repo's. Notebook execution needs a kernel registered from **this** repo's venv (`ipykernel` 7.3.0 is installed there). Global Constraints carry the exact recipe. |
| *(my own first guess)* a new `image_palette` needs a Pillow extra | **Unnecessary.** `matplotlib>=3.9.0` is a core dependency and itself requires `pillow>=8` (`importlib.metadata.requires('matplotlib')` → `['pillow>=8...']`), so Pillow is already guaranteed in every hypertools install. Task 1 declares it explicitly anyway (one line, zero new install weight). |
| *(my own first guess)* a categorical `hue=` would collapse the conversation's 28 turns and break per-segment `title=` | **False, verified.** With 6 line datasets and a nested categorical `hue`, `hyp.plot(...)` draws **6 lines** and a **3-entry legend** (`['Alice', 'Hatter', 'March Hare']`); `animate='serial'` still passes **6 datasets** to the backend (`len(ani._args[0]) == 6`). So per-turn `title=` and per-speaker `hue=`/`legend=True` compose. (``_regroup_categorical_lines`, plot.py:219` regroups *contiguous runs*, not whole categories.) |
| *(unstated)* the market accuracy readout is cheap | **It is not, and the budget is now measured.** `hyp.predict(x, model='Kalman', t=1)` costs 274 ms at 60 rows, 217–472 ms at 250, 445 ms at 500, 873 ms at 1000, **2178 ms at 2500**. The full walk-forward loop was timed: 7 series × 30 anchors on a **60-row rolling window = 210 fits in 7.3 s**; the same loop at a 250-row window costs 30.7 s. The current example's whole run is **6.2 s**. Task 2 therefore fixes `WINDOW = 60`, `N_SCORED = 30` and states the measured cost. |
| *(unstated)* `manip={'model':'Smooth','kwargs':{'kernel_width':10}}` is a clean drop-in for the pandas rolling mean | It works, but emits `UserWarning: Increasing smoothing kernel width by 1 (must be odd)` (`hypertools/manip/smooth.py:232`). Task 7 uses **11**, not 10, so the tutorial produces no warning. |
| *(unstated)* the measurement metric | The audit's "% hypertools" counts *lines matching* `\bhyp\.|\bhypertools\b`, which scores a 10-line `hyp.plot(...)` call as **1** native line. This plan uses a **logical-statement** metric (a continuation line belongs to the statement it continues). Measured on the same five scripts it gives 48 native of 739 code lines = **6.5%**, reproducing the audit's 6.0% NATIVE-line classification to within rounding — so the two agree, and the logical-statement metric is the one Task 8 gates on because it is the one that rewards a big native call. |

**Measured baseline, logical-statement metric** (the numbers every task below is held to):

| file | code lines | native lines | ratio |
|-|-|-|-|
| `examples/animate_conversation.py` | 166 | 9 | 5.4% |
| `examples/animate_market_forecast.py` | 191 | 11 | 5.8% |
| `examples/animate_morph_zoo.py` | 40 | 6 | 15.0% |
| `examples/animate_painting_embeddings.py` | 146 | 11 | 7.5% |
| `examples/animate_weather_decades.py` | 196 | 11 | 5.6% |
| **five scripts, total** | **739** | **48** | **6.5%** |
| `docs/tutorials/conversation_shape.ipynb` | 186 | 11 | 5.9% |
| `docs/tutorials/market_forecast.ipynb` | 192 | 11 | 5.7% |
| `docs/tutorials/morph_shapes_zoo.ipynb` | 45 | 8 | 17.8% |
| `docs/tutorials/painting_embeddings.ipynb` | 116 | 10 | 8.6% |
| `docs/tutorials/weather_decades.ipynb` | 206 | 10 | 4.9% |

---

## Contracts this plan establishes

1. **A gallery example's job is to demonstrate the library, not to substitute for it.** Every line that re-implements a native capability (audit class **B**) or works around a gap Plans 1–3 close (class **C**) is deleted. What remains is data acquisition (**A**) and deliberate presentation (**D**), and each surviving **D** block must be something hypertools genuinely does not claim to do.

2. **Script and notebook are one deliverable.** They are edited in the same task and land in the same commit. Task 8 enforces this mechanically: the defect-marker scan and the size-budget check run over `examples/animate_*.py` **and** `docs/tutorials/*.ipynb`, so a script fixed alone fails the gate.

   **This is currently violated on all five pairs, and that is the largest single piece of work in this plan.** `d730a085` modernised four scripts and left every notebook behind, so the notebooks now *teach the private-API approach their own scripts just abandoned*. Measured:

   | notebook | private reaches still present |
   |-|-|
   | `market_forecast.ipynb` | `ani._func`, `ani._args`, `hypertools._shared` |
   | `weather_decades.ipynb` | `ani._func` |
   | `conversation_shape.ipynb` | `ani._func`, `ani._args`, `SentenceTransformer` |
   | `morph_shapes_zoo.ipynb` | `ani._func`, `from hypertools.plot import morph` |
   | `painting_embeddings.ipynb` | `SentenceTransformer` |

3. **No private reaches where a public equivalent exists.** After this plan, no example or notebook contains `ani._func`, `ani._args`, `hypertools._shared`, `hypertools.plot.morph`, or any other undocumented name — **except** the entries of the `PRIVATE_API_EXCEPTIONS` allowlist in Task 8, each of which must carry an inline rationale in the source naming why no public path exists. Per-frame work goes through the public `on_frame=` hook (animation-core Task 7); per-segment naming goes through `title=` (animation-core Task 8).

   **Why an allowlist rather than an absolute ban (changed in v3).** v2's Contract 3 was absolute, and `d730a085` then deliberately kept two private usages in the market example *with recorded measurements*:

   - `examples/animate_market_forecast.py:204-213` — the one-time setup step that reads back the fully-revealed **antialiased** on-screen line. `ctx.datasets` is the pre-antialiasing array at a coarser resolution and fits a measurably different slope (**~2–8%, checked empirically**), so substituting it would silently change the fitted forecast geometry.
   - `examples/animate_market_forecast.py:283-287` — the PCHIP smoothing helper, which has no public re-export; reimplementing it by hand would risk drifting from what `hyp.plot` actually draws.

   Both are one-time setup, not per-frame work, and both are documented in place. The purpose of this contract is that examples must not *teach* private API as the way to do things — which a documented, allowlisted, no-public-equivalent setup step does not do. An unlisted private reach still fails the gate, so nothing new can creep in, and each retained one is reviewed rather than assumed.

   **MAINTAINER SIGN-OFF REQUIRED.** This narrows a contract the maintainer wrote. If they prefer the absolute ban, the two retentions above must be reverted in `d730a085` and this contract restored — but that trade must be made explicitly, because the measurement says the public path changes the result.

   **Ownership split with animation-core Task 9 Step 5 — both plans touch the same four `examples/animate_*.py` files, so this is explicit and reciprocal:**

   | animation-core Task 9 Step 5 | **this plan (the authority on content)** |
   |-|-|
   | **Mechanical migration only** — delete the `_func`/`_args` monkeypatches and private imports, substitute the equivalent `title=`/`on_frame=` call, leave rendered behaviour unchanged | **All narrative, visualization and notebook work** — what each example demonstrates, its prose, its figures, and the paired `docs/tutorials/*.ipynb` |
   | Touches only the docstring sentences describing the removed workaround | Owns the full docstring and narrative rewrite |
   | **Must not** assert this plan's line-count or class-mix metrics | **Task 8 owns those metrics**, and measures them only after this plan's rewrites land |

   Read it as a two-stage handoff: Plan 1 makes the examples *stop using private internals* without changing what they show; this plan then decides what they should show. Tasks 2, 3, 5 and 6 here are the final word on each file's content. If a file arrives from Plan 1 already migrated, do **not** re-migrate it — rewrite its narrative on top.

   *Do not enforce this plan's metrics from Plan 1*: measuring a file Plan 1 has migrated but this plan has not yet rewritten fails for the wrong reason, and would push Plan 1's implementer into doing editorial work that gets discarded here.

4. **Network fetches live in examples, wrapped in a fallback, never in a library test.** Every fetch follows the shape the current examples already use (`fetch_fred` in `animate_market_forecast.py`, `fetch_city_months` in `animate_weather_decades.py`): a `try/except Exception: return None` fetcher, a deterministic synthetic substitute, and a `print(...)` naming which source was used. Task 1's tests write real image files to `tmp_path` and touch no network. `image_palette()` deliberately does **not** accept a URL, so the library never fetches.

   **v3 — measured, and stronger than v2 assumed. All five examples are network-coupled**, not three, in three different severities. Blocked-connection counts are real (measured by refusing outbound sockets and running each example to completion):

   | example | blocked events | host | offline outcome |
   |-|-|-|-|
   | `animate_weather_decades` | 6 | `archive-api.open-meteo.com` | degrades, exit 0 |
   | `animate_painting_embeddings` | 7 | `commons.wikimedia.org`, `huggingface.co` | degrades, exit 0 |
   | `animate_morph_zoo` | 4 | `www.dropbox.com` (via `hyp.load`) | **HARD FAILS — `HypertoolsIOError`, exit 1** |
   | `animate_conversation` | 2 | `huggingface.co` | degrades, exit 0 |
   | `animate_market_forecast` | 1 | `fred.stlouisfed.org` | degrades, exit 0 |

   So the contract's "wrapped in a fallback" clause is **already violated by morph**: `hyp.load(name)` has no offline path and takes the whole example down. Task 6 must give it one.

   **The consequence for the gate:** a test that executes an example executes its fetches. Task 8 therefore drives a **`construct_artifact(data)` boundary** — loaders on one side, figure construction on the other — and tests only the construction half, fed by fixtures. Four of the five need **zero committed bytes** (their existing seeded synthetic fallbacks *are* the fixture); paintings needs one committed **1.7 KB** 64-px thumbnail. The whole-example run survives as an opt-in smoke test, never in the default suite. Importing an example must not fetch.

5. **Forecast scoring stays out of the library.** Standing maintainer decision, restated by the forecast-animation plan's Global Constraints (*"Forecast scoring stays OUT of the library ... accuracy/backtest logic belongs in the tutorial as legitimately custom code"*). Task 2's per-sector and overall accuracy is example code, and is budgeted and timed rather than left open-ended.

6. **Every "AFTER" number in this plan is a contracted budget, not a measurement of code that does not exist yet.** Each rewrite states `code ≤ N`; Task 8 asserts it with a committed script and a pytest module. If a rewrite cannot meet its budget, the budget is renegotiated in the plan — the assertion is never weakened to fit the code.

   **v3 changes two things here.**

   *(a) The `ratio ≥ P%` half is gone.* The per-file native-ratio floor was deleted from the gate as one of v1's Fatals (it is trivially gamed by reformatting), but ten lines went on promising it as a budget anyway. Ratio is now **reported, never gated**, and no "AFTER" line states one.

   *(b) Notebook budgets are derived, not written down.* Two of them were set *below their own script budgets* — paintings 110 vs 118, conversation 76 vs 90 — which no notebook containing that script's code can satisfy, whatever the metric does. So:

   ```
   notebook_budget = script_budget + NOTEBOOK_OVERHEAD      # NOTEBOOK_OVERHEAD = 5
   ```

   `NOTEBOOK_OVERHEAD` is measured, not guessed: the largest install cell across the five is 3 code lines, plus a 2-line display cell (`from IPython.display import HTML` + `HTML(ani.to_jshtml())`). One number is chosen per task — the script budget — and the notebook's follows. Verified against the prescribed content: market 113 ≤ 120, weather 60 ≤ 67, paintings 116 ≤ 123, conversation 93 ≤ 95, morph 30 ≤ 35. Still tight, and a notebook budget can never again be set below its script's.

7. **Behaviour parity with today, except where a defect is being removed.** Each rewrite keeps its example's visual identity (the market's quarter-turn and forecast fan, the weather figure's blue-cold/red-hot sweep, the paintings' spin, the conversation's one-turn-at-a-time reveal, the morph's closed loop and teapot). Where an effect is deliberately dropped because no 1.1 API expresses it, it is named in *Decisions still needed*, never quietly lost.

8. **Bind the animation, then destructure it — never the reverse.** `hyp.plot(..., animate=...)` returns a `HyperAnimation`, a `(figure, animation)` **tuple subclass**. Unpacking it directly throws the wrapper away:

   ```python
   anim = hyp.plot(...)          # a HyperAnimation -- has .on_frame(), .figure, .animation
   fig, ani = anim               # and ALSO destructures, when you want the parts

   fig, ani = hyp.plot(...)      # WRONG if you then need the wrapper:
   ani.on_frame(cb)              # AttributeError -- `ani` is a raw FuncAnimation
   ```

   `examples/animate_market_forecast.py` already shows the right idiom (`anim = hyp.plot(...)` at `:191`, then `fig, ani = anim` at `:195`), and every example that calls `.on_frame()` must use it. Task 8 enforces this: an example or notebook may not call a `HyperAnimation`-only method on a name produced by unpacking.

   The trap is genuinely easy to miss — `_save_count` *survives* unpacking, because the raw `FuncAnimation` has it — so a gate written against `ani._save_count` passes while the public API is being silently discarded.

---

## Global Constraints

- Target release: **1.1**. Nothing here ships to users until the whole 1.1 line is working.
- Run everything with the repo venv: `.venv/bin/python`. **The base anaconda python is BROKEN** (numpy/matplotlib mismatch); a bare `python`/`pytest` will fail confusingly.
- Run pytest from the repo root; `pyproject.toml` sets `testpaths = ["tests"]` and `timeout = 1200`.
- **Baseline, re-measured 2026-08-02: `2799/2801 tests collected (2 deselected)`.** (It was `2782/2784` at `065c841e`; Plan 3's Tasks 0-1 have since added 17 — 9 for `FrameHooks.add_internal`, 8 for `forecast_from_history`. The arithmetic reconciles exactly, which is the point: measure, do not carry a number forward.) (v2 said `2564`; that number was ~7 months of commits stale.) Plan 3's Tasks 0–1 have since landed (+17), so measure again at the moment this plan starts rather than trusting any number written here — the suite is moving while Plans 1–3 are implemented. This plan states its own deltas relative to whatever the suite is when it starts, and each task re-runs the whole suite.
- **Reading a file's BEFORE state: use `git show <base>:<path>`, never `git stash`.** v2 prescribed `git stash && measure && git stash pop`. That is a **data-loss hazard**, demonstrated end-to-end: with a clean tree — exactly the state at `065c841e` — `git stash` saves nothing and returns 0, and the following `git stash pop` then restores *and drops* an unrelated pre-existing stash (`Dropped refs/stash@{0}`; stash count 1 → 0; the unrelated file appears in the tree). `git show` is read-only, needs no clean tree, and leaves `git status --porcelain` byte-identical before and after.
- **New test files must be `git add`ed before running the full suite.** `tests/test_packaging_artifacts.py::test_sdist_contains_only_tracked_files_plus_allowlist` fails on any untracked file that lands in the sdist. This is the guard working, not a false positive — but it will look like an unrelated failure if the new test file is still untracked. (Observed twice while preparing this revision.)
- **Never simplify a test to make it pass.** If a test fails repeatedly, fix the code.
- **No mock objects.** Task 1's tests write real PNGs and read them back; the example-hygiene tests in Task 8 read the real committed files.
- Force `matplotlib.use("Agg")` in every matplotlib test module. There is **no** `conftest.py` in this repo.
- Every example must still run headless: `MPLBACKEND=Agg .venv/bin/python examples/<file>.py`.
- **Notebook execution recipe** (used by Tasks 2–7; the repo's `python3` kernelspec points at an unrelated venv, so a kernel must be registered from this repo first):

  ```bash
  .venv/bin/python -m ipykernel install --user --name hypertools-venv \
      --display-name "hypertools (.venv)"
  .venv/bin/python scripts/execute_tutorial.py docs/tutorials/<name>.ipynb
  ```

  `scripts/execute_tutorial.py` is created in Task 2 Step 1; it executes in place with the venv kernel and then restores `metadata.kernelspec` to the neutral `{"display_name": "Python 3", "language": "python", "name": "python3"}` the committed notebooks carry, so Colab is unaffected.
- When any behaviour changes, update the docstring/markdown in the same commit (repo rule: docs travel with code).
- Commit after every task. Branch off `dev-1.0`; never commit to `master`.
- Re-run **all** checks after any fix made to satisfy another check.
- **The working tree is clean, and four of the five examples have already moved.** (v2 said the tree was dirty with concurrent Plan 1–3 edits, and that "the five launch examples and their notebooks are untouched". Both statements were true when written and are false now.) At `065c841e` the tree is clean. Commit **`d730a085`** (2026-08-01 09:46) rewrote **market, weather, conversation and morph** — migrating each off the `ani._func`/`ani._args` monkeypatch onto the public `anim.on_frame(...)` — and left **paintings** untouched (its last change is `4d1d2223`). It also touched `hypertools/plot/animation_context.py`.

  Consequences, and they are structural rather than cosmetic:

  | task | script | notebook | what this plan must do |
  |-|-|-|-|
  | **2** market | partially landed | unchanged — **out of sync** | **REBASE**, and it is **BLOCKED**: see below |
  | **3** weather | partially landed | unchanged — **out of sync** | **REBASE** (light) |
  | **4** paintings | untouched; v2's baseline verified still accurate | in sync with its script | **WRITE AS-IS** (still gated on Task 1) |
  | **5** conversation | partially landed | unchanged — **out of sync** | **REBASE** |
  | **6** morph | **fully landed** — Step 1 is already done verbatim, docstring included | unchanged — **out of sync** | **REBASE**: delete Step 1, keep the rest |

  **Before starting any task, read the current file** (`git show d730a085~1:<path>` for the pre-migration state) and treat what is on disk as the baseline. Do **not** apply a prescribed "replace the file entirely" block without first reconciling it against what landed — for Task 5 that would replace working code with a crash (Contract 8).

- **Task 2 is blocked until Plan 3 Task 5 lands.** Its prescribed call passes `forecast_trail=16`, and `forecast_trail` is **absent** from `plot()`'s 75 parameters today (verified with `inspect.signature`). This is the concrete reason for the maintainer's ordering — **Plan 3 before Plan 4**.

---

## Prerequisites

Plans 1, 2 and 3 must land first. Per task:

| this plan's task | depends on | why |
|-|-|-|
| **Task 1** (palette from image) | *(code: none; docs: animation-core)* | The code is a pure library addition to `hypertools/plot/colors.py` and `plot.py`, and can start immediately in parallel with Plans 1–3. **Its Step 6 cannot**: it writes under a `## 1.1.0 (unreleased)` → `### Added` heading that the animation-core plan creates, and `CHANGELOG.md` has neither today (`grep -n "1.1.0\|### Added" CHANGELOG.md` → nothing). Either land animation-core's CHANGELOG step first, or have Step 6 create the heading if it is missing. |
| **Task 2** (Market) | **MultiIndex** T1 (`group_columns`), T2 (final-trace builder), T5 (column MultiIndex in `plot()`), T6 (hue as a per-trace auxiliary value), T8 (`predict=` over final traces); **Forecast-animation** T3 (narrow the `predict=` refusal), T4 (draw the per-frame forecast), T5 (`forecast_trail=`); **Animation-core** T1 (`title=` type contract) | The whole example *is* a column MultiIndex + a continuous hue through a hierarchy + one forecast per trace during a time-progressing animation. Without MultiIndex T6 the hue is discarded (`plot.py:3080-3086`); without Forecast-animation T3 the call raises `NotImplementedError` (`plot.py:2748-2756`). |
| **Task 3** (Weather) | *(none strictly)* — verified to run on today's `dev-1.0`; **Animation-core** T1 for the `title=` contract | The paper-style call already works today. Sequence it after Plan 1 only so the whole 1.1 line is tested together. |
| **Task 4** (Paintings) | **Task 1** (palette from image); **Animation-core** T1 (`title=`) | `color=` per cloud comes from `image_palette`; the hand-rolled title becomes `title=`. |
| **Task 5** (Conversation) | **Animation-core** T5 (`order='serial'`), T7 (`on_frame=` + `HyperAnimation.on_frame`), T8 (per-segment `title=`), T4 (plotly serial+trail parity) | `animate=True, order='serial', chemtrails=True` is exactly Animation-core T4+T5; the recency fade moves onto the public `on_frame=` hook; the caption/speaker artists are replaced by per-segment `title=`. |
| **Task 6** (Morph) | **Animation-core** T8 (per-segment `title=`), T3 (`morph_samples` guard) | The private `hypertools.plot.morph` reach is exactly what T8 replaces; T3 makes the explicit `morph_samples=N` load-bearing rather than incidental. |
| **Task 7** (15 older tutorials) | *(none)* | Every fix uses API that exists on `dev-1.0` today (`save_path='*.gif'`, `vectorizer=<hf-id>`, `ax=`, `manip='Smooth'`, `hyp.plot`, `hyp.describe`). Can run in parallel with Plans 1–3. |
| **Task 8** (verification) | Tasks 1–7 | It measures them. |

---

## File Structure

| file | responsibility | change |
|-|-|-|
| `hypertools/plot/colors.py` | `image_palette()` + the `'image:<path>'` palette spelling | modify |
| `hypertools/plot/plot.py` | `palette=` docstring entry (`plot.py:1066`) | modify |
| `pyproject.toml` | declare `pillow>=8` explicitly (already transitive via matplotlib) | modify |
| `docs/api.rst` | document `image_palette` under a new "Colors" section | modify |
| `tests/plot/test_image_palette.py` | palette-from-image, incl. the largest-cluster regression | create |
| `tests/test_examples_are_native.py` | defect-marker scan + ratio gate over examples and notebooks | create |
| `scripts/measure_native_ratio.py` | the committed measurement used by the gate and by hand | create |
| `scripts/execute_tutorial.py` | execute a notebook in place with the venv kernel, restore kernelspec | create |
| `scripts/generate_gallery_thumbs.py` | add the five launch examples to `MPL_ANIMS` | modify |
| `examples/animate_market_forecast.py` + `docs/tutorials/market_forecast.ipynb` | the MultiIndex showcase | rewrite |
| `examples/animate_weather_decades.py` + `docs/tutorials/weather_decades.ipynb` | the paper figure | rewrite |
| `examples/animate_painting_embeddings.py` + `docs/tutorials/painting_embeddings.ipynb` | native text + native palette | rewrite |
| `examples/animate_conversation.py` + `docs/tutorials/conversation_shape.ipynb` | native text + serial + per-segment titles | rewrite |
| `examples/animate_morph_zoo.py` + `docs/tutorials/morph_shapes_zoo.ipynb` | native per-segment titles | rewrite |
| `docs/tutorials/{conversation_trajectories,hugging_face_embeddings,wikipedia_embeddings,modern_sklearn_dynamics,stock_forecasting,projectile_kalman,analyze,reduce}.ipynb` | the recurring fixes | modify |
| `docs/tutorials.rst` | thumbnails for the five launch tutorials | modify |
| `CHANGELOG.md` | the 1.1 entry for `image_palette` | modify |

---

## Task 1: Native palette-from-image

**The gap.** `examples/animate_painting_embeddings.py:120-146` downloads a canvas, k-means-clusters its pixels, and picks `km.cluster_centers_[np.argmax(counts)]` — **the largest cluster**. In a painting the largest cluster is the background. Reproduced today on a synthetic 90%-beige/10%-vivid-red image: the helper returns `[0.784, 0.769, 0.737]` (beige). The example then applies a luminance clamp (`lum > 0.5 → rgb * 0.5/lum`, lines 141-143) to make the muted result legible — a hack that exists only because the wrong colour was chosen. Nothing in `hypertools/` does any of this; there is no image handling in the package at all.

**API design, and why.** Two entry points over **one** implementation:

1. **`hypertools.plot.colors.image_palette(image, n_colors=6, resize=200, random_state=0)`** → `(k, 3)` float RGB in `[0, 1]`, `k ≤ n_colors`, **most visually salient first**. It sits beside the two existing public palette helpers in the same module (`get_palette_colors`, `colors.py:227`; `continuous_colormap`, `colors.py:250`), which is where "what colour is group *i*" already lives. It accepts a **local path, a PIL image, or an (H, W, 3) array** — deliberately *not* a URL, so the library never performs a network fetch (Contract 4); the paintings example keeps its own download-and-cache (class **A**) and hands over a cached path.

2. **`palette='image:<path>'`**, intercepted in **two** places, because `palette=` reaches seaborn by two independent routes. *(v2 claimed one. That was measured and is false — see below.)*

   **(a) `_get_palette`'s string branch** (`colors.py:305-306`) serves everything that resolves colours through `colors.py`: `mat2colors`' categorical (`colors.py:106`), continuous (`:118`) and matrix (`:158`) paths, `get_palette_colors` (`:246`), `continuous_colormap` (`:259`), the MultiIndex colours (`multiindex.py:151`), and both colorbars (`plotly_backend.py:2468`, `plot.py:5383`).

   **(b) `_seaborn_palette_arg`** (`plot.py:113`) serves the five call sites that hand `palette=` to seaborn **raw**, never touching `colors.py`: `plot.py:208-209`, `:4118-4119`, `:4657-4658`, `:4767-4768`, and — the fatal one — **`:4825-4826`'s `sns.set_palette`, which runs on EVERY matplotlib plot call regardless of hue**.

   **Measured red state with only (a) patched: 0 of 6 scenarios pass.** Categorical/matplotlib, categorical/plotly, continuous, matrix, direct `palette=`, and 9-category all raise `ValueError: 'image:…' is not a valid palette name` from `seaborn/palettes.py:237` via `plot.py:4825`. v2's own test file reported only 2 failures, which flattered it — its "continuous hue" case called `continuous_colormap()` directly rather than `hyp.plot()`, and its "missing file" case asserted an error anyway. With both interceptions and a dynamic colour count: **6 of 6**.

**The ordering rule (this is the contract, and it is what fixes the bug).** For each k-means centre compute `frac` (its share of pixels) and `chroma = max(r,g,b) - min(r,g,b)` (distance from grey — the numerator of HSV saturation). Order by **descending `frac × chroma`**. A large muted background scores near zero; a smaller vivid region wins. If *every* centre is achromatic (`max(chroma) < 0.02`, i.e. a greyscale image), fall back to descending `frac`, because a grey image genuinely has no vivid colour and "largest" is then the right answer. Near-duplicate centres (equal to 3 decimals) are dropped, so `n_colors` is an **upper** bound.

Measured on the prototype: the 90/10 beige-red image → `[[0.863, 0.078, 0.078], [0.784, 0.769, 0.737]]` (red first, beige retained but demoted); a greyscale 80/20 image → `[[0.118, 0.118, 0.118], [0.784, 0.784, 0.784]]` (population order); a six-stripe image → six distinct colours; array and PIL inputs give identical results; repeated calls are bit-identical.

**Pillow.** `matplotlib>=3.9.0` is a core dependency and requires `pillow>=8`, so Pillow is already present in every install (`Pillow 12.1.0` in this venv). This task declares it explicitly in `pyproject.toml` anyway — a library that imports a package should say so — at zero install cost.

**Files:**
- Modify: `hypertools/plot/colors.py`, `hypertools/plot/plot.py` (the `palette` docstring at `plot.py:1066`), `pyproject.toml`, `docs/api.rst`, `CHANGELOG.md`
- Test: `tests/plot/test_image_palette.py` (create)

**Interfaces:**
- Produces `image_palette(image, n_colors=6, resize=200, random_state=0)` → `np.ndarray (k, 3)`, `k ≤ n_colors`, salience-ordered.
- Produces the module constants `IMAGE_PALETTE_PREFIX = 'image:'` and `IMAGE_PALETTE_N = 6`.
- Consumed by Task 4.

- [ ] **Step 1: Write the failing test**

```python
# tests/plot/test_image_palette.py
"""Palette extraction from an image, and the `palette='image:<path>'` spelling.

The ordering rule is the point: `examples/animate_painting_embeddings.py:138-140`
picked `km.cluster_centers_[np.argmax(counts)]` -- the LARGEST cluster -- which
in a painting is the background. Measured on the synthetic image below, that
rule returns the beige (0.784, 0.769, 0.737); this module pins the vivid red
(0.863, 0.078, 0.078) as the FIRST colour instead.

No network: every image is written to `tmp_path` and read back.
"""
import matplotlib
matplotlib.use("Agg")

import numpy as np
import pytest
from PIL import Image
from matplotlib.colors import to_rgb

import hypertools as hyp
from hypertools.plot.colors import (IMAGE_PALETTE_N, continuous_colormap,
                                    get_palette_colors, image_palette)

BEIGE = (0.784, 0.769, 0.737)
VIVID = (0.863, 0.078, 0.078)


def _png(tmp_path, arr, name):
    path = tmp_path / name
    Image.fromarray(arr.astype(np.uint8)).save(path)
    return str(path)


def painting_png(tmp_path, name='painting.png'):
    """90% muted beige 'canvas', 10% vivid red 'subject'."""
    arr = np.zeros((100, 100, 3), np.uint8)
    arr[:, :] = (200, 196, 188)
    arr[:10, :] = (220, 20, 20)
    return _png(tmp_path, arr, name)


def grey_png(tmp_path, name='grey.png'):
    arr = np.zeros((100, 100, 3), np.uint8)
    arr[:, :] = (30, 30, 30)
    arr[:20, :] = (200, 200, 200)
    return _png(tmp_path, arr, name)


def six_png(tmp_path, name='six.png'):
    arr = np.zeros((120, 120, 3), np.uint8)
    for i, c in enumerate([(255, 0, 0), (0, 255, 0), (0, 0, 255),
                           (255, 255, 0), (255, 0, 255), (0, 255, 255)]):
        arr[i * 20:(i + 1) * 20, :] = c
    return _png(tmp_path, arr, name)


def nine_png(tmp_path, name='nine.png'):
    """NINE genuinely distinct bands, so 'not capped at six' tests the cap
    rather than the interpolation fallback."""
    arr = np.zeros((180, 180, 3), np.uint8)
    for i, c in enumerate([(255, 0, 0), (0, 255, 0), (0, 0, 255),
                           (255, 255, 0), (255, 0, 255), (0, 255, 255),
                           (255, 128, 0), (128, 0, 255), (0, 128, 128)]):
        arr[i * 20:(i + 1) * 20, :] = c
    return _png(tmp_path, arr, name)


def one_colour_png(tmp_path, name='one.png'):
    arr = np.full((60, 60, 3), 200, np.uint8)
    return _png(tmp_path, arr, name)


def _ax(fig):
    return [a for a in fig.axes if hasattr(a, 'zaxis')][0]


# --- the extraction itself ---------------------------------------------------

def test_returns_rgb_floats_in_the_unit_range(tmp_path):
    pal = image_palette(painting_png(tmp_path))
    assert pal.ndim == 2 and pal.shape[1] == 3
    assert pal.dtype == np.float64
    assert pal.min() >= 0.0 and pal.max() <= 1.0


def test_a_vivid_minority_colour_beats_the_muted_background(tmp_path):
    """THE regression test. Largest-cluster ordering returns the beige."""
    pal = image_palette(painting_png(tmp_path))
    assert pal[0] == pytest.approx(VIVID, abs=0.02)


def test_the_background_is_kept_but_demoted(tmp_path):
    """Not discarded -- just not first. A palette should still describe the
    whole canvas."""
    pal = image_palette(painting_png(tmp_path))
    assert any(np.allclose(c, BEIGE, atol=0.02) for c in pal)
    assert not np.allclose(pal[0], BEIGE, atol=0.02)


def test_a_greyscale_image_falls_back_to_population_order(tmp_path):
    """With no chroma anywhere, `frac * chroma` is all zeros and 'largest'
    IS the right answer: the 80% dark tone leads."""
    pal = image_palette(grey_png(tmp_path))
    assert pal[0] == pytest.approx((0.118, 0.118, 0.118), abs=0.02)


def test_n_colors_is_an_upper_bound_and_colours_are_distinct(tmp_path):
    pal = image_palette(six_png(tmp_path), n_colors=6)
    assert len(pal) == 6
    assert len(np.unique(np.round(pal, 3), axis=0)) == 6
    assert len(image_palette(six_png(tmp_path), n_colors=3)) == 3


def test_an_image_with_fewer_unique_colours_returns_fewer(tmp_path):
    """Two unique pixel colours cannot yield six clusters; asking for six
    must NOT raise or emit sklearn's ConvergenceWarning."""
    import warnings
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        pal = image_palette(painting_png(tmp_path), n_colors=6)
    assert len(pal) == 2
    assert not [w for w in caught if 'ConvergenceWarning' in type(w.message).__name__]


def test_accepts_a_pil_image_and_a_numpy_array(tmp_path):
    arr = np.zeros((100, 100, 3), np.uint8)
    arr[:, :] = (200, 196, 188)
    arr[:10, :] = (220, 20, 20)
    from_path = image_palette(painting_png(tmp_path))
    assert image_palette(arr)[0] == pytest.approx(from_path[0], abs=0.02)
    assert image_palette(Image.fromarray(arr))[0] == pytest.approx(
        from_path[0], abs=0.02)


def test_a_float_array_in_unit_range_is_accepted(tmp_path):
    arr = np.zeros((100, 100, 3), float)
    arr[:, :] = (200 / 255, 196 / 255, 188 / 255)
    arr[:10, :] = (220 / 255, 20 / 255, 20 / 255)
    assert image_palette(arr)[0] == pytest.approx(VIVID, abs=0.02)


def test_extraction_is_deterministic(tmp_path):
    path = painting_png(tmp_path)
    assert np.allclose(image_palette(path), image_palette(path))


def test_a_missing_file_names_the_path(tmp_path):
    with pytest.raises(FileNotFoundError, match='no_such_canvas.jpg'):
        image_palette(str(tmp_path / 'no_such_canvas.jpg'))


def test_n_colors_must_be_a_positive_integer(tmp_path):
    with pytest.raises(ValueError, match='positive integer'):
        image_palette(painting_png(tmp_path), n_colors=0)


# --- the `palette='image:<path>'` spelling ------------------------------------

def test_palette_string_resolves_through_get_palette_colors(tmp_path):
    """One interception in _get_palette must serve every palette consumer."""
    path = painting_png(tmp_path)
    resolved = get_palette_colors(f'image:{path}', 2)
    assert resolved[0] == pytest.approx(VIVID, abs=0.02)


def test_palette_string_colours_a_categorical_hue(tmp_path):
    """Reads ax.LINES, not ax.collections. A `fmt='.'` plot draws `Line2D`
    into `ax.lines`; the only collections on a 3-D axes are pane/grid
    artists whose facecolor array is EMPTY. v2 harvested those instead, so
    the filter emptied the list and `np.vstack([])` raised `need at least
    one array to concatenate` -- against a CORRECT implementation. Measured
    while auditing: the line colors were exactly right the whole time."""
    path = painting_png(tmp_path)
    rng = np.random.default_rng(0)
    ds = [rng.normal(size=(10, 4)) for _ in range(2)]
    fig = hyp.plot(ds, '.', hue=['a'] * 10 + ['b'] * 10,
                   palette=f'image:{path}', show=False)
    drawn = [to_rgb(ln.get_color()) for ln in _ax(fig).lines]
    assert drawn, 'no line artists were drawn'
    assert any(np.allclose(c, VIVID, atol=0.02) for c in drawn)


def test_a_categorical_hue_is_not_capped_at_six_categories(tmp_path):
    """`IMAGE_PALETTE_N = 6` is the CONTINUOUS anchor count, not a limit on
    categories. With a fixed count this raised `palette= supplies 6 color(s)
    but 9 are required`. Uses a NINE-colour image, so this tests the cap and
    not the interpolation fallback."""
    path = nine_png(tmp_path)
    rng = np.random.default_rng(0)
    labels = [c for c in 'abcdefghi' for _ in range(4)]
    fig = hyp.plot([rng.normal(size=(36, 4))], '.', hue=labels,
                   palette=f'image:{path}', show=False)
    drawn = {to_rgb(ln.get_color()) for ln in _ax(fig).lines}
    assert len(drawn) == 9, f'expected 9 distinct colours, got {len(drawn)}'


def test_an_image_with_too_few_colours_interpolates_rather_than_repeats(
        tmp_path):
    """Cycling would give two categories the SAME colour -- the ambiguity
    the short-list error exists to prevent (colors.py:332-335). A caller
    cannot add colours to an image, so the anchors are blended up instead.
    `painting_png` is genuinely two-tone, so 5 categories need 3 blended."""
    path = painting_png(tmp_path)
    rng = np.random.default_rng(0)
    labels = [c for c in 'abcde' for _ in range(4)]
    fig = hyp.plot([rng.normal(size=(20, 4))], '.', hue=labels,
                   palette=f'image:{path}', show=False)
    drawn = [to_rgb(ln.get_color()) for ln in _ax(fig).lines]
    assert len({tuple(np.round(c, 6)) for c in drawn}) == 5, (
        'repeated colours would make two categories indistinguishable')
    assert np.allclose(drawn[0], VIVID, atol=0.02), (
        'the most salient anchor must survive interpolation, and lead')


def test_a_single_colour_image_raises_rather_than_inventing_colours(
        tmp_path):
    """The one case interpolation cannot honestly serve."""
    path = one_colour_png(tmp_path)
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match='single dominant color'):
        hyp.plot([rng.normal(size=(20, 4))], '.',
                 hue=[c for c in 'abcde' for _ in range(4)],
                 palette=f'image:{path}', show=False)


def test_palette_string_blends_anchors_for_a_continuous_hue(tmp_path):
    """A short list + a continuous hue is seaborn blend_palette semantics
    (colors.py:323-331), so an image palette gives a gradient between its
    extracted anchors -- no error about 'too few colors'."""
    path = six_png(tmp_path)
    cmap = continuous_colormap(f'image:{path}', n_bins=100)
    assert cmap.N == 100
    assert len(np.unique(np.round(cmap(np.linspace(0, 1, 100))[:, :3], 3),
                         axis=0)) > IMAGE_PALETTE_N


def test_palette_string_with_a_missing_file_names_the_file(tmp_path):
    rng = np.random.default_rng(0)
    ds = [rng.normal(size=(10, 4))]
    with pytest.raises(FileNotFoundError, match='gone.png'):
        hyp.plot(ds, '.', hue=np.arange(10),
                 palette=f"image:{tmp_path / 'gone.png'}", show=False)


def test_plotly_backend_accepts_an_image_palette(tmp_path):
    """Backend parity: the interception is in colors.py, above both backends."""
    pytest.importorskip('plotly')
    path = painting_png(tmp_path)
    rng = np.random.default_rng(0)
    ds = [rng.normal(size=(10, 4)) for _ in range(2)]
    hyp.set_interactive_backend('plotly')
    try:
        fig = hyp.plot(ds, '.', hue=['a'] * 10 + ['b'] * 10,
                       palette=f'image:{path}', show=False)
    finally:
        hyp.set_interactive_backend('matplotlib')
    assert len(fig.data) >= 2
```

- [ ] **Step 2: Run the test and confirm it fails**

Run: `.venv/bin/python -m pytest tests/plot/test_image_palette.py -v`

Expected: **collection FAILS** with `ImportError: cannot import name 'IMAGE_PALETTE_N' from 'hypertools.plot.colors'`. If you stub the import out to see individual failures, every `image_palette` test fails with `NameError`, and the four `palette='image:...'` tests fail with `ValueError: 'image:/.../painting.png' is not a valid palette name` (measured today).

- [ ] **Step 3: Implement `image_palette`**

Add to `hypertools/plot/colors.py`, immediately after `continuous_colormap` (which ends at `colors.py:260`) and before the `_CYCLIC_PALETTES` block:

```python
#: How many anchor colors `palette='image:<path>'` extracts for a CONTINUOUS
#: mapping, which asks `_get_palette` for `n_bins` (100) colors -- clustering
#: an image into 100 groups is both slow and meaningless, so it takes this
#: few and lets the short-list blending (colors.py:323-331) build the
#: gradient. A CATEGORICAL or matrix mapping instead extracts exactly as many
#: colors as it has categories, so the number of groups is NOT capped at this
#: value; see `_image_palette_list`.
IMAGE_PALETTE_N = 6

#: Prefix that marks a `palette=` string as "extract this from an image".
#: Seaborn/matplotlib palette names never contain a colon, so there is no
#: collision; an unmatched name still reaches seaborn and raises its own
#: "is not a valid palette name" error.
IMAGE_PALETTE_PREFIX = 'image:'

#: Below this chroma (max(RGB) - min(RGB)) an image has no colour to be
#: salient ABOUT, so `image_palette` orders by population instead.
_ACHROMATIC_EPS = 0.02


def _image_pixels(image, resize):
    """(n_pixels, 3) float RGB in [0, 1] from a path, PIL image, or array."""
    import os

    from PIL import Image

    if isinstance(image, np.ndarray):
        arr = image
        if arr.dtype.kind == 'f':
            arr = np.clip(arr, 0.0, 1.0) * 255.0
        im = Image.fromarray(arr.astype(np.uint8)).convert('RGB')
    elif hasattr(image, 'convert'):          # a PIL.Image.Image
        im = image.convert('RGB')
    else:
        path = os.path.expanduser(os.fspath(image))
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"image_palette() could not find an image at {path!r}. It "
                "takes a LOCAL path, a PIL image, or an (H, W, 3) array -- "
                "hypertools never downloads the image for you, so fetch and "
                "cache it yourself first.")
        im = Image.open(path).convert('RGB')
    im.thumbnail((int(resize), int(resize)))
    return np.asarray(im, dtype=np.float64).reshape(-1, 3) / 255.0


def image_palette(image, n_colors=IMAGE_PALETTE_N, resize=200, random_state=0):
    """Extract a color palette from an image, most VISUALLY SALIENT first.

    Parameters
    ----------
    image : str, pathlib.Path, PIL.Image.Image, or numpy array
        A LOCAL image file, an already-open PIL image, or an (H, W, 3) array
        (uint8 0-255, or float 0-1). URLs are deliberately not accepted:
        hypertools does not fetch images, so download and cache the file
        yourself and pass the cached path.
    n_colors : int
        UPPER bound on how many colors to return (default 6). Fewer come
        back when the image has fewer distinct colors, or when two cluster
        centers coincide to 3 decimal places.
    resize : int
        Longest edge the image is thumbnailed to before clustering
        (default 200). Clustering cost is linear in pixel count.
    random_state : int
        Seed for the k-means fit, so repeated calls are identical.

    Returns
    -------
    palette : numpy.ndarray
        (k, 3) float RGB in [0, 1], k <= n_colors, ordered most salient
        first.

    Notes
    -----
    Salience is ``pixel_fraction * chroma``, where
    ``chroma = max(r, g, b) - min(r, g, b)`` measures distance from grey.
    Ordering by pixel fraction ALONE returns a painting's background --
    which is exactly the bug this function exists to avoid. When every
    cluster is achromatic (max chroma < 0.02, i.e. a greyscale image) the
    ordering falls back to pixel fraction, because a grey image has no
    vivid color and "largest" is then the right answer.

    Examples
    --------
    >>> from hypertools.plot.colors import image_palette
    >>> image_palette('starry_night.jpg')[0]        # doctest: +SKIP
    array([0.16, 0.24, 0.55])

    The same extraction is reachable declaratively from any plotting call
    that takes a palette::

        hyp.plot(x, hue=values, palette='image:starry_night.jpg')
    """
    from sklearn.cluster import KMeans

    if (not isinstance(n_colors, (int, np.integer))
            or isinstance(n_colors, bool) or n_colors < 1):
        raise ValueError(
            f"n_colors= must be a positive integer; got {n_colors!r}")
    px = _image_pixels(image, resize)
    if len(px) == 0:
        raise ValueError("image_palette() got an image with no pixels")
    # cap k at the number of DISTINCT colors: asking k-means for more
    # clusters than there are distinct points emits a ConvergenceWarning
    # and returns duplicate centers
    k = int(min(n_colors, len(np.unique(px, axis=0))))
    km = KMeans(n_clusters=k, n_init=4, random_state=random_state).fit(px)
    centers = np.clip(km.cluster_centers_, 0.0, 1.0)
    frac = np.bincount(km.labels_, minlength=k) / len(px)
    chroma = centers.max(axis=1) - centers.min(axis=1)
    score = frac if chroma.max() < _ACHROMATIC_EPS else frac * chroma
    out, seen = [], set()
    for i in np.argsort(-score, kind='stable'):
        key = tuple(np.round(centers[i], 3))
        if key in seen:
            continue
        seen.add(key)
        out.append(centers[i])
    return np.asarray(out, dtype=float)
```

- [ ] **Step 4: Intercept the `'image:<path>'` spelling at BOTH resolvers**

Three patches. Patches 1–2 are `colors.py`; patch 3 is the one v2 missed entirely, without which every scenario still fails.

**Patch 1 of 3 — `hypertools/plot/colors.py`, new helper.** Insert immediately **before** `def _get_palette` (`colors.py:287`):

```python
def _image_palette_list(source, n_colors, sns, continuous):
    """Colors for a `palette='image:<path>'` string, as a list `_get_palette`
    can then handle exactly like any other color list.

    How many colors are EXTRACTED depends on the mapping. A categorical or
    matrix mapping needs one color per category, so exactly `n_colors`
    anchors are pulled: k-means with k = the number of categories is the
    best k-color summary of that image, and extracting a FIXED count would
    instead cap every plot at that many categories. A CONTINUOUS mapping
    asks for `n_bins` (100) colors, and clustering an image into 100 groups
    is both slow and meaningless, so it takes `IMAGE_PALETTE_N` anchors and
    lets the short-list blending below build the gradient from them.

    An image can hold FEWER distinct colors than there are categories (a
    two-tone image, nine groups). Unlike a user-supplied short list -- which
    raises, because the user can simply pass more colors -- a caller cannot
    add colors to an image, so the anchors are interpolated up to `n_colors`
    with the same ``blend_palette`` semantics the continuous path already
    uses (F02-006/F24-017). Interpolating keeps every category a DIFFERENT
    color and leaves the most salient anchor first; cycling the anchors
    would silently give two categories the same color, which is the
    ambiguity the short-list error exists to prevent. A single-color image
    is the one case interpolation cannot serve, and it raises."""
    colors = [tuple(c) for c in image_palette(
        source, n_colors=IMAGE_PALETTE_N if continuous else max(n_colors, 1))]
    if continuous or len(colors) >= n_colors:
        return colors
    if len(colors) == 1:
        raise ValueError(
            f"palette='{IMAGE_PALETTE_PREFIX}{source}' yielded 1 color but "
            f"{n_colors} are required (one per category/component); that "
            "image has a single dominant color, so pass a more colorful "
            "image, an explicit list of colors, or a palette name")
    return [tuple(np.asarray(c)[:3])
            for c in sns.blend_palette(colors, n_colors)]
```

**Patch 2 of 3 — `hypertools/plot/colors.py`, the `_get_palette` string branch.** Replaces `colors.py:305-306`:

```python
    if isinstance(palette, str):
        if palette.startswith(IMAGE_PALETTE_PREFIX):
            # resolve to a color LIST and fall through to the list handling
            # below, so a continuous mapping blends the extracted anchors
            # into its gradient exactly as it would any short list
            palette = _image_palette_list(
                palette[len(IMAGE_PALETTE_PREFIX):].strip(),
                n_colors, sns, continuous)
        else:
            return sns.color_palette(palette, n_colors)
```

> **The fall-through is load-bearing.** Returning `_image_palette_list(...)` directly from this branch instead of falling through breaks continuous hue with `IndexError: index 10 is out of bounds for axis 0 with size 6` — the 6 anchors never get blended up to `n_bins`. This was caught by *running* scenario 3, not by inspection.

**Patch 3 of 3 — `hypertools/plot/plot.py`, `_seaborn_palette_arg`.** Replaces `plot.py:113-124`. **This is the patch v2 lacked**, and without it all six scenarios still raise, because `sns.set_palette` at `plot.py:4825` runs on every matplotlib plot call:

```python
def _seaborn_palette_arg(palette, n_colors):
    """`palette` in a form seaborn's `color_palette`/`set_palette` accept.

    plot() documents palette= as a name, a list of colors, or a matplotlib
    `Colormap` (F02-011); seaborn handles the first two natively but not a
    Colormap INSTANCE, so that one is pre-sampled to `n_colors` RGB tuples
    via `get_palette_colors` (the same resolution `mat2colors`/the colorbar
    use, keeping every path's colors identical).

    An ``'image:<path>'`` string is the same kind of case: seaborn has no
    idea what it means and raises "is not a valid palette name", so it is
    pre-resolved through `get_palette_colors` too. This is the SECOND
    interception `palette=` needs -- every call site below hands its palette
    straight to seaborn without going through `colors._get_palette`, so
    intercepting only there would leave `sns.set_palette` (and so EVERY
    matplotlib plot call) raising on an image palette."""
    from matplotlib.colors import Colormap

    from .colors import IMAGE_PALETTE_PREFIX
    if isinstance(palette, Colormap) or (
            isinstance(palette, str)
            and palette.startswith(IMAGE_PALETTE_PREFIX)):
        return [tuple(c) for c in get_palette_colors(palette, n_colors)]
    return palette
```

Nothing else in `colors.py` changes: `_continuous_palette` already delegates here for every non-cyclic palette, and `'image:...'` is not in `_CYCLIC_PALETTES`. `plot.py` changes in exactly one function — `_seaborn_palette_arg` — because all five raw-seaborn call sites already route through it.

- [ ] **Step 5: Run the test and confirm it passes**

Run: `.venv/bin/python -m pytest tests/plot/test_image_palette.py -v`
Expected: **19 passed.**

- [ ] **Step 6: Declare Pillow, and document**

In `pyproject.toml`, add to `dependencies` (after `"seaborn>=0.13.0",`):

```toml
    # Pillow is already a hard requirement of matplotlib>=3.9 (`pillow>=8`),
    # so this adds no install weight -- but hypertools now imports PIL
    # itself (hypertools/plot/colors.py: image_palette / palette='image:...'),
    # and a library that imports a package declares it.
    "pillow>=8",
```

In `plot()`'s docstring, extend the `palette` entry at ``_seaborn_palette_arg`'s neighbour, plot.py:1066-1078` with:

```
        A palette string of the form ``'image:<path>'`` extracts colors from
        a LOCAL image file instead (``palette='image:starry_night.jpg'``):
        six anchor colors, ordered most visually salient first, so a
        painting's vivid subject leads and its muted background follows.
        For a continuous ``hue`` those anchors are blended into a gradient
        exactly as any short color list is. See
        ``hypertools.plot.colors.image_palette`` for the extraction itself
        (and to choose a different number of colors). hypertools never
        downloads the image: fetch and cache it yourself, then pass the path.
```

In `docs/api.rst`, add a **Colors** section after **Plot** (`api.rst:108-116`):

```rst
Colors
------------------

.. autofunction:: hypertools.plot.colors.image_palette

.. autofunction:: hypertools.plot.colors.get_palette_colors

.. autofunction:: hypertools.plot.colors.continuous_colormap
```

In `CHANGELOG.md`, under the `## 1.1.0 (unreleased)` → `### Added` heading created by the animation-core plan:

```markdown
- `hypertools.plot.colors.image_palette(image, n_colors=6)` extracts a color
  palette from a local image (path, PIL image, or array), ordered most
  visually salient first (`pixel_fraction * chroma`), so a painting's vivid
  subject leads and its muted background follows -- ordering by pixel share
  alone returns the background. Reachable declaratively from any plotting
  call as `palette='image:<path>'`, on both backends and on every color
  path (categorical, continuous, matrix hue, and the colorbar). hypertools
  never downloads the image.
```

- [ ] **Step 7: Run the FULL suite (palette resolution is shared by every color path)**

Run: `.venv/bin/python -m pytest -q`
Expected: baseline + 19 (the block above defines 19 `def test_` functions and parametrizes none of them; v1 said 17, v2 said 16 — this revision adds the three colour-count tests). **`git add tests/plot/test_image_palette.py` before running the full suite**, or `test_sdist_contains_only_tracked_files_plus_allowlist` fails on the untracked file; that is the packaging guard working, not a false positive. Pay attention to `tests/test_colors.py`, `tests/plot/test_colors_module.py` and `tests/test_colorbar.py`: any test that asserts `_get_palette`'s string branch is a straight seaborn passthrough must still pass, because the non-`image:` path is byte-identical.

- [ ] **Step 8: Rebuild the docs (a new autodoc section was added)**

Run: `cd docs && MPLBACKEND=Agg ../.venv/bin/python -m sphinx -b html -W -E -a . _build/html 2>&1 | tail -20`
Expected: build succeeds with **0 warnings** (the RTD-parity bar the 1.0 release gate enforces).

- [ ] **Step 9: Commit**

```bash
git add hypertools/plot/colors.py hypertools/plot/plot.py pyproject.toml \
        docs/api.rst CHANGELOG.md tests/plot/test_image_palette.py
git commit -m "feat(colors): image_palette() + palette='image:<path>', salience-ordered"
```

---

## Task 2: Market — the MultiIndex showcase

> **v3: THE SCRIPT HALF IS PARTIALLY LANDED.** `d730a085` migrated this file off the `ani._func`/`ani._args` monkeypatch onto the public `anim.on_frame(...)`. Read the file on disk before doing anything; `git show d730a085~1:examples/animate_market_forecast.py` shows the pre-migration state. Do **not** apply "replace the file entirely" without reconciling first — it would delete the migration and the two evidence-bearing rationales Contract 3 now allowlists.

**BEFORE — re-measured 2026-08-02 at `065c841e`, with the docstring-aware metric:**

| file | v2 said | actual now |
|-|-|-|
| `examples/animate_market_forecast.py` | 355 raw, 191 code, 11 native (5.8%) | **376 raw, 191 code, 11 native (5.8%)** |
| `docs/tutorials/market_forecast.ipynb` | 192 code, 11 native (5.7%), "0 of 7 executed" | **187 code, 11 native (5.9%), 4 of 7 cells carry output** |

Audit classification (unchanged, still accurate for the parts this task rewrites): A=61 **B=17 C=100** D=14 NATIVE=7.

**Already done by `d730a085`:** the per-frame monkeypatch is gone; the decorator is a `FrameContext` callback registered on the public hook (`def decorate(ctx):` and `anim.on_frame(decorate)`). **Everything else in this task remains** — the 5 FRED series are still not a 24-ticker MultiIndex, `predict=`/`t=` is still hand-rolled, and the reduce→drawn affine recovery, the 16-slot hand-drawn fan, the hand-built colorbar and the hand-built title all survive.

**BLOCKED:** the prescribed call passes `forecast_trail=16`, which does not exist in `plot()` yet (Plan 3 Task 5). Do not start this task until Plan 3 has landed.

**AFTER (contracted budget):** script **≤ 130 code lines** (115 for the rewrite + 15 for the Step 0b split); notebook **≤ 135** (= 130 + 5); **zero** defect markers. The rewrite alone measures 109; the split's cost is a placeholder until Step 0c measures this file's own.

The notebook budget is DERIVED, not written down: `script_budget + NOTEBOOK_OVERHEAD` (Contract 6b), so it can never again be set below its own script's. No ratio floor — ratio is reported, never gated (Contract 6a).

**What goes, and to what:**

| deleted | replaced by |
|-|-|
| `_frame_of`, `SLOPE`/`np.polyfit` recovery of plot's own reduce→drawn affine, `GAIN`, `CAP`, `_scale`, `BLO`/`BHI`, `_hang` (`:197-243`, class **C**) | `predict='Kalman', t=1` — the forecast is computed in the plotted space and folded into the centre/scale statistics by the library (forecast-animation Contract 4) |
| the 16-slot `hist_lines` fan (`:279-296`, class **C**) | `forecast_trail=16` (forecast-animation Task 5) |
| `_smooth` + `from hypertools._shared.helpers import antialias_line` (`:265-276`, class **B**) | native forecast antialiasing (``_draw_forecast_overlays`, plot.py:158-165`) |
| hand-built `ScalarMappable` + `fig.colorbar` + `set_label` (`:297-301`, class **B**) | `colorbar={'label': ...}` (`plot.py:1189`) |
| `fig.text(...)` title (`:303-304`, class **B**) | `title=` (`plot.py:1209`) |
| `_wrapped` + `ani._func = _wrapped` + `ani._args[1][0]` (`:199-213`, `:323-356`, class **C**) | nothing — there is no per-frame work left |
| a hand-thinned single equal-weight index over 5 FRED series | a `(Market, Sector, Ticker)` column MultiIndex over **24 tickers**, expanded natively into 6 sector traces + 1 market-mean trace, each with its own forecast |

**Data.** Verified today: 24/24 tickers from `https://query1.finance.yahoo.com/v8/finance/chart/<TICKER>?range=10y&interval=1d` (User-Agent header required), 2513 trading days each, 2016-07-28 → 2026-07-28. Six sectors × 4 tickers gives **equal widths**, required by `plot.py:3152-3153`. `yfinance` 1.5.1 is installed but the raw chart endpoint is used directly, so the example has no extra dependency.

**Accuracy readout.** Per Contract 5 this lives in the example. Budget measured: `hyp.predict(..., model='Kalman', t=1)` on a **60-row** rolling window, **30** anchors, **7** series (6 sectors + the market mean) = **210 fits in 7.3 s**. A 250-row window costs 30.7 s for the same loop, and the whole current example runs in 6.2 s — so 60/30 is the budget, and it is stated in the module docstring.

**Files:** rewrite `examples/animate_market_forecast.py`; rewrite `docs/tutorials/market_forecast.ipynb`; create `scripts/execute_tutorial.py`.

- [ ] **Step 0: Split the loader from the figure builder (Contract 4 / Step 0b)**

**Do this FIRST, before the rewrite below.** Task 8 Step 0b defines the contract and works the whole pattern through on weather; this step applies it to `examples/animate_market_forecast.py`. Without it `test_examples_produce_their_stated_artifact` fails on this example, and importing it fetches.

Produce exactly these three names in `examples/animate_market_forecast.py`:

| name | signature | notes |
|-|-|-|
| payload | `class Market(NamedTuple)` with fields `dates, prices, source` | self-documenting; `source` records which path was used |
| loader | `load_market(ids=FRED_IDS) -> Market` | the ONLY code here that may touch the network (fetch_fred) |
| fixture | `fixture_data() -> Market` | its own seeded `synthetic_basket()` — no network, no committed bytes unless stated |
| builder | `construct_artifact(data) -> HyperAnimation` | everything else, reading `data.<field>` instead of module globals. **Returns the wrapper, never the unpacked pair** (Contract 8) |

Then move every loader CALL behind a `__main__` guard, and make each fetcher honour `HYPERTOOLS_OFFLINE` by raising:

```python
if __name__ == '__main__':
    data = load_market()
    anim = construct_artifact(data)
    fig = anim.figure
```

Verify before moving on:

```bash
MPLBACKEND=Agg .venv/bin/python -c "
import importlib.util, os
os.environ['HYPERTOOLS_OFFLINE'] = '1'
spec = importlib.util.spec_from_file_location('m', 'examples/animate_market_forecast.py')
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
print('imported with no fetch; has', [n for n in ('construct_artifact', 'fixture_data') if hasattr(m, n)])
anim = m.construct_artifact(m.fixture_data()); print('frames:', anim.n_frames)"
```

Expected: it imports without touching the network and prints both names plus a frame count. If the import fetches, a loader call is still at module scope.

- [ ] **Step 1: Create the notebook execution helper (used by every task from here on)**

```python
# scripts/execute_tutorial.py
"""Execute a tutorial notebook in place with THIS repo's venv.

The user-level `python3` kernelspec points at an unrelated project's venv
(verified 2026-07-28 in `~/Library/Jupyter/kernels/python3/kernel.json`),
so `nbconvert --execute` with the default kernel does not run hypertools at
all. Register this repo's kernel once:

    .venv/bin/python -m ipykernel install --user --name hypertools-venv \
        --display-name "hypertools (.venv)"

then:

    .venv/bin/python scripts/execute_tutorial.py docs/tutorials/<name>.ipynb

Outputs are written back into the notebook (`nbsphinx_execute = 'never'`,
docs/conf.py:131, means the committed outputs are what the docs render), and
`metadata.kernelspec` is restored to the neutral python3 entry the committed
notebooks carry, so Colab is unaffected.
"""

import json
import sys

import nbformat
from nbclient import NotebookClient

NEUTRAL_KERNELSPEC = {'display_name': 'Python 3', 'language': 'python',
                      'name': 'python3'}
KERNEL = 'hypertools-venv'
TIMEOUT = 1800


def execute(path):
    nb = nbformat.read(path, as_version=4)
    original = json.loads(json.dumps(nb.metadata.get('kernelspec',
                                                     NEUTRAL_KERNELSPEC)))
    NotebookClient(nb, timeout=TIMEOUT, kernel_name=KERNEL,
                   resources={'metadata': {'path': str(path.rsplit('/', 1)[0])}}
                   ).execute()
    nb.metadata['kernelspec'] = original
    nbformat.write(nb, path)
    executed = sum(1 for c in nb.cells
                   if c.cell_type == 'code' and c.get('outputs'))
    total = sum(1 for c in nb.cells if c.cell_type == 'code')
    print(f'{path}: {executed}/{total} code cells produced output')


if __name__ == '__main__':
    if len(sys.argv) < 2:
        raise SystemExit('usage: execute_tutorial.py <notebook> [<notebook>...]')
    for target in sys.argv[1:]:
        execute(target)
```

Verify it on an untouched notebook before relying on it:

```bash
.venv/bin/python -m ipykernel install --user --name hypertools-venv \
    --display-name "hypertools (.venv)"
git stash list   # (ensure a clean tree first)
.venv/bin/python scripts/execute_tutorial.py docs/tutorials/reduce.ipynb
git diff --stat docs/tutorials/reduce.ipynb
git checkout -- docs/tutorials/reduce.ipynb
```
Expected: prints `docs/tutorials/reduce.ipynb: 5/9 code cells produced output` or more, `git diff --stat` shows a change, and the checkout restores it. If `metadata.kernelspec` appears in the diff, the restore is broken — fix `execute_tutorial.py`, do not proceed.

- [ ] **Step 2: Rewrite the example**

Replace `examples/animate_market_forecast.py` entirely:

```python
# -*- coding: utf-8 -*-
"""
==========================================================================
A market, sector by sector: one hierarchy, one hue, one forecast per line
==========================================================================

Twenty-four large-cap stocks, grouped into six sectors, plotted as a single
hierarchical DataFrame. The columns carry a ``(Market, Sector, Ticker)``
``MultiIndex``, and that is the whole layout instruction: ``hyp.plot``
expands the innermost level into features and every level above it into the
drawn hierarchy, so each **sector** becomes one trajectory and the whole
**market** becomes a second-level mean trajectory drawn heavier on top --
the classic bold-means / faint-leaves picture, with no index bookkeeping in
this file at all.

Each line is coloured continuously by its own **price index** (a `hue` that
mirrors the hierarchy: one value sequence per sector, with the market mean
taking the element-wise mean of its sectors), and each line carries **its
own next-day forecast** (``predict='Kalman', t=1``), redrawn every frame
from the history revealed so far and left behind as a fading fan
(``forecast_trail=16``). The camera makes one slow quarter-turn
(``rotations=0.25``) over the clip while ``chemtrails=True`` keeps the
traversed path glowing faintly behind each head.

**Does the forecast actually work?** The panel on the right says so, per
sector and overall, and it is computed HERE rather than in hypertools:
forecast scoring is a research decision, not a plotting one. Each series is
walked forward over its last 30 trading days, refitting on a trailing
60-day window, and a "hit" is a next-day step predicted with the right
sign. Measured cost of that loop: 210 Kalman fits, ~7 s. 50% is a coin
flip; a single linear-Gaussian filter on a near-random-walk price series
should not be expected to beat it by much, and reporting whatever it
actually scores is the point.

**Data & graceful degradation.** Ten years of daily closes are pulled from
Yahoo Finance's chart endpoint and cached on disk (verified 2026-07-28:
24/24 tickers, 2513 trading days, 2016-07-28 to 2026-07-28). If the network
is unavailable the example falls back to a synthetic basket with the same
sector structure, so it always renders -- the technique (hierarchy -> hue ->
chemtrails -> per-trace forecast) is identical either way.
"""

# Code source: Contextual Dynamics Laboratory
# License: MIT

import json
import os
import tempfile
import urllib.request

import numpy as np
import pandas as pd

import hypertools as hyp

CACHE = os.path.join(tempfile.gettempdir(), 'hypertools_gallery_cache')
os.makedirs(CACHE, exist_ok=True)

MARKET = 'Market'
RANGE = '10y'
# six sectors x FOUR tickers each: equal per-group widths, which the
# analysis pipeline requires (hypertools/plot/plot.py:3152-3153)
SECTORS = {
    'Technology': ['AAPL', 'MSFT', 'ORCL', 'IBM'],
    'Financials': ['JPM', 'BAC', 'GS', 'AXP'],
    'Healthcare': ['JNJ', 'PFE', 'MRK', 'ABT'],
    'Energy': ['XOM', 'CVX', 'COP', 'SLB'],
    'Consumer': ['KO', 'PG', 'WMT', 'MCD'],
    'Industrials': ['BA', 'CAT', 'GE', 'HON'],
}
COLUMN_NAMES = ['Market', 'Sector', 'Ticker']


def fetch_prices():
    """Daily closes with a ``(Market, Sector, Ticker)`` column MultiIndex,
    or ``None`` if anything (network, parsing) goes wrong."""
    try:
        series = {}
        for sector, tickers in SECTORS.items():
            for ticker in tickers:
                dest = os.path.join(CACHE, f'yahoo_{ticker}_{RANGE}.json')
                if not (os.path.exists(dest) and os.path.getsize(dest) > 0):
                    url = ('https://query1.finance.yahoo.com/v8/finance/chart/'
                           f'{ticker}?range={RANGE}&interval=1d')
                    req = urllib.request.Request(
                        url, headers={'User-Agent': 'hypertools-gallery/1.1'})
                    with urllib.request.urlopen(req, timeout=30) as response:
                        payload = response.read()
                    with open(dest, 'wb') as handle:
                        handle.write(payload)
                result = json.load(open(dest))['chart']['result'][0]
                series[(MARKET, sector, ticker)] = pd.Series(
                    result['indicators']['quote'][0]['close'],
                    index=pd.to_datetime(result['timestamp'], unit='s'))
        return _framed(series)
    except Exception:
        return None


def synthetic_prices(n_days=2513, seed=0):
    """Fallback: a market factor + per-sector factors + idiosyncratic noise,
    laid out with exactly the same column hierarchy."""
    rng = np.random.default_rng(seed)
    index = pd.bdate_range(end=pd.Timestamp('today').normalize(),
                           periods=n_days)
    market = np.cumsum(rng.standard_normal(n_days)) * 0.4
    series = {}
    for s, (sector, tickers) in enumerate(SECTORS.items()):
        factor = np.cumsum(rng.standard_normal(n_days)) * (0.3 + 0.05 * s)
        for k, ticker in enumerate(tickers):
            idio = np.cumsum(rng.standard_normal(n_days)) * (0.2 + 0.05 * k)
            series[(MARKET, sector, ticker)] = pd.Series(
                40.0 * np.exp(0.02 * (market + factor + idio) / 10),
                index=index)
    return _framed(series)


def _framed(series):
    prices = pd.DataFrame(series).ffill().dropna()
    prices.columns = pd.MultiIndex.from_tuples(prices.columns,
                                               names=COLUMN_NAMES)
    return prices


prices = fetch_prices()
source = 'Yahoo Finance daily closes'
if prices is None:
    prices, source = synthetic_prices(), 'synthetic basket (offline fallback)'
print(f'market data: {prices.shape[0]} days x {prices.shape[1]} tickers '
      f'in {len(SECTORS)} sectors ({source})')

# hue mirrors the hierarchy: ONE value sequence per sector leaf, each as long
# as the frame. The market-mean trace takes the element-wise mean of its
# sectors automatically, so nothing here computes it.
sector_index = [(prices[MARKET][sector].mean(axis=1)
                 / prices[MARKET][sector].mean(axis=1).iloc[0] * 100.0
                 ).to_numpy() for sector in SECTORS]

# THE hypertools call. The DataFrame's column MultiIndex IS the layout: six
# sector traces plus a heavier market-mean trace, each coloured by its own
# price index and each carrying its own next-day Kalman forecast, redrawn
# per frame and trailed. Widths/opacities come from the hierarchy, so no
# linewidth= is passed (it would be warned and ignored, plot.py:3037-3043).
duration, fps = 8, 20
fig, ani = hyp.plot(
    prices, '-',
    hue=sector_index, palette='plasma',
    colorbar={'label': 'sector price index (start = 100)'},
    manip={'model': 'Smooth', 'kwargs': {'kernel': 'boxcar',
                                         'kernel_width': 11}},
    normalize='across', reduce='IncrementalPCA', ndims=3,
    predict='Kalman', t=1, forecast_trail=16,
    animate=True, chemtrails=True, rotations=0.25,
    title='many markets as one path',
    duration=duration, frame_rate=fps, size=(11, 6.5), show=False)

# --- does the forecast work? scored HERE, not in the library -----------------
# Walk each series forward over its last N_SCORED days, refitting on a
# trailing WINDOW-day history, and count next-day steps predicted with the
# right sign. Measured: 7 series x 30 anchors = 210 Kalman fits, ~7 s.
WINDOW, N_SCORED = 60, 30


def directional_accuracy(y):
    hits = 0
    for a in range(len(y) - N_SCORED, len(y)):
        history = y[a - WINDOW:a].reshape(-1, 1)
        step = (float(np.asarray(hyp.predict(history, model='Kalman', t=1))[0, 0])
                - float(history[-1, 0]))
        hits += int(step * (y[a] - y[a - 1]) > 0)
    return 100.0 * hits / N_SCORED


market_curve = np.mean(sector_index, axis=0)
scores = {sector: directional_accuracy(curve)
          for sector, curve in zip(SECTORS, sector_index)}
scores[MARKET] = directional_accuracy(market_curve)
print('next-day direction correct: '
      + ', '.join(f'{name} {pct:.0f}%' for name, pct in scores.items()))

# --- which tickers make up which sector, and how each one scored -------------
ax = [a for a in fig.axes if hasattr(a, 'zaxis')][0]
ax.set_position([0.0, 0.03, 0.62, 0.9])
for row, (sector, tickers) in enumerate(SECTORS.items()):
    y = 0.88 - row * 0.135
    fig.text(0.66, y, f'{sector}   {scores[sector]:.0f}%', ha='left',
             va='top', fontsize=11.5, fontweight='bold', color='#1a1a1a')
    fig.text(0.66, y - 0.042, '  '.join(tickers), ha='left', va='top',
             fontsize=9.5, color='#666')
fig.text(0.66, 0.055, f'whole market   {scores[MARKET]:.0f}%', ha='left',
         va='top', fontsize=11.5, fontweight='bold', color='#1a1a1a')
fig.text(0.66, 0.015, 'next-day direction, last 30 sessions (50% = coin flip)',
         ha='left', va='top', fontsize=8.5, color='#8a8a8a', style='italic')
```

- [ ] **Step 3: Run the example and confirm it renders**

Run: `MPLBACKEND=Agg .venv/bin/python examples/animate_market_forecast.py`

Expected: exits 0, no traceback, no `UserWarning`, and prints two lines of the form

```
market data: 2513 days x 24 tickers in 6 sectors (Yahoo Finance daily closes)
next-day direction correct: Technology NN%, Financials NN%, ..., Market NN%
```

Report whatever accuracy comes out; do **not** tune the example until a number looks good. Wall clock should be roughly the current 6.2 s plus the measured ~7 s of scoring.

- [ ] **Step 4: Confirm the hierarchy actually drew what the docstring claims**

Run:

```bash
MPLBACKEND=Agg .venv/bin/python - <<'PY'
import runpy
ns = runpy.run_path('examples/animate_market_forecast.py')
fig, ani = ns['fig'], ns['ani']
ax = [a for a in fig.axes if hasattr(a, 'zaxis')][0]
ani._func(40, *ani._args)
widths = sorted({round(float(l.get_linewidth()), 2) for l in ax.lines})
print('axes:', len(fig.axes), '| line artists:', len(ax.lines),
      '| distinct linewidths:', widths)
print('title:', repr(ax.get_title()))
PY
```

Expected: `axes: 2` (the 3-D box plus the colorbar), **7 drawn traces** (6 sectors + 1 market mean) plus their forecast artists, **at least two distinct linewidths** (the MultiIndex contract is `linewidth = 1 + (L - 1 - level_idx)`, so the market mean is wider than a sector), and `title: 'many markets as one path'`. This snippet uses `ani._func` **only as a test probe**, exactly as the sibling plans' tests do; it never enters the example.

- [ ] **Step 5: Rewrite the notebook in lockstep**

Rewrite `docs/tutorials/market_forecast.ipynb` so its code cells are the script's code, split at the script's own section boundaries, keeping cell 0 (the Colab install cell) untouched:

| cell | type | content |
|-|-|-|
| 0 | code | the existing Colab install cell — **unchanged** |
| 1 | markdown | title + the script's docstring, as prose |
| 2 | markdown | `## 1. Imports and a disk cache` |
| 3 | code | imports, `CACHE`, `MARKET`, `RANGE`, `SECTORS`, `COLUMN_NAMES` |
| 4 | markdown | `## 2. Fetch 24 tickers into a (Market, Sector, Ticker) frame` |
| 5 | code | `fetch_prices`, `synthetic_prices`, `_framed`, the dispatch and `print` |
| 6 | markdown | `## 3. The hierarchy IS the layout` — explain that the column MultiIndex replaces every hand-built group list, and that hue mirrors the hierarchy |
| 7 | code | the `sector_index` comprehension |
| 8 | markdown | `## 4. One call: hierarchy, hue, chemtrails, and a forecast per trace` |
| 9 | code | the `hyp.plot(...)` call |
| 10 | markdown | `## 5. Scoring the forecast — deliberately NOT a library job` |
| 11 | code | `WINDOW`, `N_SCORED`, `directional_accuracy`, `scores`, the `print` |
| 12 | markdown | `## 6. Which tickers make up which sector` |
| 13 | code | the side-panel block |
| 14 | markdown | `## 7. Display the animation` |
| 15 | code | `from IPython.display import HTML` / `HTML(ani.to_jshtml())` |

- [ ] **Step 6: Execute the notebook and check the code stayed in lockstep**

```bash
.venv/bin/python scripts/execute_tutorial.py docs/tutorials/market_forecast.ipynb
.venv/bin/python scripts/measure_native_ratio.py \
    examples/animate_market_forecast.py docs/tutorials/market_forecast.ipynb
```

(`scripts/measure_native_ratio.py` is created in Task 8 Step 1; if you are working tasks in order, do that step first — it is standalone.)

Expected: both files inside budget (**≤ 115 / ≤ 120 code lines**). If either is missed, cut presentation code. **Do not predict an output-cell count here** — v2 guessed `7/8` and every one of its five per-task guesses was wrong. Record the measured visible-output INDEX SET from this run into `EXPECTED_VISIBLE_OUTPUTS` (Task 8), which names the offending cell when it drifts.

- [ ] **Step 7: Commit**

```bash
git add examples/animate_market_forecast.py docs/tutorials/market_forecast.ipynb \
        scripts/execute_tutorial.py
git commit -m "docs(gallery): market example is a column-MultiIndex showcase with native per-trace forecasts"
```

---

## Task 3: Weather — the paper figure, nearly all native

> **v3: THE SCRIPT HALF IS PARTIALLY LANDED.** `d730a085` migrated this file off `ani._func` onto `anim.on_frame(...)`, and changed `fig, ani =` to `anim` + `anim.figure`. Read the file on disk first (`git show d730a085~1:examples/animate_weather_decades.py` for the pre-migration state) and reconcile rather than overwrite.

**BEFORE — re-measured 2026-08-02 at `065c841e`, with the docstring-aware metric:**

| file | v2 said | actual now |
|-|-|-|
| `examples/animate_weather_decades.py` | 333 raw, 196 code, 11 native (5.6%) | **336 raw, 195 code, 11 native (5.6%)** |
| `docs/tutorials/weather_decades.ipynb` | 206 code, 10 native (4.9%), "0 of 7 executed" | **194 code, 11 native (5.7%), 2 of 7 cells carry output** |

Audit classification (unchanged): A=72 **B=8 C=44** D=70 NATIVE=19.

**Note for the defect-marker gate:** this file's docstring now contains the string `` `ani._func` `` while *explaining the migration away from it*. That is documentation, not a private reach — which is why Task 8's scan strips docstrings before matching (Contract 3).

**AFTER (contracted budget):** script **≤ 77 code lines** (62 for the rewrite + **15 measured** for the Step 0b split); notebook **≤ 82** (= 77 + 5); **zero** defect markers. The rewrite alone measures 56, and 56 + 15 = 71 ≤ 77. **v3 briefly had 62 here with the split mandated on top, i.e. 71 against 62 — unsatisfiable, the exact class this plan claims to have made impossible.**

The notebook budget is DERIVED, not written down: `script_budget + NOTEBOOK_OVERHEAD` (Contract 6b), so it can never again be set below its own script's. No ratio floor — ratio is reported, never gated (Contract 6a).

**The reframe.** The current example treats 6 cities as 6 *datasets* and hand-builds a hemisphere hierarchy on top (26 lines of **C**), plus a whole second daily-temperature panel (70 lines of **D**), plus a `Line3DCollection` linewidth workaround (**C**, and a library bug that animation-core Task 2 fixes). The paper figure is a different, simpler object: **20 cities are 20 FEATURES of one trajectory through time**, coloured by the average temperature across them. That is one `hyp.plot` call, and it needs no hierarchy at all.

Verified today, end to end: `hyp.plot(temps, fmt='-', hue=avg_temp, palette='RdBu_r', normalize='across', manip='Smooth', animate=True, chemtrails=True, colorbar=True, duration=8, frame_rate=20, show=False)` on the real `(1645, 20)` matrix runs in **0.3 s**, emits **no warnings**, produces **2 axes** (3-D box + colorbar), and at frame 150 the head/trail collections carry **879 distinct colours**.

**Files:** rewrite `examples/animate_weather_decades.py`; rewrite `docs/tutorials/weather_decades.ipynb`.

- [ ] **Step 0: Split the loader from the figure builder (Contract 4 / Step 0b)**

**Do this FIRST, before the rewrite below.** Task 8 Step 0b defines the contract and works the whole pattern through on weather; this step applies it to `examples/animate_weather_decades.py`. Without it `test_examples_produce_their_stated_artifact` fails on this example, and importing it fetches.

Produce exactly these three names in `examples/animate_weather_decades.py`:

| name | signature | notes |
|-|-|-|
| payload | `class Weather(NamedTuple)` with fields `monthly, daily, hemispheres, source` | self-documenting; `source` records which path was used |
| loader | `load_weather(cities=CITIES) -> Weather` | the ONLY code here that may touch the network (fetch_city_months, fetch_city_daily_temp) |
| fixture | `fixture_data() -> Weather` | its own seeded `synthetic_city_months` / `synthetic_city_daily` — no network, no committed bytes unless stated |
| builder | `construct_artifact(data) -> HyperAnimation` | everything else, reading `data.<field>` instead of module globals. **Returns the wrapper, never the unpacked pair** (Contract 8) |

Then move every loader CALL behind a `__main__` guard, and make each fetcher honour `HYPERTOOLS_OFFLINE` by raising:

```python
if __name__ == '__main__':
    data = load_weather()
    anim = construct_artifact(data)
    fig = anim.figure
```

Verify before moving on:

```bash
MPLBACKEND=Agg .venv/bin/python -c "
import importlib.util, os
os.environ['HYPERTOOLS_OFFLINE'] = '1'
spec = importlib.util.spec_from_file_location('m', 'examples/animate_weather_decades.py')
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
print('imported with no fetch; has', [n for n in ('construct_artifact', 'fixture_data') if hasattr(m, n)])
anim = m.construct_artifact(m.fixture_data()); print('frames:', anim.n_frames)"
```

Expected: it imports without touching the network and prints both names plus a frame count. If the import fetches, a loader call is still at module scope.

- [ ] **Step 1: Rewrite the example**

Replace `examples/animate_weather_decades.py` entirely:

```python
# -*- coding: utf-8 -*-
"""
=======================================================================
A century of weather: twenty cities as twenty features, one hot path
=======================================================================

The figure from the HyperTools paper, in one library call. Monthly mean
temperatures for **twenty cities spread across both hemispheres** (Bangkok
to Montreal, Sydney to Moscow) are treated not as twenty separate series
but as **twenty features of one measurement**: each month is a single
20-dimensional observation of "what the world's weather was doing", and
``hyp.plot`` reduces that stream to a 3-D path.

Every point on the path is coloured by the **average temperature across all
twenty cities** on a diverging blue-cold / red-hot scale
(``palette='RdBu_r'``), so the seasons show up as the path sweeping between
the ends of the colormap and the slow warming trend shows up as where the
sweep sits. ``manip='Smooth'`` takes out month-to-month jitter before
anything is drawn, ``normalize='across'`` z-scores the twenty city columns
over the stacked rows so a hot city cannot dominate the reduction purely by
scale, and ``chemtrails=True`` leaves the traversed path glowing faintly
behind the moving head as 138 years play.

There is no hand-built hierarchy, no hand-spliced colormap, no
``ScalarMappable``, and no per-frame callback: the colour axis, the
colorbar and the trail are all the library's.

**Data & graceful degradation.** The temperature matrix and the city
coordinates are the ones published with the HyperTools paper (verified
2026-07-28: 1645 complete months, 1875-2013, 20 cities), fetched once and
cached. If the network is unavailable the example synthesizes twenty
seasonal series in opposite hemispheric phase with a slow warming drift, so
it always renders.
"""

# Code source: Contextual Dynamics Laboratory
# License: MIT

import io
import os
import tempfile
import urllib.request

import numpy as np
import pandas as pd

import hypertools as hyp

CACHE = os.path.join(tempfile.gettempdir(), 'hypertools_gallery_cache')
os.makedirs(CACHE, exist_ok=True)
BASE = ('https://raw.githubusercontent.com/ContextLab/'
        'hypertools-paper-notebooks/master/data/')


def fetch_temperatures():
    """(months x 20 cities) monthly means and the city names, or ``None``."""
    try:
        frames = {}
        for name in ('temperatures.csv', 'temperature_locs.csv'):
            dest = os.path.join(CACHE, name)
            if not (os.path.exists(dest) and os.path.getsize(dest) > 0):
                req = urllib.request.Request(
                    BASE + name, headers={'User-Agent': 'hypertools-gallery/1.1'})
                with urllib.request.urlopen(req, timeout=60) as response:
                    payload = response.read()
                with open(dest, 'wb') as handle:
                    handle.write(payload)
            frames[name] = pd.read_csv(io.BytesIO(open(dest, 'rb').read()))
        # the CSV carries both '<City>' (absolute) and '<City>_anomaly'
        # columns; the locations file fixes the city order
        cities = list(frames['temperature_locs.csv']['City'])
        complete = frames['temperatures.csv'].dropna()
        return complete[cities].to_numpy(float), cities
    except Exception:
        return None


def synthetic_temperatures(n_months=1645, n_cities=20, seed=0):
    """Fallback: seasonal cycles in opposite hemispheric phase, drifting."""
    rng = np.random.default_rng(seed)
    t = np.arange(n_months)
    columns = []
    for city in range(n_cities):
        phase = 0.0 if city % 2 == 0 else np.pi          # opposite seasons
        columns.append(14 + 11 * np.sin(2 * np.pi * t / 12 + phase)
                       + 3 * (t / n_months)
                       + rng.standard_normal(n_months) * 0.6)
    return (np.column_stack(columns),
            [f'city {i + 1}' for i in range(n_cities)])


fetched = fetch_temperatures()
source = 'HyperTools paper temperature archive'
if fetched is None:
    fetched, source = synthetic_temperatures(), 'synthetic (offline fallback)'
temps, cities = fetched
print(f'weather: {temps.shape[0]} months x {temps.shape[1]} cities ({source})')

# THE hypertools call: twenty cities as twenty FEATURES of one path, coloured
# by the average temperature across them on a blue-cold / red-hot scale.
duration, fps = 8, 20
fig, ani = hyp.plot(
    temps, '-',
    hue=temps.mean(axis=1), palette='RdBu_r',
    colorbar={'label': 'average temperature across '
                       f'{len(cities)} cities (°C)'},
    manip='Smooth', normalize='across',
    animate=True, chemtrails=True,
    title=f'{len(cities)} cities, 1875–2013, as one moving path',
    duration=duration, frame_rate=fps, size=(8, 7), show=False)
```

- [ ] **Step 2: Run the example and confirm it renders**

Run: `MPLBACKEND=Agg .venv/bin/python examples/animate_weather_decades.py`

Expected: exits 0, no warnings, prints

```
weather: 1645 months x 20 cities (HyperTools paper temperature archive)
```

- [ ] **Step 3: Confirm the colour sweep and colorbar are real**

Run:

```bash
MPLBACKEND=Agg .venv/bin/python - <<'PY'
import numpy as np, runpy
ns = runpy.run_path('examples/animate_weather_decades.py')
fig, ani = ns['fig'], ns['ani']
ani._func(150, *ani._args)
ax = [a for a in fig.axes if hasattr(a, 'zaxis')][0]
cols = np.vstack([c.get_colors() for c in ax.collections
                  if c.get_label() == '_nolegend_'])
print('axes:', len(fig.axes), '| distinct colours:', len(np.unique(cols.round(4), axis=0)))
print('title:', repr(ax.get_title()))
PY
```

Expected: `axes: 2`, **several hundred distinct colours** (879 measured for these exact parameters), and the title string. If `axes` is 1, the colorbar did not render and `colorbar=` is wrong.

- [ ] **Step 4: Rewrite the notebook in lockstep**

Rewrite `docs/tutorials/weather_decades.ipynb`, keeping cell 0 (Colab install) unchanged:

| cell | type | content |
|-|-|-|
| 0 | code | existing Colab install cell — unchanged |
| 1 | markdown | title + the docstring as prose, including *why* cities are features and not datasets |
| 2 | markdown | `## 1. Imports and a disk cache` |
| 3 | code | imports, `CACHE`, `BASE` |
| 4 | markdown | `## 2. Fetch the paper's temperature matrix (with a synthetic fallback)` |
| 5 | code | `fetch_temperatures`, `synthetic_temperatures`, the dispatch and `print` |
| 6 | markdown | `## 3. One call` — spell out each kwarg's stage: `manip` → `normalize` → `reduce` → animate |
| 7 | code | the `hyp.plot(...)` call |
| 8 | markdown | `## 4. Display the animation` |
| 9 | code | `HTML(ani.to_jshtml())` |

- [ ] **Step 5: Execute and measure**

```bash
.venv/bin/python scripts/execute_tutorial.py docs/tutorials/weather_decades.ipynb
.venv/bin/python scripts/measure_native_ratio.py \
    examples/animate_weather_decades.py docs/tutorials/weather_decades.ipynb
```

Expected: both files inside budget (**≤ 62 / ≤ 67 code lines**). Record the measured visible-output index set into `EXPECTED_VISIBLE_OUTPUTS` rather than asserting a predicted count — v2's `4/5` named cells 3/5/7/9, and the real emitting cells are a different set entirely.

- [ ] **Step 6: Commit**

```bash
git add examples/animate_weather_decades.py docs/tutorials/weather_decades.ipynb
git commit -m "docs(gallery): weather example is the paper figure in one native call"
```

---

## Task 4: Paintings — native text, native palette, full descriptions

**BEFORE — re-measured 2026-08-02 at `065c841e`. This is the ONE example `d730a085` did not touch** (its last change is `4d1d2223`), so v2's baseline for the script is still exact and this task is a clean rewrite rather than a rebase:

| file | v2 said | actual now |
|-|-|-|
| `examples/animate_painting_embeddings.py` | 213 raw, 146 code, 11 native (7.5%) | **212 raw, 146 code, 11 native (7.5%)** — matches |
| `docs/tutorials/painting_embeddings.ipynb` | 116 code, 10 native (8.6%), "0 of 6 executed" | **121 code, 11 native (9.1%), 2 of 6 cells carry output** |

Audit classification (unchanged): A=97 **B=25 C=6** D=13 NATIVE=8.

**AFTER (contracted budget):** script **≤ 133 code lines** (118 + 15 placeholder); notebook **≤ 138** (= 133 + 5); **zero** defect markers. The rewrite alone measures 111. Paintings has TWO fetch sites, so its real split overhead is likely above weather's 15 — measure it in Step 0c. (The budget is generous because the `PAINTINGS` dict alone is ~54 lines of genuine class-**A** data.) **v2 set the notebook at 110 — BELOW the script's 118 — which no correct notebook could satisfy.**

The notebook budget is DERIVED, not written down: `script_budget + NOTEBOOK_OVERHEAD` (Contract 6b), so it can never again be set below its own script's. No ratio floor — ratio is reported, never gated (Contract 6a).

**What goes, and to what:**

| deleted | replaced by |
|-|-|
| `embed()` — the hand-rolled `SentenceTransformer`/TF-IDF helper (`:101-111`, class **B**) | `vectorizer='all-MiniLM-L6-v2', semantic=None, corpus=None` on the plot call (`text2mat.py:89`, dispatch at `:184`, `semantic` at `:391`, `corpus` at `:404`) |
| `all_windows`/`owners` bookkeeping + `clouds = [red[owners == name] ...]` (`:148-160`, `:181-182`, class **B**) | a **list of lists of strings** straight into `hyp.plot` — verified `hyp.reduce([[s,s,s],[s,s,s],[s,s]], ndims=3)` → `[(3,3),(3,3),(2,3)]`, so `format_data` does the splitting |
| the k-means + `np.argmax(counts)` + luminance-clamp block inside `canvas_color` (`:136-144`, class **B** once Task 1 exists) | `image_palette(path)[0]` — and the luminance clamp is no longer needed, because the salience ordering returns a vivid colour rather than the muted background it was compensating for |
| `fig.text(...)` title (`:198-201`, class **B**) | `title=` |
| the 85th-percentile outlier trim (`:172-179`, class **C**) | **dropped** — see the *Decisions still needed* entry named **"The paintings example's outlier trim"** |
| `blurb` in the side panel | the full `text` each painting already carries, which is what was embedded — the panel now shows exactly what produced the geometry |

The download-and-cache half of `canvas_color` **stays** (class **A**, textbook: hypertools does not fetch images, Contract 4).

**Files:** rewrite `examples/animate_painting_embeddings.py`; rewrite `docs/tutorials/painting_embeddings.ipynb`.

- [ ] **Step 0: Split the loader from the figure builder (Contract 4 / Step 0b)**

**Do this FIRST, before the rewrite below.** Task 8 Step 0b defines the contract and works the whole pattern through on weather; this step applies it to `examples/animate_painting_embeddings.py`. Without it `test_examples_produce_their_stated_artifact` fails on this example, and importing it fetches.

Produce exactly these three names in `examples/animate_painting_embeddings.py`:

| name | signature | notes |
|-|-|-|
| payload | `class Paintings(NamedTuple)` with fields `vectors, owners, colors, source` | self-documenting; `source` records which path was used |
| loader | `load_paintings(PAINTINGS) -> Paintings` | the ONLY code here that may touch the network (canvas_color, the SentenceTransformer load) |
| fixture | `fixture_data() -> Paintings` | the one committed 1.7 KB 64-px thumbnail — no network, no committed bytes unless stated |
| builder | `construct_artifact(data) -> HyperAnimation` | everything else, reading `data.<field>` instead of module globals. **Returns the wrapper, never the unpacked pair** (Contract 8) |

Then move every loader CALL behind a `__main__` guard, and make each fetcher honour `HYPERTOOLS_OFFLINE` by raising:

```python
if __name__ == '__main__':
    data = load_paintings()
    anim = construct_artifact(data)
    fig = anim.figure
```

Verify before moving on:

```bash
MPLBACKEND=Agg .venv/bin/python -c "
import importlib.util, os
os.environ['HYPERTOOLS_OFFLINE'] = '1'
spec = importlib.util.spec_from_file_location('m', 'examples/animate_painting_embeddings.py')
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
print('imported with no fetch; has', [n for n in ('construct_artifact', 'fixture_data') if hasattr(m, n)])
anim = m.construct_artifact(m.fixture_data()); print('frames:', anim.n_frames)"
```

Expected: it imports without touching the network and prints both names plus a frame count. If the import fetches, a loader call is still at module scope.

- [ ] **Step 1: Rewrite the example**

Keep the `PAINTINGS` dict verbatim (lines 43-96 of the current file) and replace everything else. The new body:

```python
# -*- coding: utf-8 -*-
"""
=============================================================
Five paintings, described in words, drawn in their own colors
=============================================================

Text becomes geometry, tinted by the art itself. A full paragraph
describing each of five famous paintings is cut into overlapping word
windows and handed to ``hyp.plot`` **as text** -- a list of five lists of
strings. One call embeds every window with a sentence-transformer
(``vectorizer='all-MiniLM-L6-v2'``), reduces all of them together into one
shared 3-D space with UMAP, keeps the five clouds separate (the nesting of
the input is the grouping), spins the camera, and annotates each cloud with
its painting's name.

Each cloud is drawn in a colour taken from the **actual canvas**:
``hypertools.plot.colors.image_palette`` clusters the downloaded image's
pixels and orders the result by ``pixel_fraction * chroma``, so the vivid
subject wins rather than the muted background -- Starry Night comes out
cobalt, not canvas-beige. The side panels show the complete description
that was embedded, in that painting's colour, so nothing about the geometry
is hidden.

**Data & graceful degradation.** The descriptions are bundled inline (so the
text side is fully offline and deterministic). Each canvas is downloaded
once from Wikimedia Commons and cached; if an image cannot be fetched, a
hand-picked representative colour is used instead. Text embedding needs the
``[text]`` extra (``pip install "hypertools[text]"``); without it,
``vectorizer='TfidfVectorizer'`` is used, and the pipeline (embed -> reduce
together -> one cloud/colour per painting -> spin) is identical either way.
"""

# Code source: Contextual Dynamics Laboratory
# License: MIT

import os
import tempfile
import textwrap
import urllib.request

from matplotlib.colors import to_rgb

import hypertools as hyp
from hypertools.plot.colors import image_palette

CACHE = os.path.join(tempfile.gettempdir(), 'hypertools_gallery_cache')
os.makedirs(CACHE, exist_ok=True)
FILEPATH = 'https://commons.wikimedia.org/wiki/Special:FilePath/'

PAINTINGS = {
    ...  # UNCHANGED from the current file, lines 43-96
}

WINDOW, STEP = 10, 1


def windows(text, size=WINDOW, step=STEP):
    """Overlapping word windows: one observation per window."""
    words = text.split()
    return [' '.join(words[i:i + size])
            for i in range(0, max(1, len(words) - size + 1), step)]


def canvas_color(spec):
    """The painting's most salient colour, from the real canvas.

    The download and the cache are this example's job (hypertools never
    fetches an image); choosing the colour is the library's:
    ``image_palette`` orders clusters by ``pixel_fraction * chroma``, so a
    small vivid region beats a large muted one. Ordering by cluster SIZE --
    which is what this example used to do -- returns the background.
    """
    try:
        dest = os.path.join(CACHE, 'paint_' + spec['file'][:20] + '.jpg')
        if not (os.path.exists(dest) and os.path.getsize(dest) > 0):
            req = urllib.request.Request(
                FILEPATH + spec['file'] + '?width=400',
                headers={'User-Agent': 'hypertools-gallery/1.1'})
            with urllib.request.urlopen(req, timeout=30) as response:
                payload = response.read()
            with open(dest, 'wb') as handle:
                handle.write(payload)
        return tuple(image_palette(dest)[0])
    except Exception:
        return to_rgb(spec['fallback'])


names = list(PAINTINGS)
descriptions = [windows(PAINTINGS[name]['text']) for name in names]
colors = [canvas_color(PAINTINGS[name]) for name in names]
# labels are per-OBSERVATION (plot.py:1154-1159): a nested list with one
# sub-list per cloud, carrying the painting's name on its MIDDLE window
# (roughly the centre of a text trajectory) and None everywhere else.
labels = [[name if i == len(cloud) // 2 else None
           for i in range(len(cloud))]
          for name, cloud in zip(names, descriptions)]
print(f'paintings: {len(names)}, '
      f'{sum(len(c) for c in descriptions)} description windows')

# THE hypertools call: raw TEXT in, five clouds out. The nesting of the
# input is the grouping, the vectorizer/semantic/corpus trio selects a
# sentence-transformer instead of the default bag-of-words + LDA, and
# reduce= puts every window into one shared UMAP space so the clouds are
# directly comparable. n_neighbors=12 keeps one description's windows
# together, min_dist=0.25 lets a clump pack closely, random_state=42 fixes
# the stochastic layout.
duration, fps = 12, 20
fig, ani = hyp.plot(
    descriptions, '.',
    vectorizer='all-MiniLM-L6-v2', semantic=None, corpus=None,
    reduce={'model': 'UMAP', 'kwargs': {'n_neighbors': 12, 'min_dist': 0.25,
                                        'random_state': 42}},
    ndims=3, color=colors, markersize=5, labels=labels,
    animate='spin', rotations=2,
    title='five paintings, described in words, drawn in their own colors',
    duration=duration, frame_rate=fps, size=(13, 9), show=False)

# the descriptions that were actually embedded, each in its cloud's colour
ax = fig.axes[0]
ax.set_position([0.0, 0.0, 0.52, 1.0])
for i, name in enumerate(names):
    y = 0.94 - i * 0.19
    color = colors[i]
    fig.text(0.55, y, name, ha='left', va='top', fontsize=12,
             fontweight='bold', color=color)
    body = '\n'.join(textwrap.wrap(PAINTINGS[name]['text'], 62))
    fig.text(0.55, y - 0.028, body, ha='left', va='top', fontsize=7,
             color=color)
```

- [ ] **Step 2: Run the example and confirm it renders**

Run: `MPLBACKEND=Agg .venv/bin/python examples/animate_painting_embeddings.py`

Expected: exits 0, no traceback, prints `paintings: 5, NNN description windows`. If the `[text]` extra is not installed the sentence-transformer resolution fails; per Contract 4 the example must still render, so confirm the fallback:

```bash
MPLBACKEND=Agg .venv/bin/python -c "
import hypertools as hyp
from hypertools.tools.text2mat import text2mat
print(text2mat(['a red apple','a green pear'], vectorizer='all-MiniLM-L6-v2',
               semantic=None, corpus=None).shape)"
```
Expected: `(2, 384)` with the extra installed. Without it, the call raises — in that case change the example's `vectorizer=` to be chosen once at the top:

```python
try:
    import sentence_transformers  # noqa: F401
    VECTORIZER = 'all-MiniLM-L6-v2'
except ImportError:
    VECTORIZER = 'TfidfVectorizer'
```
and pass `vectorizer=VECTORIZER`. That is 4 lines of graceful degradation using **only** documented kwargs — it is not a re-implementation, and it keeps the offline property Contract 4 requires.

- [ ] **Step 3: Confirm the colour is the vivid one, not the background**

Run:

```bash
MPLBACKEND=Agg .venv/bin/python - <<'PY'
import numpy as np, runpy
ns = runpy.run_path('examples/animate_painting_embeddings.py')
for name, c in zip(ns['names'], ns['colors']):
    rgb = np.asarray(c)
    print(f'{name:16s} rgb={np.round(rgb,3)}  chroma={rgb.max()-rgb.min():.3f}')
PY
```

Expected: five colours, each with **chroma > 0.10** for any painting whose image was fetched. A chroma near zero means the extraction returned a grey/beige — i.e. the salience ordering regressed, or the fallback hex was used because the download failed. Distinguish the two by checking `ls $TMPDIR/hypertools_gallery_cache/paint_*.jpg`.

- [ ] **Step 4: Rewrite the notebook in lockstep**

Rewrite `docs/tutorials/painting_embeddings.ipynb`, keeping cell 0 unchanged:

| cell | type | content |
|-|-|-|
| 0 | code | existing Colab install cell — unchanged |
| 1 | markdown | title + docstring prose |
| 2 | markdown | `## 1. Imports, a disk cache, and five descriptions` |
| 3 | code | imports, `CACHE`, `FILEPATH`, the `PAINTINGS` dict, `WINDOW`/`STEP` |
| 4 | markdown | `## 2. A colour from each real canvas` — explain the salience ordering and why largest-cluster is wrong |
| 5 | code | `windows`, `canvas_color`, `names`/`descriptions`/`colors`/`labels`, the `print` |
| 6 | markdown | `## 3. One call: raw text in, five clouds out` |
| 7 | code | the `hyp.plot(...)` call |
| 8 | markdown | `## 4. The descriptions that were embedded` |
| 9 | code | the side-panel block |
| 10 | markdown | `## 5. Display the animation` |
| 11 | code | `HTML(ani.to_jshtml())` |

- [ ] **Step 5: Execute and measure**

```bash
.venv/bin/python scripts/execute_tutorial.py docs/tutorials/painting_embeddings.ipynb
.venv/bin/python scripts/measure_native_ratio.py \
    examples/animate_painting_embeddings.py docs/tutorials/painting_embeddings.ipynb
```

Expected: both files inside budget (**≤ 118 / ≤ 123 code lines**). Record the measured visible-output index set into `EXPECTED_VISIBLE_OUTPUTS`; do not assert a predicted count.

- [ ] **Step 6: Commit**

```bash
git add examples/animate_painting_embeddings.py docs/tutorials/painting_embeddings.ipynb
git commit -m "docs(gallery): paintings example uses native text embedding and native image palettes"
```

---

## Task 5: Conversation — native text, serial order, per-segment titles

> **v3: THE SCRIPT HALF IS PARTIALLY LANDED, AND v2's PRESCRIBED TEXT WOULD REGRESS IT INTO A CRASH.** `d730a085` migrated this file onto `anim.on_frame(decorate)` and replaced `ani._args[0]`/`[1]` with `ctx.datasets`/`ctx.artists`. It binds `anim = hyp.plot(...)` **without unpacking**, which is what makes `.on_frame()` reachable (Contract 8). v2's prescribed notebook does `fig, ani = hyp.plot(...)` and then `ani.on_frame(recency_fade)` — an `AttributeError`, because `ani` is then the raw `FuncAnimation`. Reconcile against the file on disk; `git show d730a085~1:examples/animate_conversation.py` shows the pre-migration state.

**BEFORE — re-measured 2026-08-02 at `065c841e`, with the docstring-aware metric:**

| file | v2 said | actual now |
|-|-|-|
| `examples/animate_conversation.py` | 316 raw, 166 code, 9 native (5.4%) | **320 raw, 165 code, 9 native (5.5%)** |
| `docs/tutorials/conversation_shape.ipynb` | 186 code, 11 native (5.9%), "0 of 6 executed" | **176 code, 11 native (6.2%), 2 of 6 cells carry output** |

Audit classification (unchanged): A=61 **B=31 C=49** D=40 NATIVE=9.

**Note for the defect-marker gate:** as with weather, this file's docstring names `` `ani._func` `` while explaining its removal. Documentation, not a reach.

**AFTER (contracted budget):** script **≤ 105 code lines** (90 + 15 placeholder); notebook **≤ 110** (= 105 + 5); **zero** defect markers. The rewrite alone measures 88. (The `TURNS` list alone is 29 lines of class-**A** data.) **v2 said 72 here in prose while its enforced `BUDGETS` dict already said 90, and set the notebook at 76 — BELOW the script's — which no correct notebook could satisfy.**

The notebook budget is DERIVED, not written down: `script_budget + NOTEBOOK_OVERHEAD` (Contract 6b), so it can never again be set below its own script's. No ratio floor — ratio is reported, never gated (Contract 6a).

**What goes, and to what:**

| deleted | replaced by |
|-|-|
| `embed()` (`:88-100`, class **B**) | `vectorizer='all-MiniLM-L6-v2', semantic=None, corpus=None` |
| the manual re-split into per-turn arrays (`:144-151`, class **B**) | a **list of lists of strings** — the nesting is the grouping |
| `mpatches.Patch` + `fig.legend` (`:173-176`, class **B**) | a categorical `hue=` + `legend=True`. **Verified**: 6 line datasets with a nested categorical hue draw **6 lines** and a **3-entry** legend, and `animate='serial'` still hands 6 datasets to the backend — so per-turn identity survives |
| `fig.text(...)` title (`:177-178`, class **B**) | `title=` |
| `ani._args[0]`/`[1]`, `drawn_lens`, `starts`, `total_pts`, `shown_counts`, `current_state` (`:182-237`, class **C**, and a by-hand copy of ``_trail_kwargs` in matplotlib_backend.py:1667-1669`) | per-segment `title=` (animation-core Task 8), which is driven by the library's own schedule |
| the speaker text artist + `caption_lines` + `set_caption` (`:180-181`, `:240-283`, class **D**) | the per-segment title itself carries `Speaker  "the line"` |
| `_wrapped` + `ani._func = _wrapped` (`:286-316`, class **C**) | `on_frame=` (animation-core Task 7) for the recency fade only — the sole remaining per-frame effect |

The `word_spans` window helper collapses to a plain `windows()` (the span bookkeeping existed only to bold the current word in the deleted caption), and `min_wins` stays: it prevents a real rendering artefact (a one-row dataset draws as a dot), and the comment at `:110-117` documents it accurately.

**Files:** rewrite `examples/animate_conversation.py`; rewrite `docs/tutorials/conversation_shape.ipynb`.

- [ ] **Step 0: Split the loader from the figure builder (Contract 4 / Step 0b)**

**Do this FIRST, before the rewrite below.** Task 8 Step 0b defines the contract and works the whole pattern through on weather; this step applies it to `examples/animate_conversation.py`. Without it `test_examples_produce_their_stated_artifact` fails on this example, and importing it fetches.

Produce exactly these three names in `examples/animate_conversation.py`:

| name | signature | notes |
|-|-|-|
| payload | `class Conversation(NamedTuple)` with fields `vectors, speakers, spans, source` | self-documenting; `source` records which path was used |
| loader | `embed_turns(TURNS) -> Conversation` | the ONLY code here that may touch the network (the SentenceTransformer load) |
| fixture | `fixture_data() -> Conversation` | the TF-IDF branch, already a deterministic real sklearn fit — no network, no committed bytes unless stated |
| builder | `construct_artifact(data) -> HyperAnimation` | everything else, reading `data.<field>` instead of module globals. **Returns the wrapper, never the unpacked pair** (Contract 8) |

Then move every loader CALL behind a `__main__` guard, and make each fetcher honour `HYPERTOOLS_OFFLINE` by raising:

```python
if __name__ == '__main__':
    data = embed_turns()
    anim = construct_artifact(data)
    fig = anim.figure
```

Verify before moving on:

```bash
MPLBACKEND=Agg .venv/bin/python -c "
import importlib.util, os
os.environ['HYPERTOOLS_OFFLINE'] = '1'
spec = importlib.util.spec_from_file_location('m', 'examples/animate_conversation.py')
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
print('imported with no fetch; has', [n for n in ('construct_artifact', 'fixture_data') if hasattr(m, n)])
anim = m.construct_artifact(m.fixture_data()); print('frames:', anim.n_frames)"
```

Expected: it imports without touching the network and prints both names plus a frame count. If the import fetches, a loader call is still at module scope.

- [ ] **Step 1: Rewrite the example**

Keep `SPEAKER_COLOR` and the `TURNS` list verbatim (lines 44-85). Replace everything below:

```python
WINDOW, STEP, MIN_WINDOWS = 6, 2, 3


def windows(text, size=WINDOW, step=STEP, min_windows=MIN_WINDOWS):
    """Sliding word windows over one turn.

    ``min_windows`` prevents a real rendering artifact: ``hyp.plot`` draws a
    ONE-ROW dataset as a dot (there is no line through a single point), and
    with a fixed 6-word window, 12 of the 28 turns below collapse to a
    single window and would show up as stray specks. Shrinking the window,
    and the step if needed, keeps every turn a real path.
    """
    words = text.split()
    n = len(words)
    size = max(1, min(size, n - min_windows + 1))
    step = step if (n - size) // step + 1 >= min_windows else 1
    return [' '.join(words[i:i + size]) for i in range(0, n - size + 1, step)]


turns = [windows(text) for _speaker, text in TURNS]
# category order is FIRST APPEARANCE (hypertools/plot/colors.py:105), so the
# palette must be listed in that order for each speaker to get their colour
speakers = list(dict.fromkeys(speaker for speaker, _text in TURNS))
print(f'conversation: {len(TURNS)} turns, {len(speakers)} speakers, '
      f'{sum(len(t) for t in turns)} windows')

# THE hypertools call: raw dialogue in, one disjoint trajectory per turn,
# coloured by speaker, revealed ONE TURN AT A TIME (order='serial') with the
# already-spoken path glowing faintly behind each head (chemtrails=True).
# title= carries one string per turn, so the label under the box always names
# the turn currently being drawn -- the library drives it from the same
# reveal schedule it renders from.
duration, fps = 12, 16
FLOOR, DECAY = 0.10, 0.45
fig, ani = hyp.plot(
    turns, '-',
    vectorizer='all-MiniLM-L6-v2', semantic=None, corpus=None,
    reduce={'model': 'UMAP', 'kwargs': {'n_neighbors': 8, 'min_dist': 0.5,
                                        'random_state': 1}},
    ndims=3,
    hue=[[speaker] * len(window_list)
         for (speaker, _text), window_list in zip(TURNS, turns)],
    palette=[SPEAKER_COLOR[s] for s in speakers], legend=True,
    linewidth=1.6,
    animate=True, order='serial', chemtrails=True,
    title=[f'{speaker}   “{text}”' for speaker, text in TURNS],
    duration=duration, frame_rate=fps, elev=16, size=(8, 8), show=False)


def turn_alpha(i, revealed, current):
    """How visible turn `i` should be while turn `current` is being drawn.

    Assigns a value for EVERY dataset on EVERY frame, including turns not
    yet spoken -- the portable callback rule (animation.rst): put the
    condition in the VALUE, never around the assignment. A skipped
    assignment leaves matplotlib's shared artists at whatever the previous
    frame set, which is how a fade turns into a smear.
    """
    if i > current or revealed < 2:
        return 0.0                     # unspoken, or a single stray point
    if i == current:
        return 1.0
    return FLOOR + (1.0 - FLOOR) * DECAY ** (current - i)


def recency_fade(ctx):
    """The one bespoke effect left: earlier turns recede as the talk moves on.

    ``chemtrails``/``precog``/``bullettime`` fade WITHIN one trajectory;
    nothing in 1.1 fades ACROSS already-revealed datasets, so this is real
    custom work -- but it now runs on the public per-frame hook and reads the
    library's own published schedule instead of re-deriving it.

    ``ctx.artists`` is NOT one artist per dataset. It is heads first, then
    trails (animation_context.FrameContext), so with ``chemtrails=True`` it
    holds 2N entries against ``revealed_counts``' N. Zipping the two
    directly walks off the end of the counts. Split by role first.
    """
    current = ctx.current_index
    if current is None:
        raise RuntimeError(
            "recency_fade needs a serial reveal: ctx.current_index is None, "
            "which means this plot is animating in parallel. Keep "
            "order='serial' (or animate='serial') on the plot() call above.")

    n_datasets = len(ctx.revealed_counts)
    heads = ctx.artists[:n_datasets]
    trails = ctx.artists[n_datasets:]
    # chemtrails=True is broadcast to every dataset, so this holds here. It
    # is asserted rather than assumed because a dataset drawn marker-only
    # gets no trail artist, and the mismatch would otherwise show up as a
    # silently mis-paired head/trail rather than an error.
    if len(trails) != n_datasets:
        raise RuntimeError(
            f"expected one trail artist per dataset, got {len(trails)} "
            f"trails for {n_datasets} datasets")

    for i, (head, trail, revealed) in enumerate(
            zip(heads, trails, ctx.revealed_counts)):
        alpha = turn_alpha(i, revealed, current)
        head.set_alpha(alpha)
        # the library's own trail convention: 0.3x the head it belongs to
        # (matplotlib_backend draws trails at alpha * 0.3)
        trail.set_alpha(0.3 * alpha)


ani.on_frame(recency_fade)
```

**Why not a general head/trail mapping.** `FrameContext` publishes artists as a flat tuple with a documented order, not as a role-tagged mapping. Making the split general — every backend branch populating per-role, per-dataset artist groups identically — is a public-API expansion, and it is not needed for this example: the plot sets `chemtrails=True` for every dataset, so the layout is known and the guard above proves it at runtime. If a future example needs a mix of trailed and untrailed datasets in one plot, that is when the richer metadata earns its cost.

> **Interface check before writing this:** `FrameContext` is defined in animation-core Task 7 (`hypertools/plot/animation_context.py`) with the fields `current_index`, `revealed_counts` and `artists`, and `HyperAnimation.on_frame()` registers against the shared `FrameHooks` registry that `plot()` created (animation-core contract #3). If any field is named differently when Task 7 lands, follow the implemented names — do not add a shim here.

- [ ] **Step 2: Run the example and confirm it renders**

Run: `MPLBACKEND=Agg .venv/bin/python examples/animate_conversation.py`
Expected: exits 0, prints `conversation: 28 turns, 4 speakers, NNN windows`, no warnings.

- [ ] **Step 2a: Write the callback's tests**

The head/trail split is the part that was wrong in v1 and the part a future edit is most likely to break, so it gets real tests rather than an eyeball check. Create `tests/plot/test_recency_fade.py`:

```python
# -*- coding: utf-8 -*-
"""The conversation example's recency_fade callback.

Drives the REAL callback from the REAL example module (no reimplementation:
a copy of the logic here would pass while the example was broken). The
example is executed once per module -- it builds sentence embeddings and a
UMAP reduction, so this is not cheap -- and every case then runs against
synthetic FrameContexts, which is what makes frame ORDER testable at all.
"""

import runpy

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pytest

pytest.importorskip('sentence_transformers')
pytest.importorskip('umap')

from hypertools.plot.animation_context import FrameContext

#: `recency_fade` is dataset-count agnostic, so the fixture exercises a
#: RANGE rather than one number. v2 pinned a single `N_DATASETS = 6` while
#: telling the implementer it "must equal the number of FINAL drawn datasets
#: the example produces" -- and the example produces **28** (`TURNS` has 28
#: entries; its own docstring says "12 of the 28 turns"). So the constant
#: contradicted its own rule.
#:
#: Parametrising fixes that and covers strictly more: 1 is the degenerate
#: single-turn case, 6 keeps the fast default, and 28 is the real
#: conversation. The reviewer's exhaustive n = 1..29 sweep found no
#: n-dependent behaviour, so this formalises what was already checked ad
#: hoc rather than adding speculative coverage.
DATASET_COUNTS = (1, 6, 28)
N_DATASETS = 6      # the default for tests that do not vary it


@pytest.fixture(scope='module')
def example():
    ns = runpy.run_path('examples/animate_conversation.py')
    yield ns
    plt.close('all')


def _ctx(current, revealed=None, n=N_DATASETS, trails=True):
    """A FrameContext shaped exactly like the example's own plot: heads
    first, then one trail per dataset (chemtrails=True)."""
    heads = [plt.Line2D([], []) for _ in range(n)]
    tails = [plt.Line2D([], []) for _ in range(n)] if trails else []
    if revealed is None:
        # `current` may be None (the parallel-animation guard case), so the
        # comparison has to be guarded here too -- `i <= None` is a
        # TypeError, and it would fire in the FIXTURE before the callback
        # under test ever ran.
        revealed = tuple(10 if (current is not None and i <= current) else 0
                         for i in range(n))
    return FrameContext(
        frame=0, n_frames=100, figure=None, axes=None,
        artists=tuple(heads + tails), datasets=(),
        style=True, order='serial', current_index=current,
        current_fraction=0.5, revealed_counts=tuple(revealed))


def test_every_head_and_trail_is_assigned_on_every_frame(example):
    """The portable rule: assign the complete value on every invocation.
    A skipped assignment leaves matplotlib's shared artists at the previous
    frame's value, which is how a fade becomes a smear."""
    fade = example['recency_fade']
    ctx = _ctx(current=2)
    for art in ctx.artists:
        art.set_alpha(None)
    fade(ctx)
    assert all(a.get_alpha() is not None for a in ctx.artists), (
        'some artist was left unassigned')


@pytest.mark.parametrize('current', [0, N_DATASETS // 2, N_DATASETS - 1])
def test_first_middle_and_last_turn(example, current):
    fade = example['recency_fade']
    ctx = _ctx(current=current)
    fade(ctx)
    heads = ctx.artists[:N_DATASETS]
    assert heads[current].get_alpha() == 1.0, 'the current turn is opaque'
    for i in range(current + 1, N_DATASETS):
        assert heads[i].get_alpha() == 0.0, 'unspoken turns are invisible'
    earlier = [heads[i].get_alpha() for i in range(current)]
    assert earlier == sorted(earlier), 'older turns must not be brighter'


def test_trails_track_their_own_head(example):
    fade = example['recency_fade']
    ctx = _ctx(current=3)
    fade(ctx)
    heads, trails = ctx.artists[:N_DATASETS], ctx.artists[N_DATASETS:]
    for head, trail in zip(heads, trails):
        assert trail.get_alpha() == pytest.approx(0.3 * head.get_alpha())


def test_the_callback_never_indexes_past_revealed_counts(example):
    """The v1 defect: iterating ctx.artists (2N under chemtrails) while
    indexing ctx.revealed_counts (N) raised IndexError on the N+1th artist."""
    fade = example['recency_fade']
    fade(_ctx(current=N_DATASETS - 1))  # must not raise


def test_a_missing_trail_artist_is_an_explicit_error(example):
    """Rather than silently pairing head i with head i+1."""
    fade = example['recency_fade']
    with pytest.raises(RuntimeError, match='one trail artist per dataset'):
        fade(_ctx(current=1, trails=False))


def test_a_parallel_animation_is_an_explicit_error(example):
    fade = example['recency_fade']
    with pytest.raises(RuntimeError, match='serial'):
        fade(_ctx(current=None))


@pytest.mark.parametrize('order', [
    [0, 1, 2, 3, 4, 5],              # forward
    [5, 4, 3, 2, 1, 0],              # backward
    [3, 0, 5, 3, 1, 3],              # shuffled, with repeats
])
def test_alpha_depends_only_on_the_frame_not_on_history(example, order):
    """matplotlib re-delivers frame indices on loop and on save(), so the
    same current_index must always give the same alphas regardless of what
    ran before it."""
    fade = example['recency_fade']
    reference = {}
    for current in range(N_DATASETS):
        ctx = _ctx(current=current)
        fade(ctx)
        reference[current] = [a.get_alpha() for a in ctx.artists]
    for current in order:
        ctx = _ctx(current=current)
        fade(ctx)
        assert [a.get_alpha() for a in ctx.artists] == reference[current], (
            f'current_index={current} faded differently out of order')


def test_a_single_point_turn_stays_invisible(example):
    """revealed < 2 is a stray point, not a drawn trajectory."""
    fade = example['recency_fade']
    revealed = [10] * N_DATASETS
    revealed[1] = 1
    ctx = _ctx(current=N_DATASETS - 1, revealed=revealed)
    fade(ctx)
    assert ctx.artists[1].get_alpha() == 0.0
```

- [ ] **Step 2b: Run the callback's tests**

Run: `.venv/bin/python -m pytest tests/plot/test_recency_fade.py -v`

Expected: **12 passed**, derived from the block above (8 `def test_` functions, 2 of them parametrized):

| test | IDs |
|-|-|
| `test_every_head_and_trail_is_assigned_on_every_frame` | 1 |
| `test_first_middle_and_last_turn` (3 params) | 3 |
| `test_trails_track_their_own_head` | 1 |
| `test_the_callback_never_indexes_past_revealed_counts` | 1 |
| `test_a_missing_trail_artist_is_an_explicit_error` | 1 |
| `test_a_parallel_animation_is_an_explicit_error` | 1 |
| `test_alpha_depends_only_on_the_frame_not_on_history` (3 orders) | 3 |
| `test_a_single_point_turn_stays_invisible` | 1 |
| **total** | **12** |

`DATASET_COUNTS` deliberately does **not** have to track the example: `recency_fade` is dataset-count agnostic and the fixture parametrises over 1, 6 and 28 to prove it. What *does* have to be checked against the real example is that 28 is still its FINAL drawn dataset count — `hue=` reshapes, so confirm with `len(ctx.revealed_counts)` from the Step 3 script below and update the tuple if the example's turn count changes.

- [ ] **Step 3: Confirm the reveal, the legend and the titles**

Run:

```bash
MPLBACKEND=Agg .venv/bin/python - <<'PY'
import runpy
ns = runpy.run_path('examples/animate_conversation.py')
fig, ani = ns['fig'], ns['ani']
ax = [a for a in fig.axes if hasattr(a, 'zaxis')][0]
titles = []
for f in (0, 60, 120, 191):
    ani._func(f, *ani._args)
    titles.append(ax.get_title())
legend = ax.get_legend() or (fig.legends[0] if fig.legends else None)
print('legend:', [t.get_text() for t in legend.get_texts()])
print('titles at frames 0/60/120/191:')
for t in titles:
    print('   ', t[:70])
print('distinct alphas:', sorted({round(l.get_alpha() or 1.0, 2) for l in ax.lines}))
PY
```

Expected: the legend has exactly **4 entries** in first-appearance order (`Alice`, `March Hare`, `Hatter`, `Dormouse`); the four titles are **different** and each begins with a speaker name; and at least three distinct alpha values are present (the fade is working). If every title is identical, `title=` did not receive the per-segment list — check `order='serial'` reached `_validate_title`.

- [ ] **Step 4: Rewrite the notebook in lockstep**

Rewrite `docs/tutorials/conversation_shape.ipynb`, keeping cell 0 unchanged:

| cell | type | content |
|-|-|-|
| 0 | code | existing Colab install cell — unchanged |
| 1 | markdown | title + docstring prose, incl. the "spoken text only" note |
| 2 | markdown | `## 1. Imports and the dialogue` |
| 3 | code | imports, `SPEAKER_COLOR`, `TURNS` |
| 4 | markdown | `## 2. One trajectory per turn` |
| 5 | code | `WINDOW`/`STEP`/`MIN_WINDOWS`, `windows`, `turns`, `speakers`, `print` |
| 6 | markdown | `## 3. One call: text in, a serial reveal out` — name each kwarg's job, and that `title=` takes one string per turn for serial-style animations |
| 7 | code | the `hyp.plot(...)` call |
| 8 | markdown | `## 4. The one bespoke effect: a recency fade, on the public hook` |
| 9 | code | `recency_fade` + `ani.on_frame(recency_fade)` |
| 10 | markdown | `## 5. Display the animation` |
| 11 | code | `HTML(ani.to_jshtml())` |

- [ ] **Step 5: Execute and measure**

```bash
.venv/bin/python scripts/execute_tutorial.py docs/tutorials/conversation_shape.ipynb
.venv/bin/python scripts/measure_native_ratio.py \
    examples/animate_conversation.py docs/tutorials/conversation_shape.ipynb
```

Expected: both files inside budget (**≤ 90 / ≤ 95 code lines**). Record the measured visible-output index set into `EXPECTED_VISIBLE_OUTPUTS`; do not assert a predicted count.

- [ ] **Step 6: Commit**

```bash
git add examples/animate_conversation.py docs/tutorials/conversation_shape.ipynb
git commit -m "docs(gallery): conversation example uses native text, order='serial', and per-segment titles"
```

---

## Task 6: Morph — per-segment titles, natively

> **v3: THE SCRIPT HALF OF THIS TASK IS ALREADY DONE.** Commit `d730a085` landed it on 2026-08-01. Step 1 below is retained only as a record of what was asked for; **do not apply it** — see "Step 1 (ALREADY LANDED)". The remaining work is the notebook, which still reaches into the private API the script has stopped using.

**BEFORE — re-measured 2026-08-02 at `065c841e`, with the docstring-aware metric:**

| file | v2 said | actual now | note |
|-|-|-|-|
| `examples/animate_morph_zoo.py` | 129 raw, 40 code, 6 native (15.0%) | **96 raw, 26 code, 6 native (23.1%)** | `d730a085` already deleted the workaround |
| `docs/tutorials/morph_shapes_zoo.ipynb` | 45 code, 8 native (17.8%), "0 of 6 executed" | **46 code, 9 native (19.6%), 1 of 6 cells carries output** | the "0 executed" claim was never true |

**AFTER (contracted budget):** script **≤ 45 code lines** (30 for the file + 15 placeholder for the Step 0b split); notebook **≤ 50** (= 45 + 5); **zero** defect markers. The script is at 26 today and needs only the split — plus an offline fallback for `hyp.load`, which is the one loader in the five that hard-fails rather than degrading. No ratio floor (Contract 6a).

**Gate status today:** the script is **the only one of the five that already passes** both the defect-marker scan and its budget. The notebook fails on `ani._func` and `from hypertools.plot import morph`.

**What goes, and to what:**

| deleted | replaced by |
|-|-|
| `from hypertools.plot import morph as _morph` (`:35`) and the `morph_schedule` recomputation with its hardcoded `azim0=-60` (`:105-107`, class **C**) | nothing — the schedule is the library's business again |
| `shape_title`, `label`, `_wrapped`, `ani._func = _wrapped` (`:108-128`, class **C**) | `title=titles` (animation-core Task 8), which blanks morph **transitions** by segment **parity** and names every **hold** |

**Kept, deliberately, with the reasons the current file already states correctly:**
- the **teapot** (maintainer instruction), with its `hyp.load('teapot')` 1728-rows / 301-unique note (`:45-50`);
- `CUBE_SCALE = 0.8`, because a cube normalized to ±1 fills the drawn axes box exactly and reads as noise in a wireframe (`:63-66`);
- the closed loop `clouds.append(clouds[0])`, and the hand sampling that makes it possible — `morph_samples=` draws a **fresh** subset per dataset, so it cannot produce the identical closing sample (`:54-61`);
- `normalize()`, because `plot()` rescales with **one shared pooled affine** (`plot.py:4568-4605`, `_shared/helpers.py:24-69`), so clouds left in their raw units would be drawn at wildly different sizes. See the *Decisions still needed* entry named **"The morph example's hand-written `normalize()`"**;
- the explicit `morph_samples=N`. **Corrected 2026-07-30 — this bullet previously said animation-core Task 3 makes an uncapped morph above 2000 points "raise", which is the pre-`simplify=` behaviour and is no longer true.** Under the resolved decision, the default `simplify=True` **silently downsamples** to the cap; only `simplify=False` raises. So `morph_samples=N` is *not* load-bearing in the "or it errors" sense — the example would still run without it. Keep it anyway, and for a better reason: it makes the sampling **explicit and reproducible** rather than leaving the reader to discover that a silent cap was applied. State that in the prose.

**Files:** rewrite the tail of `examples/animate_morph_zoo.py`; rewrite `docs/tutorials/morph_shapes_zoo.ipynb`.

- [x] **Step 1 (ALREADY LANDED — verify, do not apply)**

`d730a085` implemented this. **Verify rather than rewrite:**

```bash
grep -nE "_morph|morph_schedule|ani\._func|shape_title" examples/animate_morph_zoo.py
# expected: no output -- every class-C workaround is gone
grep -n "title=titles" examples/animate_morph_zoo.py
# expected: one hit, inside the hyp.plot(...) call
```

Both were run on 2026-08-02 and give exactly that. The landed code is **semantically equivalent to the block below, not textually identical** — the comment wording differs and `title=titles` sits at the end of the call rather than mid-list. That is fine; the contract is the behaviour, and re-applying the block would be a pointless diff. The docstring rewrite also landed, in different words that say the same thing.

**If the greps above do NOT come back clean**, the migration was reverted somewhere; only then apply the original instruction, recorded here for that case: delete `from hypertools.plot import morph as _morph` and replace everything from the `hyp.plot` call to the end of the file with:

```python
# THE hypertools call: black pixel-sized dots morphing through the zoo, with
# one title per shape. For a morph, title= takes one string per cloud: each
# is shown while its cloud is fully formed and blanked through the
# transitions, so the label never sits over a half-formed shape. hypertools
# drives it from the same segment schedule it renders from -- this file no
# longer recomputes that schedule, and no longer has to know plot's default
# azimuth.
duration, fps = 12, 20
fig, ani = hyp.plot(clouds, fmt='.', color='k', markersize=1.6,
                    animate='morph', rotations=rotations, morph_samples=N,
                    title=titles, duration=duration, frame_rate=fps,
                    size=(6, 6), show=False)
```

Update the module docstring's second paragraph (`:14-22`) to describe the native feature rather than the workaround:

```
The shape names come straight from the library: ``title=`` takes one string
per cloud for a morph animation, shown while that cloud is fully formed and
blanked through the transitions, so the label never sits over a half-formed
shape. Nothing here recomputes hypertools' morph schedule or reaches into
its private modules.
```

- [ ] **Step 0: Split the loader from the figure builder (Contract 4 / Step 0b)**

**Do this FIRST, before the rewrite below.** Task 8 Step 0b defines the contract and works the whole pattern through on weather; this step applies it to `examples/animate_morph_zoo.py`. Without it `test_examples_produce_their_stated_artifact` fails on this example, and importing it fetches.

Produce exactly these three names in `examples/animate_morph_zoo.py`:

| name | signature | notes |
|-|-|-|
| payload | `class Shapes(NamedTuple)` with fields `clouds, titles` | self-documenting; `source` records which path was used |
| loader | `load_shapes(SHAPES, n=N) -> Shapes` | the ONLY code here that may touch the network (hyp.load -- which has NO offline fallback today and hard-fails, exit 1) |
| fixture | `fixture_data() -> Shapes` | deterministic parametric clouds — no network, no committed bytes unless stated |
| builder | `construct_artifact(data) -> HyperAnimation` | everything else, reading `data.<field>` instead of module globals. **Returns the wrapper, never the unpacked pair** (Contract 8) |

Then move every loader CALL behind a `__main__` guard, and make each fetcher honour `HYPERTOOLS_OFFLINE` by raising:

```python
if __name__ == '__main__':
    data = load_shapes()
    anim = construct_artifact(data)
    fig = anim.figure
```

Verify before moving on:

```bash
MPLBACKEND=Agg .venv/bin/python -c "
import importlib.util, os
os.environ['HYPERTOOLS_OFFLINE'] = '1'
spec = importlib.util.spec_from_file_location('m', 'examples/animate_morph_zoo.py')
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
print('imported with no fetch; has', [n for n in ('construct_artifact', 'fixture_data') if hasattr(m, n)])
anim = m.construct_artifact(m.fixture_data()); print('frames:', anim.n_frames)"
```

Expected: it imports without touching the network and prints both names plus a frame count. If the import fetches, a loader call is still at module scope.

- [ ] **Step 2: Run the example and confirm the titles track the schedule**

```bash
MPLBACKEND=Agg .venv/bin/python examples/animate_morph_zoo.py
MPLBACKEND=Agg .venv/bin/python - <<'PY'
import runpy
from hypertools.plot.morph import segment_frame_counts, frame_to_segment
ns = runpy.run_path('examples/animate_morph_zoo.py')
fig, ani, titles = ns['fig'], ns['ani'], ns['titles']
ax = [a for a in fig.axes if hasattr(a, 'zaxis')][0]
total = 12 * 20
counts = segment_frame_counts(len(ns['clouds']), total)
bad = []
for frame in range(total):
    ani._func(frame, *ani._args)
    seg, _step, _n = frame_to_segment(counts, frame)
    expected = titles[seg // 2] if seg % 2 == 0 else ''
    if ax.get_title() != expected:
        bad.append((frame, seg, ax.get_title(), expected))
print('frames checked:', total, '| mismatches:', len(bad), bad[:5])
PY
```

Expected: exits 0 and prints `frames checked: 240 | mismatches: 0 []`. Any mismatch means the native titles are not tracking `frame_to_segment`'s parity — that is animation-core Task 8's contract, so fix it there, not here.

- [ ] **Step 3: Rewrite the notebook in lockstep**

Rewrite `docs/tutorials/morph_shapes_zoo.ipynb`. The current cell 9 (24 lines of schedule recomputation and `_func` monkeypatching) is **deleted outright**, and its markdown heading (cell 8, `## 4. A title that tracks the current shape`) is folded into the plot cell's markdown:

| cell | type | content |
|-|-|-|
| 0 | code | existing Colab install cell — unchanged |
| 1 | markdown | title + updated docstring prose |
| 2 | markdown | `## 1. Imports` |
| 3 | code | `import numpy as np` / `import hypertools as hyp` |
| 4 | markdown | `## 2. Load, normalize, sample, and close the loop` (keep the teapot note and the "why normalize by hand" note) |
| 5 | code | `SHAPES`, `TITLES`, `N`, `CUBE_SCALE`, `rng`, `normalize`, `load`, `clouds`, `titles` |
| 6 | markdown | `## 3. One call: morph, with one title per shape` |
| 7 | code | `rotations`, `duration`/`fps`, the `hyp.plot(...)` call |
| 8 | markdown | `## 4. Display (or save) the animation` |
| 9 | code | `HTML(ani.to_jshtml())` / `# or: ani.save('morph_zoo.gif', fps=fps)` |

- [ ] **Step 4: Execute and measure**

```bash
.venv/bin/python scripts/execute_tutorial.py docs/tutorials/morph_shapes_zoo.ipynb
.venv/bin/python scripts/measure_native_ratio.py \
    examples/animate_morph_zoo.py docs/tutorials/morph_shapes_zoo.ipynb
```

Expected: both files inside budget (**≤ 30 / ≤ 35 code lines**). Record the measured visible-output index set into `EXPECTED_VISIBLE_OUTPUTS`; do not assert a predicted count.

- [ ] **Step 5: Commit**

```bash
git add examples/animate_morph_zoo.py docs/tutorials/morph_shapes_zoo.ipynb
git commit -m "docs(gallery): morph example uses native per-segment titles, drops the private schedule reach"
```

---

## Task 7: The fifteen older tutorials

Grouped by the recurring fix so each step is one reviewable diff. Every group ends by executing the touched notebooks and committing.

**Baseline** (from `notes/audit/other_tutorials_audit.md`, §1): `conversation_trajectories` 2.5%, `projectile_kalman` 3.2%, `stock_forecasting` 3.7%, `wikipedia_embeddings` 7.1%, `hugging_face_embeddings` 10.0%, `modern_sklearn_dynamics` 10.2%, `analyze` 20.0% (and **never calls `hyp.plot`**), `reduce` 36.8% (never plots, never mentions `hyp.describe`). The seven clean ones (`align`, `plot`, `normalize`, `cluster`, `streaming_data`, `text`, `lsl_streaming`) are **not touched**.

- [ ] **Step 1 (G2): Delete the four ffmpeg cells; ask for a GIF directly**

`save_path='foo.gif'` writes a GIF with **no ffmpeg at all** (`plot.py:1513-1520`, writer dispatch at `animate.py:84`) — verified today: a real 24 832-byte GIF. Three notebooks in this same set already prove it (`streaming_data` cells 4/8, `lsl_streaming` cell 6).

For each pair below, change the `save_path='*.mp4'` in the plot cell to `'*.gif'`, drop the `print(f"mp4: ...")` line that follows it, and **delete the entire next cell**:

| notebook | plot cell | ffmpeg cell to delete |
|-|-|-|
| `conversation_trajectories.ipynb` | 14 (`save_path='conversation_serial.mp4'`) | 15 (15 lines) |
| `hugging_face_embeddings.ipynb` | 12 (`save_path='hf_embeddings_spin.mp4'`) | 13 (16 lines) |
| `modern_sklearn_dynamics.ipynb` | 12 (`save_path='lorenz_trajectory.mp4'`) | 13 (16 lines) |
| `wikipedia_embeddings.ipynb` | 10 (`save_path='wikipedia_embeddings_spin.mp4'`) | 11 (15 lines) |

Also delete the markdown cell immediately before each ffmpeg cell if it exists only to explain the mp4→gif conversion, and update the surviving markdown to say the GIF is written directly.

Run: `.venv/bin/python scripts/execute_tutorial.py docs/tutorials/conversation_trajectories.ipynb docs/tutorials/hugging_face_embeddings.ipynb docs/tutorials/modern_sklearn_dynamics.ipynb docs/tutorials/wikipedia_embeddings.ipynb`
Expected: each reports at least as many executed cells as before, and **no `ffmpeg not found` message appears anywhere in the outputs**:
```bash
grep -l "ffmpeg" docs/tutorials/*.ipynb
```
Expected: no output.

```bash
git add docs/tutorials/conversation_trajectories.ipynb docs/tutorials/hugging_face_embeddings.ipynb \
        docs/tutorials/modern_sklearn_dynamics.ipynb docs/tutorials/wikipedia_embeddings.ipynb
git commit -m "docs(tutorials): save GIFs natively; delete 62 lines of ffmpeg boilerplate"
```

- [ ] **Step 2 (G1): Delete the four hand-rolled sentence-transformer blocks**

Two of these notebooks *already document the native call in adjacent markdown and then ignore it* (`hugging_face_embeddings` cell 3, `wikipedia_embeddings` cell 5). Native form, verified: `vectorizer='<hf-model-id>', semantic=None, corpus=None` (`text2mat.py:89`, dispatch `:184`, `semantic` `:391`, `corpus` `:404`).

| notebook / cell | delete | replace with |
|-|-|-|
| `hugging_face_embeddings` cell 4 | `model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')` + `embeddings = model.encode(headlines, ...)` + `embeddings.shape` | keep the `load_dataset(...)`/`headlines`/`categories` lines; move the embedding into every downstream `hyp.plot` call as `vectorizer='all-MiniLM-L6-v2', semantic=None, corpus=None` and pass `headlines` (the raw strings) as the data |
| `wikipedia_embeddings` cell 6 | `model = SentenceTransformer('BAAI/bge-small-en-v1.5')` + `embeddings = model.encode(truncated, ...)` | keep `truncated = [a[:2000] for a in articles]`; pass `truncated` to `hyp.plot` with `vectorizer='BAAI/bge-small-en-v1.5', semantic=None, corpus=None` |
| `wikipedia_embeddings` cell 17 | the second `model.encode(...)` | same, on `live_truncated` |
| `conversation_trajectories` cell 8 | `SentenceTransformer(...).encode(...)`, the running-`start` re-split loop, and `np.vstack([emb, emb])` | pass `utterance_windows` (already a **list of lists of strings**) straight to `hyp.plot`; `format_data` splits per dataset — verified `hyp.reduce([[s,s,s],[s,s,s],[s,s]], ndims=3)` → `[(3,3),(3,3),(2,3)]` |

Also delete the now-unused `from sentence_transformers import SentenceTransformer` imports, and update the markdown cells that promised the native call so they now describe what the code does.

> **`np.vstack([emb, emb])` caveat:** it existed to dodge a one-row dataset (audit M2). If any utterance still yields a single window after the change, the notebook must handle it in *text* space (widen or drop that utterance) and say so — never by duplicating an embedded row.

Run: `.venv/bin/python scripts/execute_tutorial.py docs/tutorials/hugging_face_embeddings.ipynb docs/tutorials/wikipedia_embeddings.ipynb docs/tutorials/conversation_trajectories.ipynb`
Then:
```bash
grep -l "SentenceTransformer" docs/tutorials/*.ipynb examples/*.py
```

**Dependency note:** unlike the rest of Task 7, this gate is not runnable in isolation -- it globs `examples/*.py`, and `examples/animate_conversation.py` and `examples/animate_painting_embeddings.py` both still hand-roll `SentenceTransformer` today (verified: `grep -l SentenceTransformer examples/*.py` currently matches both). The gate only reports "no output" once Task 4 (Paintings) and Task 5 (Conversation) have rewritten those two files to use `vectorizer=`. Run this check after Tasks 4-5 land, not in parallel with them.
Expected: no output.

```bash
git add docs/tutorials/hugging_face_embeddings.ipynb docs/tutorials/wikipedia_embeddings.ipynb \
        docs/tutorials/conversation_trajectories.ipynb
git commit -m "docs(tutorials): embed text with vectorizer=<hf-id> instead of hand-rolling it"
```

- [ ] **Step 3 (G3): Route the hand-drawn comparison figures through `hyp.plot(..., ax=)`**

Verified today: `hyp.plot([d, d + 1, d - 1], ['-', '--', '*'], reduce=None, ndims=2, ax=axes[0, 0], legend=['train', 'held out', 'forecast'], show=False)` returns a `Figure` and draws **3 lines on the supplied axes**, with no warnings.

| notebook / cell | today | becomes |
|-|-|-|
| `stock_forecasting` cell 14 (27 lines) | `plt.subplots(2, 2)` + 3 `ax.plot` calls per panel + manual labels/legend, **re-running `hyp.predict` inside the plotting loop** | keep `plt.subplots(2, 2)`; per panel one `hyp.plot([train, held_out, forecast], ['-', '--', '*'], reduce=None, ndims=2, ax=ax, legend=[...], show=False)`. Reuse the forecasts cell 6 already computed rather than refitting |
| `projectile_kalman` cell 6 (9 lines) | `plt.subplots` + `ax.plot(arc['x_ft'], arc['z_ft'], marker='o')` | `hyp.plot(arc[['x_ft', 'z_ft']], '-o', reduce=None, ndims=2, xlabel='court position, x (ft)', ylabel='ball height, z (ft)', title='Real SportVU jump-shot arc (side view): a genuine parabola', show=False)` (`plot.py:1013` documents `xlabel`/`ylabel`/`zlabel`) |
| `projectile_kalman` cell 15 (12 lines) | 3 hand-drawn series with a manual legend | `hyp.plot([first30[['x_ft', 'z_ft']], actual_tail[['x_ft', 'z_ft']], forecast[['x_ft', 'z_ft']]], ['-o', '-o', '--x'], reduce=None, ndims=2, legend=['observed (frames 0-29)', 'actual (frames 30-49)', 'Kalman forecast'], show=False)` — the exact shape `plot.ipynb` cell 32 already demonstrates |

`projectile_kalman` cell 11 (the 1×3 per-feature-vs-time panel grid) is **left alone**: per-feature-vs-time panels are not what `hyp.plot` draws, and the audit classifies it as defensible.

Run: `.venv/bin/python scripts/execute_tutorial.py docs/tutorials/stock_forecasting.ipynb docs/tutorials/projectile_kalman.ipynb`
Expected: both execute; the figures render with the same series and legends as before.

```bash
git add docs/tutorials/stock_forecasting.ipynb docs/tutorials/projectile_kalman.ipynb
git commit -m "docs(tutorials): draw comparison figures with hyp.plot(..., ax=) instead of raw matplotlib"
```

- [ ] **Step 4 (smoothing): pandas rolling mean → `manip='Smooth'`**

`stock_forecasting` cell 12 builds its log-volume column with `pandas.rolling(smooth, min_periods=1).mean()` and never mentions `manip=`. Replace the rolling call with the plot-stage kwarg:

```python
manip={'model': 'Smooth', 'kwargs': {'kernel': 'boxcar', 'kernel_width': 11}}
```

on the `hyp.plot(...)` call at the end of the same cell (`plot.py:1064`; kernels at `hypertools/manip/smooth.py:14`). **Use 11, not 10**: `kernel_width=10` emits `UserWarning: Increasing smoothing kernel width by 1 (must be odd)` (`hypertools/manip/smooth.py:232`) — measured today. Update the markdown to say the smoothing now runs at the canonical first pipeline stage rather than in pandas.

Run: `.venv/bin/python scripts/execute_tutorial.py docs/tutorials/stock_forecasting.ipynb`
Expected: executes with **no `UserWarning` about kernel width** in any cell output:
```bash
grep -c "must be odd" docs/tutorials/stock_forecasting.ipynb
```
Expected: `0`.

```bash
git add docs/tutorials/stock_forecasting.ipynb
git commit -m "docs(tutorials): smooth with manip='Smooth' at the pipeline stage, not pandas.rolling"
```

- [ ] **Step 5 (structural): make `analyze.ipynb` plot, and `reduce.ipynb` describe**

`analyze.ipynb` **never calls `hyp.plot`** — a pipeline tutorial that shows `normalize → reduce → align` only as `sb.heatmap(x)` never demonstrates why the pipeline exists. Cells 18, 23 and 28 each hold the identical 3-line seaborn loop over **already-reduced 3-D output**, which is exactly what `hyp.plot` is for:

```python
# cells 18, 23, 28 -- replace
for x in <result>:
    sb.heatmap(x)
    plt.show()
# with
hyp.plot(<result>, '.', reduce=None, show=False)
```

Cells 8 and 13 operate on the raw / normalized **high-dimensional** matrices, where a heatmap is a reasonable stand-in (there is no native matrix view; audit M10) — leave them, and add one markdown sentence saying so, so the contrast is deliberate rather than accidental.

`reduce.ipynb` never plots and never mentions `hyp.describe()` (`hypertools/reduce/describe.py:13-23`, *"Useful for evaluating quality of dimensionality reduced plots"*) — the obvious companion to a reduction tutorial. Append two cells:

```python
# markdown: "## How many dimensions do you actually need?"
scores = hyp.describe(data, show=False)
print({k: (v if not hasattr(v, '__len__') else list(v)[:5]) for k, v in scores.items()})
hyp.plot(hyp.reduce(data, ndims=3), '.', reduce=None, show=False)
```

(`hyp.describe(..., show=False)` returns a **dict** — verified today — so print it rather than treating it as a figure.)

Run: `.venv/bin/python scripts/execute_tutorial.py docs/tutorials/analyze.ipynb docs/tutorials/reduce.ipynb`
Then:
```bash
.venv/bin/python -c "
import json
for nb in ('analyze', 'reduce'):
    src = ''.join(''.join(c['source']) for c in json.load(open(f'docs/tutorials/{nb}.ipynb'))['cells'] if c['cell_type']=='code')
    print(nb, 'hyp.plot:', src.count('hyp.plot'), '| hyp.describe:', src.count('hyp.describe'))"
```
Expected: `analyze hyp.plot: 3 | hyp.describe: 0` and `reduce hyp.plot: 1 | hyp.describe: 1`.

```bash
git add docs/tutorials/analyze.ipynb docs/tutorials/reduce.ipynb
git commit -m "docs(tutorials): analyze.ipynb finally plots its pipeline; reduce.ipynb gains hyp.describe"
```

- [ ] **Step 6: Re-measure the eight touched notebooks**

```bash
.venv/bin/python scripts/measure_native_ratio.py \
    docs/tutorials/conversation_trajectories.ipynb docs/tutorials/hugging_face_embeddings.ipynb \
    docs/tutorials/wikipedia_embeddings.ipynb docs/tutorials/modern_sklearn_dynamics.ipynb \
    docs/tutorials/stock_forecasting.ipynb docs/tutorials/projectile_kalman.ipynb \
    docs/tutorials/analyze.ipynb docs/tutorials/reduce.ipynb
```

Expected: every one of the eight has a **strictly higher** ratio than the audit's baseline (2.5 / 10.0 / 7.1 / 10.2 / 3.7 / 3.2 / 20.0 / 36.8 percent). Record the measured numbers in the commit message of Step 5 if they are not already there; Task 8 turns them into an assertion.

---

## Task 8: Verification — measure it, and keep it measured

- [ ] **Step 0: Give `HyperAnimation` the three accessors the gate needs**

**Files:** Modify `hypertools/plot/hyper_animation.py`; test `tests/plot/test_hyper_animation_accessors.py` (create).

**Why this is library work and not test-only.** The v2 gate inspected `ani._save_count` — matplotlib's private field — which Contract 3 forbids and `DEFECT_MARKERS` lists ten lines above the gate that used it. A gate may not reach for what it bans. `HyperAnimation` today exposes only `figure` and `animation` (`hyp_animation.py:67`, `:72`), so the supported accessors have to exist first.

Write these tests first:

```python
# tests/plot/test_hyper_animation_accessors.py
"""n_frames / n_segments / draw_frame -- the supported way to inspect and
drive an animation, replacing reaches into FuncAnimation internals."""

import matplotlib
matplotlib.use('Agg')

import numpy as np
import pytest

import hypertools as hyp


def _data(n=60, d=3, seed=0):
    return np.random.default_rng(seed).normal(size=(n, d)).cumsum(axis=0)


def test_n_frames_matches_the_requested_rate_and_duration():
    anim = hyp.plot(_data(), '-', animate=True, duration=4, frame_rate=10,
                    show=False)
    assert anim.n_frames == 40


def test_n_frames_is_never_zero_for_a_sub_frame_request():
    """`max(1, ...)`: an animation that asks for less than one frame still
    draws one. Pinned because the gate's floor assertion is only meaningful
    if this cannot silently be 0."""
    anim = hyp.plot(_data(), '-', animate=True, duration=0.01, frame_rate=1,
                    show=False)
    assert anim.n_frames == 1


def test_n_frames_survives_being_read_twice():
    anim = hyp.plot(_data(), '-', animate=True, duration=2, frame_rate=5,
                    show=False)
    assert anim.n_frames == anim.n_frames == 10


def test_n_segments_counts_holds_and_transitions():
    """`n` clouds give `2n - 1` segments: n holds interleaved with n-1
    transitions, ending on a hold. There is NO implicit closing transition
    back to the first cloud -- a caller who wants the loop to close appends
    `clouds[0]` itself, as `examples/animate_morph_zoo.py` does. Measured
    against `morph.segment_frame_counts`: 2 clouds -> 3, 3 -> 5, 5 -> 9."""
    clouds = [_data(40, 3, s) for s in range(3)]
    anim = hyp.plot(clouds, '.', animate='morph', duration=6, frame_rate=5,
                    show=False)
    assert anim.n_segments == 5


def test_n_segments_is_none_for_a_non_morph_animation():
    anim = hyp.plot(_data(), '-', animate=True, duration=2, frame_rate=5,
                    show=False)
    assert anim.n_segments is None


def test_draw_frame_renders_the_requested_index():
    anim = hyp.plot(_data(), '-', animate=True, duration=2, frame_rate=5,
                    show=False)
    ax = anim.figure.axes[0]
    anim.draw_frame(0)
    early = len(np.asarray(ax.lines[0].get_data_3d())[0])
    anim.draw_frame(anim.n_frames - 1)
    late = len(np.asarray(ax.lines[0].get_data_3d())[0])
    assert late > early, 'a later frame must reveal more of the trajectory'


def test_draw_frame_is_idempotent_and_order_independent():
    """The FrameContext contract: callbacks must be deterministic for a
    given frame, so driving out of order must give identical geometry."""
    anim = hyp.plot(_data(), '-', animate=True, duration=2, frame_rate=5,
                    show=False)
    ax = anim.figure.axes[0]
    anim.draw_frame(3)
    once = np.asarray(ax.lines[0].get_data_3d()).copy()
    anim.draw_frame(7)
    anim.draw_frame(0)
    anim.draw_frame(3)
    assert np.allclose(np.asarray(ax.lines[0].get_data_3d()), once)


def test_draw_frame_rejects_an_out_of_range_index():
    anim = hyp.plot(_data(), '-', animate=True, duration=2, frame_rate=5,
                    show=False)
    with pytest.raises(IndexError, match='0 and 9'):
        anim.draw_frame(anim.n_frames)
```

Run: `.venv/bin/python -m pytest tests/plot/test_hyper_animation_accessors.py -v`
Expected: **9 failed** — `AttributeError: 'HyperAnimation' object has no attribute 'n_frames'` and the same for `n_segments`/`draw_frame`.

Then implement, beside the existing `figure`/`animation` properties:

```python
    @property
    def n_frames(self):
        """How many frames this animation draws.

        `hyp.plot` always hands `FuncAnimation` an int frame count --
        `max(1, round(frame_rate * duration))` for parallel/serial/spin, and
        `sum(segment_frame_counts(...))` for a morph -- so this is exact
        rather than an estimate. Reading it is the supported alternative to
        matplotlib's private `_save_count`.
        """
        return int(self[1]._save_count)

    @property
    def n_segments(self):
        """Hold/transition segments for ``animate='morph'``; ``None``
        otherwise.

        `n` clouds give ``2n - 1`` segments: `n` holds interleaved with
        `n - 1` transitions, beginning and ending on a hold. There is NO
        implicit closing transition back to the first cloud -- a caller who
        wants the animation to loop seamlessly appends ``clouds[0]``
        themselves, as ``examples/animate_morph_zoo.py`` does (5 shapes plus
        the repeat = 6 clouds = 11 segments). Measured against
        ``morph.segment_frame_counts``: 2 clouds -> 3, 3 -> 5, 5 -> 9.
        """
        return getattr(self[1], '_hyp_morph_segments', None)

    def draw_frame(self, frame):
        """Render frame `frame`, and return `self` so calls chain.

        The supported way to drive an animation from a test or a script
        without reaching into `FuncAnimation._func`/`._args`. Frames are
        idempotent and order-independent by contract (see `FrameContext`),
        so any index may be drawn at any time.
        """
        if not 0 <= frame < self.n_frames:
            raise IndexError(
                f'frame {frame} is out of range; this animation has '
                f'{self.n_frames} frames, so valid indices are 0 and '
                f'{self.n_frames - 1}')
        self[1]._func(frame, *self[1]._args)
        return self
```

`_hyp_morph_segments` is tagged where the morph frame counts are already computed, beside the `sum(frame_counts)` that becomes `_save_count`:

```python
    line_ani._hyp_morph_segments = len(frame_counts)
```

**There are TWO such sites, and both must be tagged** — 3-D at `matplotlib_backend.py:2036` (`sum(frame_counts)` at `:2039`) and 2-D at `:2448` (`:2451`). Tagging only the 3-D one leaves `anim.n_segments is None` for every 2-D morph, which is worse than an error: `test_n_segments_is_none_for_a_non_morph_animation` would then PASS on a 2-D morph, so the gate would confirm a wrong answer. Add a 2-D case to that test:

```python
def test_n_segments_is_set_for_a_2d_morph_too():
    """Two FuncAnimation morph branches exist (3-D and 2-D); a tag on only
    one makes n_segments silently None for half of them."""
    clouds = [_data(40, 2, s) for s in range(3)]
    anim = hyp.plot(clouds, '.', animate='morph', duration=6, frame_rate=5,
                    reduce=None, show=False)
    assert anim.n_segments == 5
```

Run: `.venv/bin/python -m pytest tests/plot/test_hyper_animation_accessors.py -v`
Expected: **9 passed** (the 8 above plus the 2-D morph case).

Then the whole suite, since `hyper_animation.py` is on every animated path:
`.venv/bin/python -m pytest -q` → baseline + 8, no failures.

> **The private access is now in ONE place, inside the library, where it belongs.** `draw_frame` and `n_frames` still touch `_func`/`_args`/`_save_count` — that is unavoidable, because matplotlib exposes no public equivalent — but the library is entitled to know its own backend's internals, and every example, notebook and test now goes through the documented accessor instead of repeating the reach. That is the same reasoning as Contract 3's allowlist, applied one layer down.

- [ ] **Step 0b: The loader / builder split each example must expose**

**This step defines two functions per example. Tasks 2–6 each WRITE them; nothing else in the plan defines them, so skipping this step leaves `test_examples_produce_their_stated_artifact` calling names that do not exist.**

Every example splits in half:

```
fetch/load data  ->  load_<thing>()          # the ONLY code that may touch the network
                          |
                     construct_artifact(data) -> HyperAnimation   # no network, no I/O
                          |
                     fixture_data()          # the same payload, built from the example's
                                             # OWN seeded synthetic fallback -- what tests drive
```

**Required signatures, per example.** Each returns a `NamedTuple` so the fields are self-documenting:

| example | loader | payload fields | fixture bytes |
|-|-|-|-|
| `animate_weather_decades` | `load_weather(cities=CITIES)` | `Weather(monthly, daily, hemispheres, source)` | **0** — its own seeded `synthetic_city_months`/`synthetic_city_daily` |
| `animate_market_forecast` | `load_market(ids=FRED_IDS)` | `Market(dates, prices, source)` | **0** — its own seeded `synthetic_basket()` |
| `animate_conversation` | `embed_turns(TURNS)` | `Conversation(vectors, speakers, spans, source)` | **0** — the TF-IDF branch is a real `sklearn` fit, already deterministic |
| `animate_painting_embeddings` | `load_paintings(PAINTINGS)` | `Paintings(vectors, owners, colors, source)` | **one 1.7 KB** 64-px thumbnail (measured: 48 px = 1258 B, 64 px = 1744 B, 96 px = 2967 B) |
| `animate_morph_zoo` | `load_shapes(SHAPES, n=N)` | `Shapes(clouds, titles)` | **0** — deterministic parametric clouds |

**`HYPERTOOLS_OFFLINE` has to be made real — nothing reads it today.** (`grep -rn HYPERTOOLS_OFFLINE examples/ hypertools/ scripts/ tests/` returns nothing.) Each fetcher gains one line at its top, so the variable actually does something:

```python
def fetch_city_months(name, lat, lon):
    if os.environ.get('HYPERTOOLS_OFFLINE'):
        raise RuntimeError(
            'HYPERTOOLS_OFFLINE is set; refusing to fetch. This is the '
            'gate proving the import path performs no network access.')
    ...                                     # existing body unchanged
```

Without this the env var is decoration and the Task 8 helper's guarantee is false. **And no example has a `__main__` guard today** (measured: `grep -c "__main__" examples/animate_*.py` → 0 for all ten), so every loader currently runs at module scope — moving those calls behind the guard is the other half of this step, not an optional tidy-up.

Every one of these `fixture_data()` bodies calls synthetic functions **the example already has**, because Contract 4's offline fallback shape means the deterministic substitute is already written. `fixture_data()` is a two-or-three-line function that assembles the payload from them; it is not a second implementation.

**Worked example — `examples/animate_weather_decades.py`.** Structure only; every body is the existing code, moved:

```python
"""...existing docstring, plus one new paragraph, "Shape of this file"..."""
# imports; CACHE; START/END; CITIES; FEATS          -- unchanged


class Weather(NamedTuple):
    monthly: list
    daily: list
    hemispheres: list
    source: str


# --- the data half: the ONLY code here that reaches the network -------------
def fetch_city_months(name, lat, lon): ...          # unchanged
def fetch_city_daily_temp(name): ...                # unchanged, moved up beside its sibling
def synthetic_city_months(hemi, ...): ...           # unchanged
def synthetic_city_daily(hemi, n_days, ...): ...    # unchanged


def load_weather(cities=CITIES):
    """The two existing fetch loops, now named."""
    ...


def fixture_data():
    """The same payload from the seeded synthetic path. No network, no
    committed bytes -- this is what the Task 8 gate drives."""
    hemis = [hemi for _n, _lat, _lon, hemi in cities_spec()]
    return Weather([synthetic_city_months(h) for h in hemis],
                   [synthetic_city_daily(h, N_DAYS) for h in hemis],
                   hemis, 'synthetic (fixture)')


# --- the figure half: no network, no I/O, deterministic given its input -----
def construct_artifact(data):
    """Everything from `min_len = ...` to `anim.on_frame(decorate)`, verbatim,
    indented one level, reading `data.monthly` / `data.daily` /
    `data.hemispheres` instead of module globals. RETURNS the HyperAnimation
    (Contract 8: return the wrapper, never the unpacked pair)."""
    ...
    return anim


if __name__ == '__main__':
    weather = load_weather()
    print(f'weather: {len(CITIES)} cities ({weather.source})')
    anim = construct_artifact(weather)
    fig = anim.figure
```

**Verified, not assumed:**

- **Readability is preserved.** All five examples contain **zero** sphinx-gallery narration blocks (`grep -c "^# %%\|^####"` → 0 for each), so sphinx-gallery already renders each as one docstring plus one code block; the split cannot fragment interleaved narration because there is none. The reader gains two labelled halves in place of a 336-line straight line whose two fetch loops sit 125 lines apart.
- **sphinx-gallery still runs the guarded driver.** It executes each example inside a *fake `__main__` module* (`sphinx_gallery/gen_rst.py:1271-1280`), so `if __name__ == '__main__':` fires at docs build. Confirmed end-to-end on the split file: `weather: 6 cities (open-meteo archive)`, `EXIT=0`.

- [ ] **Step 0c: Renegotiate the budgets the split costs**

The split is not free, and Contract 6 says a budget is renegotiated **in the plan**, never weakened in the test. Measured on weather: **+15 code lines** (195 → 210), being the `NamedTuple` (6), two `def` lines, `load_weather`'s scaffolding, and the 4-line `__main__` guard. That is ~24% of its 62-line budget.

So each of Tasks 2–6 raises its script budget by its own measured split overhead, and the notebook budget follows automatically (Contract 6b). **Measure it, do not copy weather's 15** — the overhead depends on how many loaders an example has and how big its payload is. Weather's is the worked figure; paintings has two fetch sites and morph has one, so theirs will differ.

Record each measured overhead in `SCRIPT_BUDGETS` with a one-line comment naming it, exactly as the conversation entry already does.

- [ ] **Step 1: Commit the measurement**

```python
# scripts/measure_native_ratio.py
r"""Measure how much of an example or tutorial is a hypertools call.

Definitions (these are the contract Task 8 of the 1.1 examples plan gates on):

CODE line    -- non-blank, not comment-only, not part of a bare docstring.
LOGICAL stmt -- consecutive code lines joined while bracket depth > 0, or
                while a line ends in a backslash. A continuation line belongs
                to the statement it continues, so a 10-line ``hyp.plot(...)``
                call counts as 10 native lines rather than 1. This is the
                whole point: the metric must reward a big native call.
NATIVE       -- every code line of a logical statement whose text matches
                ``\bhyp\.|\bhypertools\b``.

Measured against the 2026-07-26 audit's independent NATIVE-line
classification, this metric gives 48/739 = 6.5% for the five launch scripts
where the audit reported 6.0% -- i.e. the two agree.

    .venv/bin/python scripts/measure_native_ratio.py examples/animate_*.py
    .venv/bin/python scripts/measure_native_ratio.py docs/tutorials/*.ipynb
"""

import ast
import json
import re
import sys

HYP = re.compile(r'\bhyp\.|\bhypertools\b')


def _docstring_line_numbers(source):
    """1-based line numbers occupied by REAL docstrings.

    `ast` is what makes this correct, and a heuristic cannot be. A docstring
    is the FIRST statement of a module/class/function and is a bare string
    expression. A line-scanner that keys on "the stripped line starts with a
    triple quote" cannot tell that from the CLOSING quote of an ordinary
    multi-line string -- it flips into docstring mode there and silently
    drops everything after it.

    That is not hypothetical. The first version of this function did exactly
    that, and measured against the real repo it dropped 171 code lines from
    `tests/test_density.py`, 123 from `tests/test_backend_state_safety.py`
    and 121 from `tests/test_surface.py` -- 8 files in all. Every dropped
    line is invisible to BOTH the size budget and the defect-marker ban, so
    a private reach sitting after an ordinary multi-line string would have
    passed the gate. A scan that silently drops code is worse than no scan,
    because it reports green.
    """
    # IPython magics and shell escapes are not Python; comment them out so a
    # notebook cell still parses. Line numbering is preserved.
    prepared = '\n'.join(
        ('# ' + line) if line.lstrip()[:1] in ('%', '!') else line
        for line in source.split('\n'))
    try:
        tree = ast.parse(prepared)
    except SyntaxError:
        # Unparseable: KEEP EVERY LINE. A spurious marker hit fails loudly
        # and gets investigated; a silently dropped line hides a defect for
        # good. When in doubt, keep the line.
        return set()
    drop = set()
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Module, ast.FunctionDef,
                                 ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        body = getattr(node, 'body', None)
        if not body:
            continue
        first = body[0]
        if isinstance(first, ast.Expr) \
                and isinstance(first.value, ast.Constant) \
                and isinstance(first.value.value, str):
            drop.update(range(first.lineno,
                              (first.end_lineno or first.lineno) + 1))
    return drop


def strip_docstrings(lines):
    """Yield the CODE lines from an iterable of source lines.

    Drops blank lines, comment-only lines, and real docstrings. This is the
    ONE place that logic lives -- shared by both counters below AND by
    `tests/test_examples_are_native.py`'s `_code_text`, so none of the three
    can drift out of sync.

    Public (no leading underscore) precisely because the test module imports
    it. Before this was shared, `_code_lines_py` and `_code_lines_nb` carried
    two INDEPENDENT copies and the notebook one was never written, so
    identical source measured (code=3, native=2) as `.py` but
    (code=11, native=2) as `.ipynb`.
    """
    lines = list(lines)
    drop = _docstring_line_numbers('\n'.join(lines))
    for n, line in enumerate(lines, 1):
        if n in drop:
            continue
        stripped = line.strip()
        if not stripped or stripped.startswith('#'):
            continue
        yield line


def _code_lines_py(path):
    return list(strip_docstrings(
        open(path, encoding='utf-8').read().splitlines()))


def _code_lines_nb(path):
    out = []
    for cell in json.load(open(path, encoding='utf-8'))['cells']:
        if cell.get('cell_type') != 'code':
            continue
        # Reset per cell: a bare docstring cannot span a cell boundary (each
        # cell is parsed and executed independently), so carrying in_doc /
        # delim across cells would be wrong, not merely unnecessary.
        out.extend(strip_docstrings(
            line.rstrip('\n') for line in cell['source']))
    return out


def _depth_delta(line):
    depth, quote, i = 0, None, 0
    while i < len(line):
        ch = line[i]
        if quote:
            if ch == '\\':
                i += 2
                continue
            if ch == quote:
                quote = None
        elif ch in '"\'':
            quote = ch
        elif ch == '#':
            break
        elif ch in '([{':
            depth += 1
        elif ch in ')]}':
            depth -= 1
        i += 1
    return depth


def measure(path):
    """Return ``(code_lines, native_lines)`` for one .py or .ipynb file."""
    lines = _code_lines_nb(path) if str(path).endswith('.ipynb') \
        else _code_lines_py(path)
    statements, current, depth = [], [], 0
    for line in lines:
        current.append(line)
        depth += _depth_delta(line)
        if depth <= 0 and not line.rstrip().endswith('\\'):
            statements.append(current)
            current, depth = [], 0
    if current:
        statements.append(current)
    total = sum(len(s) for s in statements)
    native = sum(len(s) for s in statements
                 if HYP.search('\n'.join(s)))
    return total, native


if __name__ == '__main__':
    for target in sys.argv[1:]:
        code, native = measure(target)
        pct = 100.0 * native / code if code else 0.0
        print(f'{target:56s} code={code:4d} native={native:4d} '
              f'ratio={pct:5.1f}%')
```

Verify it reproduces the recorded baseline **before** any rewrite is measured against it:

```bash
git stash && .venv/bin/python scripts/measure_native_ratio.py examples/animate_conversation.py && git stash pop
```
Expected on the untouched file: `code= 166 native=   9 ratio=  5.4%`.

- [ ] **Step 2: Write the gate as a real test**

```python
# tests/test_examples_are_native.py
"""The gallery examples and their notebooks must SHOWCASE hypertools.

Measured on 2026-07-26/28, before the 1.1 examples plan: 48 of 739 code
lines across the five launch examples belonged to a hypertools call (6.5%),
and 37.9% of the code either re-implemented something native or worked
around a gap. This module makes the fix permanent -- it fails if a defect
marker comes back, or if a file drifts back above its size budget.

**The native-code ratio is REPORTED, not gated.** v1 of this plan asserted a
per-file minimum ratio and picked the floors before the rewrites existed;
measured against the plan's own proposed code, four of the five missed their
own floors (market 14.7% vs 26, paintings 12.5% vs 20, conversation 18.9% vs
25, morph 22.2% vs 26), so the gate could not have gone green no matter how
good the rewrite was. Raising the floors to whatever the code happens to
measure would make the gate tautological, and the ratio is trivially gamed
in the wrong direction anyway -- splitting one `hyp.plot(...)` call across
six lines "improves" it, and so does deleting a comment. What the ratio is
genuinely good for is watching a trend, so this module PRINTS it and asserts
only things that cannot be satisfied by reformatting:

1. no private API or named defect pattern (`DEFECT_MARKERS`);
2. a maximum code-line budget per file;
3. executable semantic checks -- the example actually produces the artifact
   it claims (`test_examples_produce_their_stated_artifact`);
4. exact notebook execution success (Step 2c).

No network, no mocks: it reads the committed files.
"""
import ast
import os
import re

import numpy as np
import pytest

from scripts.measure_native_ratio import measure

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

#: A notebook holds its script's code, plus a Colab install cell, plus a
#: display cell.
#:
#: MEASURED: the largest install cell across the five is 3 code lines
#: (paintings and conversation are 3, the other three are 2).
#: NOT MEASURED -- a design decision: the 2-line display cell
#: (`from IPython.display import HTML` + `HTML(ani.to_jshtml())`). No
#: current notebook has one (`grep -l "to_jshtml\|IPython.display"` over
#: the five returns nothing), so this is a budget for a cell that does not
#: exist yet, and it is LOAD-BEARING: conversation has only 2 lines of
#: headroom. If the display cell turns out to be 3 lines, or a second one
#: is added, raise NOTEBOOK_OVERHEAD here -- in the plan -- rather than
#: letting a task quietly exceed its budget.
#:
#: This is why the notebook budgets are DERIVED rather than written down. v2
#: wrote them down and set two of them BELOW their own script's -- paintings
#: 110 against a script of 118, conversation 76 against 90 -- which no
#: correct notebook can satisfy, whatever the metric does. Deriving makes
#: that class of mistake impossible and means only ONE number per example is
#: ever chosen by hand.
NOTEBOOK_OVERHEAD = 5

#: script path -> max code lines. Measured against the code this plan
#: actually prescribes (see each task's AFTER line), never guessed ahead of
#: it: market 109, weather 56, paintings 111, conversation 88, morph 26.
#: EVERY figure here includes the Step 0b loader/builder split overhead,
#: because the split is part of what each task delivers. Measured on
#: weather: **+15** code lines (the NamedTuple 6, two def lines,
#: load_weather's scaffolding, and the 4-line __main__ guard).
#:
#: Weather's is measured. The other four are weather's +15 carried across as
#: a PLACEHOLDER, and Step 0c replaces each with that file's own measured
#: overhead -- an example with two fetch sites (paintings) or one
#: (morph) will not cost the same as weather's two. Until a task measures
#: its own, `test_file_is_within_its_size_budget` may fail for that file,
#: and that failure is the instruction, exactly as with
#: EXPECTED_VISIBLE_OUTPUTS. Do NOT satisfy it by trimming the split.
SCRIPT_BUDGETS = {
    'examples/animate_market_forecast.py': 130,   # 115 + 15 (placeholder)
    'examples/animate_weather_decades.py': 77,    # 62 + 15 (MEASURED)
    'examples/animate_painting_embeddings.py': 133,  # 118 + 15 (placeholder)
    # 90, not v2's prose figure of 72: the prescribed rewrite measures 88
    # code lines (87 at best, with `turn_alpha` inlined -- which was
    # deliberately split OUT to fix the recency_fade Fatal).
    'examples/animate_conversation.py': 105,      # 90 + 15 (placeholder)
    'examples/animate_morph_zoo.py': 45,          # 30 + 15 (placeholder)
}

#: script stem -> notebook, so the derivation below has something to pair.
NOTEBOOKS = {
    'examples/animate_market_forecast.py': 'docs/tutorials/market_forecast.ipynb',
    'examples/animate_weather_decades.py': 'docs/tutorials/weather_decades.ipynb',
    'examples/animate_painting_embeddings.py': 'docs/tutorials/painting_embeddings.ipynb',
    'examples/animate_conversation.py': 'docs/tutorials/conversation_shape.ipynb',
    'examples/animate_morph_zoo.py': 'docs/tutorials/morph_shapes_zoo.ipynb',
}

#: (path, max_code_lines) for every gated file -- scripts as chosen,
#: notebooks as derived.
BUDGETS = ([(p, n) for p, n in SCRIPT_BUDGETS.items()]
           + [(NOTEBOOKS[p], n + NOTEBOOK_OVERHEAD)
              for p, n in SCRIPT_BUDGETS.items()])


def test_notebook_budgets_are_derived_not_written_down():
    """The v2 defect, pinned so it cannot return.

    Asserts the DERIVATION is still in force -- each notebook limit equals
    its script's plus exactly `NOTEBOOK_OVERHEAD` -- not merely that it is
    larger. `>= limits[script]` would be `n + 5 >= n`, true for every `n`,
    a comment wearing a test's clothes and the same inert-assertion defect
    this plan has now hit twice (`_save_count >= 1`, `'morph' in 'morph'`).

    Equality CAN fail, and fails on the thing actually worth catching:
    someone replacing the comprehension with hand-written numbers, which is
    how paintings ended up at 110 against a script of 118.
    """
    limits = dict(BUDGETS)
    for script, nb in NOTEBOOKS.items():
        assert limits[nb] == limits[script] + NOTEBOOK_OVERHEAD, (
            f'{nb} is budgeted at {limits[nb]}, but the derivation says '
            f'{limits[script]} + {NOTEBOOK_OVERHEAD} = '
            f'{limits[script] + NOTEBOOK_OVERHEAD}. Change the SCRIPT budget '
            f'and let the notebook follow; do not hand-write this one.')


#: Private reaches that are DELIBERATELY retained, with the reason. Contract
#: 3 bans private API only where a public equivalent exists; these two have
#: none, are one-time setup rather than per-frame work, and each carries an
#: inline rationale in the source. Anything NOT listed here still fails, so
#: a new reach cannot creep in, and each of these was reviewed rather than
#: assumed. Landed in `d730a085` with measurements.
PRIVATE_API_EXCEPTIONS = {
    ('examples/animate_market_forecast.py', r'ani\._args'):
        'one-time readback of the fully-revealed ANTIALIASED on-screen line; '
        'ctx.datasets is the pre-antialiasing array at a coarser resolution '
        'and fits a measurably different slope (~2-8%, checked empirically)',
    ('examples/animate_market_forecast.py', r'hypertools\._shared'):
        'PCHIP smoothing has no public re-export; reimplementing it here '
        'would risk drifting from what hyp.plot actually draws',
}

#: Every one of these was found in the launch examples or the older
#: tutorials and removed. Each maps to the native API that replaced it.
DEFECT_MARKERS = {
    r'\bSentenceTransformer\b': "use vectorizer='<hf-model-id>', semantic=None, corpus=None",
    r'ani\._func': 'use on_frame= / HyperAnimation.on_frame()',
    r'ani\._args': 'use the FrameContext passed to on_frame=',
    r'hypertools\._shared': 'private module; use a documented kwarg',
    r'from hypertools\.plot import morph': "use title=[...] for per-segment names",
    r'\bantialias_line\b': 'plot() antialiases every drawn line already',
    r'\bffmpeg\b': "save_path='*.gif' needs no ffmpeg (plot.py:1513-1520)",
    r'morph_schedule|frame_to_segment': 'the morph schedule is the library\'s business',
}

def _read(path):
    full = os.path.join(REPO, path)
    with open(full, encoding='utf-8') as handle:
        return handle.read()


def _code_text(path):
    """Code only -- and DOCSTRINGS ARE NOT CODE here.

    Two reasons, both load-bearing:

    1. Markdown/prose may still discuss a removed workaround, so notebook
       markdown cells are excluded.
    2. `d730a085` documented each migration by NAMING the pattern it
       removed -- `animate_weather_decades.py` and `animate_conversation.py`
       both contain the string ``ani._func`` inside a docstring explaining
       that the monkeypatch is gone. Scanning raw source would fail those
       files for their own documentation.

    This shares `strip_docstrings` with `scripts/measure_native_ratio.py`
    rather than re-implementing it: the two counters previously disagreed
    (one stripped, one did not, so identical source measured differently as
    .py and .ipynb), and a shared callee cannot drift from itself.
    """
    from scripts.measure_native_ratio import strip_docstrings
    if path.endswith('.ipynb'):
        import json
        nb = json.loads(_read(path))
        # PER CELL, exactly as `_code_lines_nb` does -- not concatenated
        # first. Concatenating makes the two disagree on the same file: a
        # notebook whose first cell holds an unclosed bare `"""` note
        # measured 5 code lines under the budget test while this function
        # returned '' and the defect-marker ban passed unconditionally on an
        # empty string. That is the F2 defect class -- two counters
        # disagreeing on identical input -- relocated into the code written
        # to eliminate it.
        kept = []
        for cell in nb['cells']:
            if cell.get('cell_type') != 'code':
                continue
            kept.extend(strip_docstrings(
                ''.join(cell['source']).split('\n')))
        return '\n'.join(kept)
    return '\n'.join(strip_docstrings(_read(path).split('\n')))


def test_a_docstring_naming_a_removed_pattern_is_not_a_defect():
    """Pins the above. `d730a085` explains each migration by naming what it
    removed; that is documentation, not a reach. Red before the docstring
    strip: weather and conversation both failed the marker scan for their
    own prose."""
    for path in ('examples/animate_weather_decades.py',
                 'examples/animate_conversation.py'):
        assert 'ani._func' in _read(path), (
            f'{path}: expected the migration docstring to still name the '
            f'pattern it replaced')
        assert 'ani._func' not in _code_text(path), (
            f'{path}: the docstring mention leaked into the scanned code')


def _docstring_lines(path):
    """1-based line numbers occupied by docstrings in a .py file.

    Used to tell a real private reach from a docstring that merely NAMES
    one while explaining why it was removed (or why it has to stay).
    """
    if not path.endswith('.py'):
        return set()
    try:
        tree = ast.parse(_read(path))
    except SyntaxError:
        return set()
    spans = set()
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Module, ast.FunctionDef,
                                 ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        body = getattr(node, 'body', None)
        if not body:
            continue
        first = body[0]
        if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) \
                and isinstance(first.value.value, str):
            spans.update(range(first.lineno, (first.end_lineno or first.lineno) + 1))
    return spans


def _parsable_code(path):
    """Code text that `ast.parse` will accept.

    Notebook cells legitimately contain IPython magics (`%pip install`,
    `%matplotlib inline`) and shell escapes (`!cmd`), which are not Python
    and raise SyntaxError. Commenting them out preserves line numbering,
    which keeps any reported position meaningful.
    """
    return '\n'.join(('# ' + line) if line.lstrip()[:1] in ('%', '!') else line
                      for line in _code_text(path).split('\n'))


@pytest.mark.parametrize('path,max_code', BUDGETS)
def test_file_is_within_its_size_budget(path, max_code):
    code, _native = measure(os.path.join(REPO, path))
    assert code <= max_code, (
        f'{path}: {code} code lines exceeds the {max_code}-line budget')


def test_native_ratio_is_reported(capsys):
    """REPORTED, not gated -- see the module docstring. Fails only if a file
    is missing or unparseable, so the number can never be met by
    reformatting. Read it with `pytest -s` or in the CI log."""
    rows = []
    for path, _max_code in BUDGETS:
        full = os.path.join(REPO, path)
        # `measure()` raises FileNotFoundError on a moved/renamed file long
        # before any assert here could report it, so check existence first
        # -- otherwise the "moved or renamed?" message is unreachable.
        assert os.path.exists(full), f'{path}: moved or renamed?'
        code, native = measure(full)
        assert code > 0, f'{path}: parsed to zero code lines'
        rows.append((path, code, native, 100.0 * native / code))
    with capsys.disabled():
        print('\nnative-code ratio (reported, not gated):')
        for path, code, native, ratio in rows:
            print(f'  {ratio:5.1f}%  {native:3d}/{code:3d}  {path}')


@pytest.mark.parametrize('path,_max', BUDGETS)
@pytest.mark.parametrize('marker,fix', sorted(DEFECT_MARKERS.items()))
def test_no_defect_marker_in_the_launch_examples(path, _max, marker, fix):
    if (path, marker) in PRIVATE_API_EXCEPTIONS:
        pytest.skip(f'allowlisted: {PRIVATE_API_EXCEPTIONS[(path, marker)]}')
    text = _code_text(path)
    assert not re.search(marker, text), (
        f'{path} contains {marker!r} again -- {fix}')


#: How far from an allowlisted private reach its rationale may sit. 15 lines
#: is the size of a comment block plus the statement it explains -- close
#: enough that a reader who lands on the reach sees the reason without
#: scrolling.
RATIONALE_WINDOW = 15


def test_every_allowlisted_reach_is_still_present_and_still_explained():
    """An allowlist entry that no longer matches anything is dead weight --
    it would silently permit a pattern nobody uses. And an allowlisted reach
    with no inline rationale is exactly the 'private API taught as normal'
    that Contract 3 exists to prevent.

    The rationale must sit WITHIN `RATIONALE_WINDOW` lines of the reach, not
    merely somewhere in the file. An earlier version searched the whole file
    for the words 'deliberately' or 'no public', which a 380-line example
    satisfies by accident -- it would have passed even if the explanation
    were 200 lines from the code it explains, or explained something else
    entirely.
    """
    for (path, marker), reason in PRIVATE_API_EXCEPTIONS.items():
        lines = _read(path).split('\n')
        # Skip matches inside docstrings: `animate_market_forecast.py`'s
        # module docstring explains the reach by NAMING it
        # (`ani._args[1][0]`, in the "Coordinate note" paragraph), and that
        # prose is documentation, not a second reach. Same reason
        # `_code_text` strips docstrings before the marker scan.
        doc_lines = _docstring_lines(path)
        hits = [i for i, line in enumerate(lines)
                if re.search(marker, line) and (i + 1) not in doc_lines]
        assert hits, (
            f'{path} no longer contains {marker!r}; drop the '
            f'PRIVATE_API_EXCEPTIONS entry rather than leaving it to permit '
            f'a pattern that is gone')
        for i in hits:
            window = '\n'.join(lines[max(0, i - RATIONALE_WINDOW):
                                     i + RATIONALE_WINDOW])
            explained = ('deliberately' in window or 'no public' in window
                         or 'no publicly' in window)
            assert explained, (
                f'{path}:{i + 1} reaches {marker!r} with no rationale within '
                f'{RATIONALE_WINDOW} lines. Contract 3 allowlists it only '
                f'because the source explains itself where a reader will '
                f'find it (reason on record: {reason})')


#: Members that live ONLY on the `HyperAnimation` wrapper. Unpacking or
#: indexing a plot result throws the wrapper away, so reaching any of these
#: on a name that came out of a tuple is an `AttributeError`.
#:
#: `figure`/`animation` are included because they are wrapper properties
#: too, and `n_frames`/`n_segments`/`draw_frame` because Step 0 ADDS them --
#: a guard that greps only for `.on_frame(` would widen the trap and leave
#: the check where it was.
WRAPPER_ONLY = ('on_frame', 'n_frames', 'n_segments', 'draw_frame',
                'figure', 'animation')


#: Properties that hand back the RAW FuncAnimation, discarding the wrapper.
UNWRAPPING_ATTRS = ('animation',)


def _hypertools_names(tree):
    """(module aliases, bare names) in this file that refer to hypertools.

    Learned from the file's OWN imports. An earlier version matched any
    attribute call named `plot`, which made matplotlib's `ax.plot` and
    pandas' `df.plot` collateral -- and `Line2D.figure`/`Axes.figure` are
    real public attributes, so `WRAPPER_ONLY` containing `figure` turned
    them into false positives with a factually wrong message. The trigger
    was already present in a gated file:
    `examples/animate_market_forecast.py` does
    `fc_line, = ax.plot([], [], [], '--', ...)`, so `fc_line` already
    entered the unpacked set, and the test passed only because nothing
    happened to read `fc_line.figure`.
    """
    mods, bare = set(), set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split('.')[0] == 'hypertools':
                    mods.add(alias.asname or alias.name.split('.')[0])
        elif isinstance(node, ast.ImportFrom):
            if (node.module or '').split('.')[0] == 'hypertools':
                for alias in node.names:
                    if alias.name == 'plot':
                        bare.add(alias.asname or alias.name)
    return mods, bare


def _unpacked_wrapper_uses(source):
    """[(name, attr), ...] for names holding an UNPACKED plot result that
    then reach a wrapper-only member.

    Raises `SyntaxError` if `source` will not parse -- the caller must
    handle it. Returning `[]` there would make this silently vacuous on
    exactly the files most likely to be odd, which is the
    assertion-that-cannot-fail class this plan has shipped four revisions
    running.
    """
    tree = ast.parse(source)
    mods, bare = _hypertools_names(tree)

    def is_plot_call(node):
        if not isinstance(node, ast.Call):
            return False
        fn = node.func
        if isinstance(fn, ast.Attribute) and fn.attr == 'plot':
            return isinstance(fn.value, ast.Name) and fn.value.id in mods
        return isinstance(fn, ast.Name) and fn.id in bare

    wrappers, unpacked = set(), set()

    def unwraps(value):
        """`hyp.plot(...).animation` or `<wrapper>.animation` -- the
        documented property that hands back the raw FuncAnimation, and so
        the most plausible form of this bug after direct unpacking."""
        if isinstance(value, ast.Attribute) and value.attr in UNWRAPPING_ATTRS:
            base = value.value
            return is_plot_call(base) or (isinstance(base, ast.Name)
                                          and base.id in wrappers)
        return False

    for _ in range(3):          # propagate `b = a` aliases to a fixed point
        for node in ast.walk(tree):
            if not isinstance(node, ast.Assign):
                continue
            value = node.value
            for target in node.targets:
                if isinstance(target, ast.Name):
                    if is_plot_call(value):
                        wrappers.add(target.id)
                    elif isinstance(value, ast.Name) and value.id in wrappers:
                        wrappers.add(target.id)
                    elif unwraps(value):
                        unpacked.add(target.id)
                    elif isinstance(value, ast.Subscript):
                        base = value.value
                        if (isinstance(base, ast.Name) and base.id in wrappers) \
                                or is_plot_call(base):
                            unpacked.add(target.id)
                elif isinstance(target, (ast.Tuple, ast.List)):
                    if is_plot_call(value) or (isinstance(value, ast.Name)
                                               and value.id in wrappers):
                        unpacked.update(e.id for e in target.elts
                                        if isinstance(e, ast.Name))
    return sorted({(n.value.id, n.attr) for n in ast.walk(tree)
                   if isinstance(n, ast.Attribute)
                   and n.attr in WRAPPER_ONLY
                   and isinstance(n.value, ast.Name)
                   and n.value.id in unpacked})


@pytest.mark.parametrize('path,_max', BUDGETS)
def test_no_example_or_notebook_unpacks_then_uses_the_wrapper(path, _max):
    """Contract 8. `fig, ani = hyp.plot(...)` binds `ani` to the raw
    FuncAnimation, so every wrapper member raises AttributeError -- while
    `_save_count` SURVIVES the unpack, so a gate written against that
    attribute passes as the public API is discarded.

    PASSES on all ten files today: it is a CONTROL against regression, not
    coverage of a present defect. It would have caught v2's prescribed
    conversation notebook, which unpacked and then called `.on_frame()`.
    """
    try:
        hits = _unpacked_wrapper_uses(_parsable_code(path))
    except SyntaxError as exc:
        pytest.fail(
            f'{path}: could not be parsed ({exc}), so this guard would be '
            f'silently vacuous on it. `_parsable_code` comments out magics '
            f'that start a line; a cell magic (%%bash), an INDENTED magic '
            f'inside a block, or a `hyp.plot?` help suffix still defeats it. '
            f'Fix the notebook or extend `_parsable_code` -- do not let the '
            f'file through unchecked.')
    assert not hits, (
        f'{path}: ' + '; '.join(
            f'`{name}` comes from unpacking a hyp.plot() result, so it is a '
            f'raw FuncAnimation and has no .{attr}' for name, attr in hits)
        + '. Bind the HyperAnimation first (`anim = hyp.plot(...)`), then '
          '`fig, ani = anim` if the parts are wanted.')


#: Measured 2026-08-02 against the committed notebooks, so a reader can
#: tell coverage from controls at a glance. RED today (Task 7 turns them
#: green): conversation_trajectories, hugging_face_embeddings and
#: wikipedia_embeddings fail BOTH assertions; modern_sklearn_dynamics fails
#: on ffmpeg only. ALREADY GREEN, and therefore CONTROLS rather than
#: coverage: stock_forecasting and projectile_kalman -- they are here to
#: prove Task 7 does not REGRESS a clean notebook, not to prove it fixed
#: one. Do not read six passing IDs as six notebooks repaired.
@pytest.mark.parametrize('nb', [
    'conversation_trajectories', 'hugging_face_embeddings',
    'wikipedia_embeddings', 'modern_sklearn_dynamics',
    'stock_forecasting',        # control -- already clean
    'projectile_kalman',        # control -- already clean
])
def test_older_tutorials_dropped_their_hand_rolled_helpers(nb):
    text = _code_text(f'docs/tutorials/{nb}.ipynb')
    assert 'SentenceTransformer' not in text
    assert 'ffmpeg' not in text


def test_analyze_tutorial_actually_plots():
    """A pipeline tutorial that never calls hyp.plot never shows why the
    pipeline exists (audit: analyze.ipynb, 20.0% hypertools, 0 hyp.plot)."""
    assert 'hyp.plot' in _code_text('docs/tutorials/analyze.ipynb')


def test_reduce_tutorial_mentions_describe():
    assert 'hyp.describe' in _code_text('docs/tutorials/reduce.ipynb')


#: The artifact each example exists to produce. These are the SEMANTIC
#: gates that replaced the native-ratio floor: unlike a line-count ratio,
#: none of them can be satisfied by reformatting, and each fails loudly if
#: the rewrite drops the thing the example is for.
STATED_ARTIFACT = {
    # min_frames is a real floor per example (frame_rate x duration), not
    # `>= 1`, which every animation satisfies by construction.
    'animate_market_forecast': dict(min_frames=100, predicts=True),
    'animate_weather_decades': dict(min_frames=100, axes=2),
    'animate_painting_embeddings': dict(min_frames=60, palette=True),
    'animate_conversation': dict(min_frames=100, on_frame=True),
    # 5 shapes, plus `clouds.append(clouds[0])` to close the loop = 6
    # clouds -> 2*6 - 1 = 11 segments, matching the example's own
    # 11-entry `rotations` list. (Measured; NOT 10 -- the schedule has no
    # implicit closing transition, and the example's inline comment
    # "for the 5 clouds = 9 segments" counts the shapes, not what it passes.)
    'animate_morph_zoo': dict(min_frames=200, morph=11),
}


def _import_example_without_fetching(stem):
    """Import an example as a module, and prove the import fetched nothing.

    **This depends on Step 0b having been done, and fails loudly if it has
    not.** Measured 2026-08-02: NO example currently has a
    `if __name__ == '__main__':` guard (`grep -c __main__ examples/animate_*.py`
    -> 0 for all ten), so today every loader runs at module scope --
    `animate_morph_zoo.py:74` and `animate_market_forecast.py:113` fetch
    during import. `runpy.run_path` (v2) had the same problem for the same
    reason.

    Step 0b is what makes the premise true: it moves every loader call
    behind the `__main__` guard so the module body only DEFINES things, and
    it makes each fetcher honour `HYPERTOOLS_OFFLINE` by raising instead of
    silently substituting. Until then this helper is not merely ineffective
    -- it would download Dropbox shape files, FRED CSVs and HuggingFace
    models inside the default suite.

    The guard below turns that from a silent regression into a failure that
    names the file.
    """
    import importlib.util
    import matplotlib
    matplotlib.use('Agg')
    path = os.path.join(REPO, 'examples', f'{stem}.py')
    source = _read(f'examples/{stem}.py')
    # Refuse to import an example that has not been split yet, rather than
    # letting it fetch. Checked BEFORE exec, because after exec the damage
    # is done.
    assert "__name__ == '__main__'" in source, (
        f'examples/{stem}.py has no __main__ guard, so importing it would '
        f'run its loaders and hit the network (Step 0b). Do the loader / '
        f'construct_artifact split before enabling this gate.')
    os.environ['HYPERTOOLS_OFFLINE'] = '1'
    try:
        spec = importlib.util.spec_from_file_location(stem, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        for required in ('construct_artifact', 'fixture_data'):
            assert hasattr(module, required), (
                f'examples/{stem}.py does not define {required}() (Step 0b)')
        return module
    finally:
        os.environ.pop('HYPERTOOLS_OFFLINE', None)


def _drive(anim, frame):
    """Render one frame, by index, through the public animation object.

    `HyperAnimation.draw_frame(i)` (Task 8 Step 0) is the supported way to
    do this. v2 reached for matplotlib's private `ani._func(i, *ani._args)`,
    which Contract 3 forbids and `DEFECT_MARKERS` lists.
    """
    anim.draw_frame(frame)


@pytest.mark.parametrize('stem', sorted(STATED_ARTIFACT))
def test_examples_produce_their_stated_artifact(stem):
    """Executable semantics, not source-shape -- driven by a FIXTURE.

    v2 ran each example with `runpy`. Measured: **all five are
    network-coupled** (weather 6 blocked connections, paintings 7, morph 4,
    conversation 2, market 1), so that version put model downloads and
    remote fetches in the default suite, contradicting Contract 4 and making
    CI nondeterministic. Morph does not even degrade -- `hyp.load()` has no
    offline path and takes the example down with `HypertoolsIOError`.

    So each example splits its loader from its figure builder (Contract 4),
    and this drives `construct_artifact(data)` with the example's own seeded
    synthetic data. Four of the five need ZERO committed fixture bytes,
    because their existing offline fallbacks already ARE deterministic
    fixtures; paintings ships one 1.7 KB thumbnail. Importing an example
    must not fetch. The whole-example run survives as an opt-in smoke test
    (`HYPERTOOLS_EXAMPLE_SMOKE=1`), never in the default suite.

    Every assertion below had to be rewritten, because v2's could not fail:

    * `_save_count >= 1` is a TAUTOLOGY -- hyp.plot always passes
      `max(1, round(frame_rate * duration))`, so it holds for a zero-length,
      zero-dataset animation. Measured at duration=0.01/frame_rate=1: 1.
      It is also a private matplotlib field, which Contract 3 forbids and
      `DEFECT_MARKERS` lists ten lines above the gate that used it.
    * `'morph' in str(ns.get('ANIMATE', 'morph'))` is a TAUTOLOGY -- no
      example binds `ANIMATE`, so the default makes it `'morph' in 'morph'`.
    * `ns['ani']` does not exist in weather or conversation, which bind
      `anim` (Contract 8), so 2 of 5 parametrisations failed on day one for
      a reason unrelated to what they gate.

    An unsatisfiable gate and a vacuous one are the same defect wearing
    opposite clothes: neither discriminates. v1 shipped the first, v2
    replaced it with the second.
    """
    module = _import_example_without_fetching(stem)
    want = STATED_ARTIFACT[stem]
    anim = module.construct_artifact(module.fixture_data())
    fig = anim.figure

    assert anim.n_frames >= want['min_frames'], (
        f'{stem}: {anim.n_frames} frames, expected at least '
        f"{want['min_frames']}")
    if want.get('axes'):
        assert len(fig.axes) >= want['axes']
    if want.get('predicts'):
        _drive(anim, frame=anim.n_frames - 1)
        live = [a for a in fig.axes[0].lines
                if getattr(a, '_hyp_forecast_role', None) == 'live']
        assert live, 'no live forecast artist after driving a frame'
        pts = np.asarray(live[0].get_data_3d()
                         if hasattr(live[0], 'get_data_3d')
                         else live[0].get_data())
        assert pts.size and np.isfinite(pts).all(), (
            'the forecast artist exists but its geometry is empty or '
            'non-finite -- artists can be created and never filled')
    if want.get('on_frame'):
        _drive(anim, frame=0)
        first = [a.get_alpha() for a in fig.axes[0].lines]
        _drive(anim, frame=anim.n_frames - 1)
        last = [a.get_alpha() for a in fig.axes[0].lines]
        assert first != last, (
            'the per-frame hook never changed any alpha, so the recency '
            'fade is not actually running')
    if want.get('morph'):
        assert anim.n_segments == want['morph'], (
            f'{stem}: {anim.n_segments} morph segments, expected '
            f"{want['morph']}")
        _drive(anim, frame=anim.n_frames // 2)
        assert fig.axes[0].collections or fig.axes[0].lines, (
            'a driven mid-morph frame drew nothing')


LAUNCH_NOTEBOOKS = ('market_forecast', 'weather_decades',
                    'painting_embeddings', 'conversation_shape',
                    'morph_shapes_zoo')


def _is_install_cell(source):
    """Detected by CONTENT, never by index.

    Two measurements forced this. First, the install cell is NOT uniformly
    unexecuted: 9 of the 20 notebooks in docs/tutorials/ ship it executed,
    the five launch notebooks do not -- so a gate asserting either polarity
    fails on half the repo, and it must simply be EXEMPT. Second, indexing
    by position breaks the moment a cell is inserted above it, or a notebook
    has no install cell at all.
    """
    return 'pip install' in source


def _code_cells(stem):
    import json
    nb = json.loads(_read(f'docs/tutorials/{stem}.ipynb'))
    return [c for c in nb['cells'] if c.get('cell_type') == 'code']


#: Per notebook, the INDEX SET of code cells that carry a visible output --
#: not a count. Recorded from a real nbclient run in each task's "Execute
#: and measure" step; do not write a number here before the notebook exists.
#:
#: Why an index set and not a total: v2 hardcoded five counts and ALL FIVE
#: were wrong, as was every per-task prediction in Tasks 2-6, because each
#: assumed every non-install cell emits when several are bare imports, bare
#: assignments, or `fig, ani = hyp.plot(..., show=False)`. Weather is the
#: instructive one -- its TOTAL happened to be right while naming entirely
#: the wrong cells. A count cannot tell those apart and is satisfied by a
#: stray print() landing anywhere; an index set fails immediately and names
#: the cell.
#:
#: Install-cell indices are filtered out of both sides before comparing.
EXPECTED_VISIBLE_OUTPUTS = {
    # 'market_forecast': {2, 5, 6, 7},   <- fill in from the measured run
}


@pytest.mark.parametrize('stem', LAUNCH_NOTEBOOKS)
def test_every_launch_notebook_ran_every_cell_it_should(stem):
    """`nbsphinx_execute = 'never'` (docs/conf.py:131) renders the COMMITTED
    outputs, so a half-executed notebook is a figure-less docs page.

    Gates EXECUTION, which is a different property from OUTPUT: a cell can
    run perfectly and legitimately emit nothing. v1 allowed `len(code) - 2`
    unexecuted cells, which would pass a notebook whose only two code cells
    both failed; v2 demanded every code cell carry output, which no notebook
    can satisfy. This asserts what is actually required -- every cell ran --
    and leaves what each cell EMITS to the index-set test below.
    """
    cells = _code_cells(stem)
    unrun = [i for i, c in enumerate(cells)
             if c.get('execution_count') is None
             and not _is_install_cell(''.join(c['source']))]
    assert not unrun, (
        f'{stem}.ipynb: code cells {unrun} were never executed; re-run '
        f'scripts/execute_tutorial.py')


@pytest.mark.parametrize('stem', LAUNCH_NOTEBOOKS)
def test_the_right_cells_carry_visible_output(stem):
    """Which cells emit, not how many."""
    if stem not in EXPECTED_VISIBLE_OUTPUTS:
        pytest.fail(
            f'{stem}: no measured index set recorded. Execute the notebook '
            f'and paste the measured set into EXPECTED_VISIBLE_OUTPUTS -- '
            f'do not guess it ahead of the artifact (v2 guessed five and got '
            f'all five wrong)')
    cells = _code_cells(stem)
    installs = {i for i, c in enumerate(cells)
                if _is_install_cell(''.join(c['source']))}
    got = {i for i, c in enumerate(cells) if c.get('outputs')} - installs
    want = set(EXPECTED_VISIBLE_OUTPUTS[stem]) - installs
    assert got == want, (
        f'{stem}.ipynb: cells {sorted(got)} carry output, expected '
        f'{sorted(want)} (missing {sorted(want - got)}, unexpected '
        f'{sorted(got - want)})')


@pytest.mark.parametrize('stem', LAUNCH_NOTEBOOKS)
def test_each_notebook_ships_its_rendered_artifact(stem):
    """The artifact assertion, keyed to how these notebooks ACTUALLY ship.

    Measured: there is no `image/png` and no `text/html` output anywhere in
    any of the five -- the display_data entries are tqdm progress widgets
    from sentence_transformers. The convention (commit 9b94d86f), shared
    with conversation_trajectories/streaming_data/wikipedia_embeddings, is a
    companion GIF written by the last code cell and embedded from a MARKDOWN
    cell. So "did a figure render" is not answerable from cell outputs, and
    a rule like "a cell calling hyp.plot must emit something" is satisfied
    by an unrelated print() in the same cell.

    This asserts the artifact that actually exists, and that its reference
    resolves.

    **It PASSES today, on all five -- it is a CONTROL, not coverage.**
    Measured 2026-08-02: every reference resolves, `morph_zoo.gif` included
    (4.5 MB, present). An earlier draft of this plan claimed the test
    "catches morph_shapes_zoo.ipynb embedding morph_zoo.gif"; it does not,
    because that file exists -- the stem mismatch is a naming
    inconsistency, not a broken link. What this test does is stop a rewrite
    from DROPPING the GIF or breaking its reference, which is worth having
    and is why it stays. Do not read five green IDs as five things fixed.
    """
    import json
    import os
    import re as _re
    nb = json.loads(_read(f'docs/tutorials/{stem}.ipynb'))
    md = '\n'.join(''.join(c['source']) for c in nb['cells']
                   if c.get('cell_type') == 'markdown')
    refs = _re.findall(r'!\[[^\]]*\]\(([^)]+\.gif)\)', md)
    assert refs, f'{stem}.ipynb: no rendered artifact is embedded'
    for ref in refs:
        target = os.path.join(REPO, 'docs', 'tutorials', ref)
        assert os.path.exists(target), (
            f'{stem}.ipynb embeds {ref!r}, which does not exist')


@pytest.mark.skipif(not os.environ.get('HYPERTOOLS_EXAMPLE_SMOKE'),
                    reason='set HYPERTOOLS_EXAMPLE_SMOKE=1 to run the '
                           'examples end to end (network + model downloads)')
@pytest.mark.parametrize('stem', sorted(STATED_ARTIFACT))
def test_example_runs_end_to_end(stem):
    """The whole-example run, OPT-IN.

    v2 ran every example in the default suite via `runpy`, which put model
    downloads and remote fetches on every CI run. v3 moved the default gate
    onto `construct_artifact(fixture_data())`, and this is what replaces the
    coverage that removed -- the loaders, the `__main__` guard, and the real
    data path, exercised on demand rather than never.

    Enable with `HYPERTOOLS_EXAMPLE_SMOKE=1 pytest -k end_to_end`. Run it
    before a release and whenever a loader changes; a failure here means the
    example is broken for a user even though the fixture-driven gate is
    green.
    """
    import subprocess
    import sys as _sys
    path = os.path.join(REPO, 'examples', f'{stem}.py')
    env = dict(os.environ, MPLBACKEND='Agg')
    env.pop('HYPERTOOLS_OFFLINE', None)
    proc = subprocess.run([_sys.executable, path], env=env, cwd=REPO,
                          capture_output=True, text=True, timeout=900)
    assert proc.returncode == 0, (
        f'examples/{stem}.py exited {proc.returncode}\n'
        f'--- stdout ---\n{proc.stdout[-2000:]}\n'
        f'--- stderr ---\n{proc.stderr[-2000:]}')


def test_no_launch_notebook_committed_an_error_output():
    """A notebook can be fully executed and still be broken."""
    import json
    for stem in ('market_forecast', 'weather_decades', 'painting_embeddings',
                 'conversation_shape', 'morph_shapes_zoo'):
        nb = json.loads(_read(f'docs/tutorials/{stem}.ipynb'))
        for cell in nb['cells']:
            for out in cell.get('outputs', []):
                assert out.get('output_type') != 'error', (
                    f"{stem}.ipynb: committed a traceback "
                    f"({out.get('ename')})")
```

> **Import note — measured, and the hedge resolves to "it already works".** `from scripts.measure_native_ratio import measure` was reproduced under CI's exact invocation (`.github/workflows/test.yml` runs the bare `pytest` **console script** from the repo root, no `--import-mode`, no `PYTHONPATH`) and it **passes today without `scripts/__init__.py`**. The mechanism: `tests/__init__.py` exists, so pytest's default *prepend* import mode walks up to the first directory without an `__init__.py` — the repo root — and inserts it on `sys.path`; `scripts` then resolves as an implicit PEP 420 namespace package. So v2's "if the import fails, add…" hedge was correct to hedge, and the answer is that it does not fail.
>
> **But that is an accident, not a guarantee**, and one flag breaks it: `--import-mode=importlib` makes the same import fail with `ModuleNotFoundError: No module named 'scripts'` — **confirmed both with and without `scripts/__init__.py`**, because that mode does no rootdir-walk-and-insert at all. Nothing in this repo uses it today, so this is not an active bug.
>
> **Add `scripts/__init__.py` anyway, in the same commit.** It costs one empty file, it makes the dependency explicit rather than incidental, and nothing else in the repo imports from `scripts.` yet (`grep -rn "from scripts\.\|import scripts\b"` returns nothing) — so this is the first use and the right moment to declare it. Do **not** reach for `pythonpath = ["."]`: it is a global change to every test's `sys.path` for the sake of one helper. Whichever is chosen, verify with the **same bare `pytest` invocation CI uses**, not `python -m pytest`, which puts the CWD on `sys.path` and would mask the difference.

- [ ] **Step 3: Run the gate and confirm it passes**

Run: `.venv/bin/python -m pytest tests/test_examples_are_native.py -v`

Expected: **138 collected — 126 passed, 5 failed, 7 skipped** on the FIRST run, then 131 passed once the index sets are recorded. Derived:

| test | IDs |
|-|-|
| `test_no_notebook_budget_is_below_its_own_scripts` | 1 |
| `test_a_docstring_naming_a_removed_pattern_is_not_a_defect` | 1 |
| `test_file_is_within_its_size_budget` (10 files) | 10 |
| `test_native_ratio_is_reported` | 1 |
| `test_no_defect_marker_in_the_launch_examples` (8 markers × 10 files) | 80 |
| `test_every_allowlisted_reach_is_still_present_and_still_explained` | 1 |
| `test_no_example_or_notebook_unpacks_then_uses_the_wrapper` (10 files) | 10 |
| `test_older_tutorials_dropped_their_hand_rolled_helpers` | 6 |
| `test_analyze_tutorial_actually_plots` / `test_reduce_tutorial_mentions_describe` | 2 |
| `test_examples_produce_their_stated_artifact` (5 examples) | 5 |
| `test_every_launch_notebook_ran_every_cell_it_should` (5 notebooks) | 5 |
| `test_the_right_cells_carry_visible_output` (5 notebooks) | 5 |
| `test_each_notebook_ships_its_rendered_artifact` (5 notebooks) | 5 |
| `test_example_runs_end_to_end` (5 examples, opt-in) | 5 |
| `test_no_launch_notebook_committed_an_error_output` | 1 |
| **total** | **138** |

**The 5 first-run failures are `test_the_right_cells_carry_visible_output`, and they are intentional.** `EXPECTED_VISIBLE_OUTPUTS` ships EMPTY, and the test calls `pytest.fail()` naming the notebook and telling you to paste in the measured set. That is the whole design — a number written before the artifact exists is a guess, and this plan has now been wrong five times that way. The red is the instruction.

**Ordering, which v3 got wrong at first:** Tasks 2–6 each say to record their measured index set into `EXPECTED_VISIBLE_OUTPUTS`, but Task 8 Step 2 is what CREATES that file — so on a strict Task-2-through-8 pass there is nothing to edit yet. Resolve by running **Step 2 before Tasks 2–6** (it is a pure test-module addition with no dependency on the rewrites), or, if Task 8 is genuinely run last, by treating the five failures as this step's to-do list and populating them here. Either way the dict is filled from a real `scripts/execute_tutorial.py` run, never from arithmetic.

**The 2 skips are the `PRIVATE_API_EXCEPTIONS` pairs** (`ani\._args` and `hypertools\._shared`, both on `animate_market_forecast.py`) — allowlisted by Contract 3, and each skip prints its recorded reason. Plus Step 0's `tests/plot/test_hyper_animation_accessors.py` → **9 passed** (8 + the 2-D morph case from M7), so Task 8 contributes **147** in total.

v1 expected 109 by counting a 10-ID ratio gate that this revision removed; v2 expected 106 before this revision split the notebook gate into execution / index-set / artifact and added the Contract 3 and Contract 8 guards. **Verify by real collection, not by this table** — `pytest tests/test_examples_are_native.py --collect-only -q` — because `BUDGETS` is now computed from `SCRIPT_BUDGETS`, and a naive AST count of the parametrize argument returns 1 for it rather than 10.

If a size budget fails, cut presentation code or renegotiate the budget **in this plan** — never raise it silently in the test file.

- [ ] **Step 4: Re-measure everything and record the result**

```bash
.venv/bin/python scripts/measure_native_ratio.py examples/animate_conversation.py \
    examples/animate_market_forecast.py examples/animate_morph_zoo.py \
    examples/animate_painting_embeddings.py examples/animate_weather_decades.py \
    docs/tutorials/*.ipynb
```

Paste the table into the commit message. **This is a record, not a gate** — there is no floor to be "at or above" any more. Read it as a trend against the pre-plan audit baseline (five launch examples: 48/739 native code lines, 6.5%), and if a rewrite lands far below what its siblings manage, ask why *in review* rather than letting a threshold decide. The v1 floors (26/18/20/25/26%) were set before the rewrites existed and four of the five missed them; keeping them would have blocked the plan on a number that measures formatting as much as content.

- [ ] **Step 5: Run every example headless**

```bash
for f in examples/animate_conversation.py examples/animate_market_forecast.py \
         examples/animate_morph_zoo.py examples/animate_painting_embeddings.py \
         examples/animate_weather_decades.py; do
  echo "== $f"; MPLBACKEND=Agg .venv/bin/python "$f" || break
done
```
Expected: each exits 0, with no traceback and no `UserWarning` about an ignored kwarg.

- [ ] **Step 6: Give the five launch tutorials a visible figure**

Measured: none of the five has a gallery thumbnail (`scripts/generate_gallery_thumbs.py:26` hard-codes six stems). Extend it:

```python
MPL_ANIMS = ['animate', 'animate_MDS', 'animate_spin', 'chemtrails',
             'precog', 'save_movie',
             'animate_conversation', 'animate_market_forecast',
             'animate_morph_zoo', 'animate_painting_embeddings',
             'animate_weather_decades']
```

Then, after a docs build has produced `docs/auto_examples/images/`:

```bash
.venv/bin/python scripts/generate_gallery_thumbs.py
ls -la docs/_static/thumbnails/sphx_glr_animate_{conversation,market_forecast,morph_zoo,painting_embeddings,weather_decades}_thumb.gif
```

and add an `.. image::` line to each of the five sections of `docs/tutorials.rst`, following the pattern already used for `plot_story_trajectories` (`docs/tutorials.rst`, the "Story trajectories" section):

```rst
.. image:: _static/thumbnails/sphx_glr_animate_market_forecast_thumb.gif
   :width: 400
   :alt: Six sector trajectories and a market mean, each with its own next-day forecast
```

Expected: five new thumbnails, each **under 1.1 MB** (the largest existing one, `sphx_glr_plot_story_trajectories_thumb.gif`, is 1 065 855 bytes).

- [ ] **Step 7: Run the FULL suite**

Run: `.venv/bin/python -m pytest -q`
Expected: the baseline plus **178** — Task 1's **19** (`test_image_palette.py`), Task 5's **12** (`test_recency_fade.py`), and Task 8's **147** (138 in `test_examples_are_native.py` + 9 in `test_hyper_animation_accessors.py`) — all passing, 13 skipped, plus **7 more skips** (2 `PRIVATE_API_EXCEPTIONS` + 5 opt-in smoke tests).

**Verify by real collection, never by this number.** Three revisions running, the stated figure here has been stale: v1 said 17 + 109 = +126, v2 said 16 + 106 = +134, and both were wrong. Run `pytest <file> --collect-only -q` per file and add them up. Any new failure in `tests/test_docs_thumbnails.py` or `tests/test_docs_gallery_log_filter.py` is Step 6's doing — fix it there.

- [ ] **Step 8: Build the docs to the RTD-parity standard**

Run: `cd docs && MPLBACKEND=Agg ../.venv/bin/python -m sphinx -b html -W -E -a . _build/html 2>&1 | tail -30`
Expected: build succeeds with **0 warnings**. Then verify the five tutorial pages actually show something:

```bash
grep -c "sphx_glr_animate_market_forecast_thumb" docs/_build/html/tutorials.html
.venv/bin/python -c "
import re
html = open('docs/_build/html/tutorials/market_forecast.html').read()
print('output blocks:', len(re.findall(r'nboutput', html)))"
```
Expected: `1` for the thumbnail, and a non-zero count of `nboutput` blocks (the executed outputs are rendering).

- [ ] **Step 9: Re-run everything that could have been disturbed**

Per the repo rule (*"repeat **all** checks if any changes were made to fix any of the checks"*): if Steps 6–8 changed anything, re-run Steps 3, 5, 7 and 8 in that order and confirm all four are green **in the same tree**.

- [ ] **Step 10: Commit**

```bash
git add scripts/measure_native_ratio.py scripts/generate_gallery_thumbs.py \
        tests/test_examples_are_native.py docs/tutorials.rst \
        docs/_static/thumbnails/
git commit -m "test(docs): gate examples and tutorials on native ratio + defect markers; add launch thumbnails"
```

---

## Decisions still needed

> **These entries are deliberately UNNUMBERED — cite them by name.** Five instances of citation drift in this plan set traced to numeric references going stale under reordering; the README's and animation-core's lists were de-numbered for the same reason.

Flagged rather than invented. Each states the options and the exact change to switch; the plan implements the option marked **(implemented)** so it stays runnable end to end.

- **Where `image_palette` is exported.** `hypertools/__init__.py` carries a curated `__all__` (`__init__.py:46-52`) and adding a name to it is a public-API decision.
   - **(implemented)** `hypertools.plot.colors.image_palette`, beside the two existing public palette helpers (`get_palette_colors`, `continuous_colormap`), documented in a new `docs/api.rst` **Colors** section, plus the declarative `palette='image:<path>'` spelling that needs no import at all.
   - *Alternative:* also export it top-level as `hyp.image_palette`. To switch: add `from .plot.colors import image_palette` to `hypertools/__init__.py` and append `'image_palette'` to `__all__`; `tests/test_d1_code_residue.py` and any star-import test will need the new name.
   - **Needs:** maintainer preference on growing the top-level surface in a minor release.

- **The paintings example's outlier trim.** The current 85th-percentile per-cloud trim (`animate_painting_embeddings.py:172-179`) only exists between `hyp.reduce` and `hyp.plot`. Once the example is a single `hyp.plot` call over raw text, there is no such gap, and `hyp.reduce` cannot select the sentence-transformer (it has no `vectorizer=`; verified `TypeError`).
   - **(implemented)** drop the trim. UMAP with `n_neighbors=12, min_dist=0.25` already clumps each description, and the plan does not invent a library feature to preserve a cosmetic step.
   - *Alternative A:* add `vectorizer=`/`semantic=`/`corpus=` to `reduce()`/`analyze()` (audit recommendation #6), restoring the two-step shape. Small and independently useful, but it is library work no 1.1 plan currently owns.
   - *Alternative B:* add a `manip='TrimOutliers'` manipulator (audit recommendation #7) so the trim becomes a native pipeline stage — but `manip` runs **before** `reduce`, so it would trim in the 384-dimensional embedding space, which is not the same operation.
   - **Needs:** maintainer decision on whether either library addition belongs in 1.1.

- **The morph example's hand-written `normalize()`.** Per-cloud centring and isotropic rescaling is genuinely not redundant — `plot()` uses **one shared pooled affine** (`plot.py:4568-4605`, `_shared/helpers.py:24-69`) — but `normalize='within'` (`tools/normalize.py:175`, modes at `:86`) z-scores each dataset per column, which distorts a point cloud's aspect ratio.
   - **(implemented)** keep the 5-line helper, with the comment explaining exactly why it is not redundant.
   - *Alternative:* add an aspect-preserving `normalize='isotropic'` (or `'unit-cube'`) mode, and delete the helper. Roughly 20 lines plus tests in `hypertools/tools/normalize.py`; no 1.1 plan owns it.
   - **Needs:** maintainer decision on adding a normalize mode in 1.1.

- **The conversation caption.** The current example bolds the words of the window being drawn, using ~44 lines of `TextArea`/`HPacker`/`VPacker` packing rebuilt every frame (`animate_conversation.py:240-283`) plus the span bookkeeping that exists solely to feed it.
   - **(implemented)** delete it. Per-segment `title=` shows `Speaker  "the whole line"`, which is the information the caption carried; the word-level highlight is dropped.
   - *Alternative:* keep the caption, rebuilt from `on_frame=` instead of `ani._func`. It would stay legitimate class-**D** presentation and no longer reach into private state — but it re-adds ~50 lines and would push the example past its 72-line budget, so the budget would move too.
   - **Needs:** maintainer call on whether the word-level highlight is load-bearing for the demo.

- **How the five launch tutorials get a visible figure.** They currently ship only **1–4 executed cells each** (re-measured 2026-08-01: 2/6, 4/7, 1/6, 2/6, 2/7), and `nbsphinx_execute = 'never'` means their docs pages show the rest as code and nothing else.
   - **(implemented)** both halves: execute them (Tasks 2–6 Step "Execute and measure", pinned by `test_every_launch_notebook_ran_every_cell_it_should`) **and** add a gallery thumbnail to `docs/tutorials.rst` (Task 8 Step 6), the pattern the repo already uses for `plot_story_trajectories`.
   - *Alternative A:* thumbnails only, leaving the notebooks unexecuted. Cheapest in repo size; but then the *notebook* a reader downloads still shows nothing until they run it.
   - *Alternative B:* execute, but replace each final `HTML(ani.to_jshtml())` with `ani.save('<name>.gif')` + an `Image` display, committing the GIF. The repo already commits gallery GIFs of 9–11 MB (`docs/tutorials/conversation_serial.gif` is 9 466 849 bytes), so this is precedented but heavy.
   - **Needs:** maintainer's repo-size preference. A `jshtml` blob for a 240-frame animation is large; if it turns out to exceed roughly 5 MB per notebook, switch to Alternative B and record the measured sizes.

- **Whether the market example should report a disappointing number.** The current example prints a 66% directional accuracy computed over 4-month horizons on a 5-series FRED basket. At `t=1` (next day, the maintainer's specified horizon) on a near-random-walk price series, a single linear-Gaussian filter should be expected to land close to 50%.
   - **(implemented)** report whatever it measures, with "50% = coin flip" printed alongside, and make no attempt to tune the example until the number flatters the library.
   - *Alternative:* restore a multi-day horizon, where the current example's measured 66% came from. That contradicts the explicit `t=1` instruction, so it is not implemented.
   - **Needs:** nothing, unless the maintainer would rather the gallery not advertise a coin flip — in which case the honest fix is a different demonstrator, not a different horizon.

---

## Self-Review

**Every requirement in the brief, mapped to where it is discharged.**

| requirement | discharged by |
|-|-|
| Read both audits first | Both read in full; their per-file classifications drive every "what goes, and to what" table, and their headline numbers are reproduced independently (48/739 = 6.5% vs. the audit's 6.0%) in *Verification note*. |
| Match the siblings' v2 format and rigor | Same skeleton: goal / architecture / tech stack → verification note → contracts → global constraints → prerequisites → file structure → TDD tasks with `- [ ] **Step N:**` → decisions → self-review. The "Revision note (v2)" slot is filled by a **Verification note (v1)** that plays the same role — a table of received claims against measurements — because this plan has not yet been adversarially reviewed and inventing a revision history would be a fabrication. |
| Explicit contracts | Seven, covering the script/notebook lockstep, the no-private-reaches rule, network-in-examples-only, scoring-stays-out-of-the-library, and the "budgets are contracts, never weakened to fit the code" rule. |
| Prerequisites, per task | A per-task table naming the *specific* tasks of Plans 1–3 each rewrite needs (e.g. Market ← MultiIndex T1/T2/T5/T6/T8 + Forecast T3/T4/T5 + Animation-core T1) and *why*, including the two tasks (1 and 7) that have none and can start immediately. |
| Task 1 is library work, TDD, justified API, no largest-cluster bug | Task 1: **19** real tests written before the implementation; the API choice (one function + **two** interception points — `_get_palette`'s string branch at `colors.py:305-306` and `_seaborn_palette_arg` at `plot.py:113`) is justified against the consumers each serves; the ordering rule is `frac × chroma` with a documented achromatic fallback, and `test_a_vivid_minority_colour_beats_the_muted_background` asserts the exact colour (`0.863, 0.078, 0.078`) the buggy rule fails to produce. Both states were **run**: red = `ValueError: 'image:...' is not a valid palette name`, green = the prototype's measured output. |
| Tasks 2–6 rewrite one example + its notebook each, in lockstep | Each task rewrites both, in one commit, and specifies the notebook's full cell table. Lockstep is enforced mechanically by `tests/test_examples_are_native.py`, which scans `.py` **and** `.ipynb`. |
| Market = MultiIndex showcase, per-sector + market forecasts, colour by price, ticker panel, accuracy overall + per sector in the tutorial, `t=1` | Task 2: `(Market, Sector, Ticker)` columns over 24 verified tickers; 6 sector traces + 1 market-mean trace with hierarchy-derived widths; `hue=` nested one sequence per sector (MultiIndex T6 form 2); `predict='Kalman', t=1, forecast_trail=16`; a right-hand panel listing each sector's tickers and score; the accuracy loop is example code with a **measured** 210-fit / 7.3 s budget. |
| Weather = the paper figure, nearly all native, a handful of lines | Task 3: one `hyp.plot` call, verified end to end today (0.3 s, no warnings, 2 axes, 879 distinct colours at frame 150); the 70-line second panel and the 26-line hand-built hierarchy are deleted; budget ≤ 62 code lines. |
| Paintings = full `text` displayed, native embeddings, native palette, names via `labels=` | Task 4: the side panel renders `PAINTINGS[name]['text']` (not `blurb`); `vectorizer='all-MiniLM-L6-v2', semantic=None, corpus=None`; `color=[image_palette(path)[0] ...]` from Task 1; `labels=` nested, one non-None entry per cloud at its middle window — the per-observation semantics verified on real annotations. |
| Conversation = native text, `animate='serial'` + `chemtrails`, per-segment titles | Task 5: list-of-lists of strings in; `animate=True, order='serial', chemtrails=True`; `title=[one per turn]`. The collision I feared (categorical hue collapsing 28 turns and breaking per-dataset titles) was **measured and disproved**: 6 datasets stay 6 datasets with a 3-entry legend. |
| Morph = native per-segment titles, explicit `morph_samples`, keep the teapot, cube scaling, closed loop | Task 6: `title=titles` replaces the private `_morph` reach; `morph_samples=N` kept and now load-bearing; teapot, `CUBE_SCALE`, and `clouds.append(clouds[0])` all kept with the reasons restated. |
| Task 7 groups the 15 older tutorials so each step is reviewable | Five steps, one per recurring fix (ffmpeg, HF embed, `ax=`, `manip='Smooth'`, `analyze`/`reduce`), each with its own execution check, its own `grep` assertion, and its own commit. |
| Task 8 re-measures per file, asserts improvement, full suite, 0-warning docs | Task 8: a committed metric, a 109-test gate, a per-file re-measure, all five examples run headless, the full suite, the CI-parity `python -m sphinx -b html -W -E -a` build with 0 warnings, plus a rendered-output check and a re-run-everything step. |
| BEFORE and AFTER per example, from the audit's baseline | Each of Tasks 2–6 opens with the measured BEFORE (raw lines, code lines, native lines, ratio, **and** the audit's A/B/C/D/NATIVE classification) and the contracted AFTER budget, which Task 8 asserts. |
| Network in examples only; keep the offline-fallback property | Contract 4, and every rewrite implements the existing shape: `try/except Exception: return None` + a deterministic synthetic substitute + a `print` naming the source. Task 1's tests write PNGs to `tmp_path`; `image_palette` refuses URLs by design. |
| Real file:line citations | Every claim about existing code cites a line I opened in this session: `docs/conf.py:131`, `plot.py:1066/882/895/930/950/1013/1064/1246/2750-2751/3039-3050`, `colors.py:24/105/227/250/269/287/305-306/323-331`, `text2mat.py:89/184/391/404`, `animate.py:84`, `smooth.py:14/232`, `morph.py:36`, `scripts/generate_gallery_thumbs.py:26`, plus per-example line ranges. |
| Don't invent unspecified decisions | Six items in *Decisions still needed*, each with the implemented option and the exact edit to switch. |

**Placeholders.** None. Every step carries runnable code or an exact command with its expected output. No step says "similar to Task N"; the five example rewrites are written out rather than cross-referenced, precisely because they differ.

**Type consistency.** `image_palette` returns `np.ndarray (k, 3)` float64 in `[0, 1]`, `k ≤ n_colors`, in every path (file, PIL, uint8 array, float array) — asserted by `test_returns_rgb_floats_in_the_unit_range` and `test_accepts_a_pil_image_and_a_numpy_array`. `_get_palette` still returns a list of RGB tuples for the `'image:'` branch, because it rebinds `palette` to a colour list and falls through to the existing list handling — so the short-list blending and the too-few-colours error are the same code paths as for any user-supplied list. `measure(path)` returns `(int, int)` and is imported by both the script's `__main__` and the test module, so there is exactly one implementation of the metric.

**Task dependencies.** 1 → 4 (`image_palette`). 2–6 each depend on Plans 1–3 as tabulated. Task 2 Step 1 creates `scripts/execute_tutorial.py`, which Tasks 3–7 use; Task 8 Step 1 creates `scripts/measure_native_ratio.py`, which Tasks 2–7 use in their measure steps — **do Task 8 Step 1 first** if working strictly in order, as noted in Task 2 Step 6. Tasks 1 and 7 have no dependency on Plans 1–3 and can run in parallel with them.

**Suite arithmetic.** Task 1 adds **19**; Task 5 adds **12**; Task 8 adds **147** (138 in `test_examples_are_native.py`, itemised in its Step 3 table, plus 9 in `test_hyper_animation_accessors.py`). Total **+178**, on top of whatever Plans 1–3 leave the suite at. The baseline itself is moving — re-measured 2026-08-02 at `2782/2784 collected`, and Plan 3's Tasks 0–1 have since added 17 — so measure it when this plan starts rather than trusting any number written here.

**Remaining risk.** Three places:

1. **Task 2 is the largest rewrite and the most dependent** — it consumes eight tasks across two other plans. If MultiIndex T6 (`hue` through a hierarchy) slips, the example still runs but colours by group instead of by price; that is a visible regression, not a crash, so no size or defect-marker gate would catch it. The guard is Task 2 Step 4, which asserts `axes: 2` (a colorbar exists ⇒ a continuous hue survived) and at least two distinct linewidths.
2. **The accuracy readout is the only unbounded cost in the plan.** It is pinned to a measured budget (210 fits / 7.3 s at `WINDOW=60, N_SCORED=30`), and the measurements at 250 rows (30.7 s) are recorded so a future change to those constants is an informed one.
3. **Notebook execution is the step most likely to be skipped under time pressure**, and it is exactly the step that keeps the defect from staying published. `test_every_launch_notebook_ran_every_cell_it_should` makes skipping it a test failure rather than an oversight.
