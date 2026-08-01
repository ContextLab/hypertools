# HyperTools 1.1 — Animation Core Implementation Plan (v4.4)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give `hyp.plot` the animation primitives its own gallery examples currently hand-roll, so a tutorial can showcase hypertools instead of working around it — and make the matplotlib and plotly backends mean the same thing by the same call.

**Architecture:** Three defects are fixed first (title stringification, animated-hue `linewidth`, morph intractability); they are independent and cheap. Then the plotly backend gains the serial trail composition matplotlib already has, so the two backends are at parity **before** a new spelling is layered on top. Only then is `order=` introduced — and it is **folded into the resolved backend mode** inside `_resolve_animate_mode`, so every one of the four downstream `animate` consumers gets the right value without a per-site substitution. Per-dataset `alpha=` joins the existing named-kwarg machinery with an explicit precedence rule. A public per-frame hook is added on **one shared mutable callback registry** created in `plot()` and adopted (never re-created) by `HyperAnimation`. Per-segment titles are built on that registry, discriminating morph holds from transitions by **segment parity**, not by a float. The hook's receipt type, `FrameContext`, joins the curated public surface; the `FrameHooks` registry stays internal. The plan closes with a written **animation guide** (`docs/animation.rst`), linked into the site navigation and pinned by its own tests. Every change is additive: no existing call signature changes meaning.

**Tech Stack:** Python 3.12.10, numpy 2.3.5, pandas, matplotlib 3.10.8 (primary backend), plotly 6.8.0 (interactive backend), scipy 1.17.0, pytest 9.0.2.

---

## Revision note (v4.4)

One finding from the 2026-07-31 review of v4.3 (round 13). Wording only — no task, code block, test count or interface changed, so v4.3's arithmetic and checkpoints below stand unaltered.

| # | finding | fix |
|-|-|-|
| **B1** *(baseline)* | round 13's own fix added **3 tests** to `tests/test_load_sources.py` (mixed-aggregate, exception-type, and strict-live-source cases), so the verified baseline moved **2551 → 2554** | **Every derived checkpoint shifted by +3**, because the plan's drift detection compares each task's run against its stated running total — leaving them would have fired a false drift at *every* checkpoint and masked a real one. New chain, re-derived from the additions rather than hand-edited: **2563, 2568, 2579, 2589, 2609, 2619, 2660, 2673, 2692**. The additions themselves are unchanged (9+5+11+10+20+10+41+13+19 = **138**), so 2554 + 138 = **2692** |
| **W1** *(Low)* | v4.3's portable callback rule read *"never write a mutation that fires on one frame only"*, which **forbids a legitimate thing** — highlighting exactly one frame — rather than the defect it meant to ban | The rule is restated as an **assignment** rule: *assign the complete desired value on every invocation, including the default*. It bans a per-frame **assignment**, not a per-frame **decision** — putting the condition in the value (`set_color(HIGHLIGHT if ctx.frame == target else DEFAULT)`) is portable across both artist-lifetime regimes and is now shown as a worked example in `FrameContext.artists` **and** in the guide. `test_animation_guide_gives_both_failure_modes_not_just_persistence` gains two assertions (`'assign the complete value'`, `'highlighting exactly one frame'`) so the absolute form cannot return; it stays **one** test, so no count moves |

---

## Revision note (v4.3)

Two findings from the fourth 2026-07-30 review of v4.2, plus two the same audit turned up. **Every claim below was measured against the real backends before any edit.**

| # | finding | verification | fix |
|-|-|-|-|
| **H1** *(High)* | the artist-persistence prose contradicted the lifetime table two paragraphs above it | **Measured, and the review is right.** Drove all five plotly styles and compared frame payload identity: `fig.frames[0].data[0] is not fig.frames[1].data[0]` for `True`, `'serial'`, `'window'` **and** `'morph'` — 4/4 isolated, data present in every frame; `'spin'` carries **no frame data at all** (0/4). So *"on both backends … `if ctx.frame == 0:` colours the entire animation"* is **false for four of the five plotly styles**, where that callback colours **only** frame 0. The guide repeated it and demonstrated it with `.set_color()`, a **matplotlib** `Artist` method that does not exist on a plotly trace | The portable rule is kept — **set the complete state for the current frame** — but re-founded on the correct per-backend reason, and both failure modes are now stated as opposites: shared artists (matplotlib all styles, plotly spin) leak a conditional mutation **forward**; per-frame payloads (plotly parallel/serial/window/morph) **confine** it. Fixed in `FrameContext.artists` and in the guide, which now carries a labelled `# MATPLOTLIB ONLY` `.set_color()` example **and** a `# PLOTLY ONLY` `.line.color` counterpart (`.line.color` verified settable on `go.Scatter` and `go.Scatter3d`; `go.Mesh3d` has no `.line`). New `test_plotly_non_spin_frames_are_isolated_per_frame` (×4) pins the mirror-image behaviour, and a new guide test fails if the one-sided claim returns |
| **H2** *(Medium)* | `FrameContext.artists` declared `List[Any]` but plotly spin supplied a `tuple` | **Confirmed, and broader than reported.** It is not two sites but **eleven**: seven matplotlib updaters pass lists (`artists=list(lines) + [...]`, `artists=[morph_state["artist"]]`) and four plotly branches pass `frame_traces` or `tuple(fig.data[i] …)`. `type(ctx.artists)` would have varied by backend **and** style on a new public field. The same split reaches `datasets` and `revealed_counts` | Canonical containers, normalized in **one** place: `artists`/`datasets` are `Tuple[Any, ...]`, `revealed_counts` is `Optional[Tuple[int, ...]]`, and `__post_init__` coerces via `object.__setattr__` so no branch — present or future — has to remember. Tuples rather than lists because the dataclass is `frozen=True`, which a mutable list made half-true; membership is now fixed while the contained artists stay mutable, which is the point of the hook. Imports change to `from dataclasses import dataclass` / `from typing import Any, Optional, Tuple`. New `test_frame_context_containers_are_canonical_tuples` covers all six container-producing branches |

**Found by the same audit, not in the review:**

| # | defect | verification | fix |
|-|-|-|-|
| **H3** | **Task 8 referenced a helper that Task 7 never defines.** Task 8's serial-title block called `_serial_current_index` (leading underscore) and its prose said that name is *"imported from `matplotlib_backend`"*, but Task 7 Step 4 defines `serial_current_index` — **no** underscore, the spelling used at all four other sites. Task 8 would have failed with `ImportError` | Grepped every occurrence: Task 7 uses the bare name in **four** places — its Step 4 definition, its Step 5 matplotlib call, its Step 6a plotly-branch table, and the *Interfaces* list — and Task 8's two sites were alone in the underscored spelling. *(Cited by step, not by plan line number: this revision shifts every line below it, which is how five of the six recorded citation-drift instances started.)* | Renamed at both Task 8 sites. While there: Task 8 recomputed `_counts = [np.atleast_2d(a).shape[0] for a in data]`, which is a duplicate of `lengths` — verified against source that `lengths` (`plotly_backend.py:2823`) and `starts` (`:2825`) are both already bound immediately above the frame loop, so the block now uses them directly and the plan says not to recompute |
| **H4** | `assert ctx.revealed_counts == drawn` would break under H2 | Ran it: `(17, 4, 0) == [17, 4, 0]` is **False**. H2 would have turned `test_revealed_counts_match_the_drawn_artists_with_unequal_lengths` red for a reason unrelated to what it tests | `drawn` is built as a tuple, with the reason in the docstring so nobody "fixes" it back |

**Suite arithmetic:** Task 7 **31 → 41** (+4 isolation cases, +6 container cases); Task 9 **18 → 19**. Total **127 → 138**; final **2,689 passed / 13 skipped**. Checkpoints from Task 7 on: **2657, 2670, 2689**. The Task 9 `def test_` column is also corrected from 8 to 9 — a long-standing miscount in that column only; the collected figure was always right.

## Revision note (v4.2)

Two High findings from the third 2026-07-30 review of v4.1. **Both reproduced against the real backends before editing.**

| # | finding | verification | fix |
|-|-|-|-|
| **G1** *(High)* | v4.1 claimed *"every style on matplotlib is per-frame throughout"* — **false** | **Measured.** Drove a real animation three frames and compared artist identities: `id()` of every `Line2D` is **unchanged** across frames 0/1/2. `FuncAnimation`'s updater mutates the same objects in place; matplotlib never hands out a fresh artist. My v4.1 framing treated spin as *the* exception when in fact **shared is the majority case** and plotly's reveal/morph payloads are the exception | The spin-exception framing is **replaced by an explicit backend/style lifetime table** in `FrameContext.artists`, in the guide (new *"Artist lifetime"* section) and in Step 6a's table: *matplotlib all styles* = shared live artists; *plotly spin* = shared figure traces; *plotly surfaced spin* = shared traces + per-frame `Mesh3d`; *plotly parallel/serial/window/morph* = per-frame payloads. The consequence is stated as the rule callers need — **set the complete state for the current frame**, since `if ctx.frame == 0: …` colours the whole animation — and the guide now says explicitly that *"retained in the rendered frame"* does **not** imply isolated per-frame artists. New test `test_matplotlib_artists_are_shared_across_frame_deliveries` pins the identity, and the guide test now fails if the corrected claim regresses |
| **G2** *(High)* | v4.1's surfaced-spin test used a **dead discriminator** | **Confirmed:** `go.Mesh3d` has an `.x` attribute (checked directly), so `hasattr(t, 'x')` matches meshes *and* scatters. `plain` therefore equalled the full length and `assert len(artists) > plain` could never pass — the test would have failed outright rather than testing the mixed contract | Discriminates on **type** now: `any(isinstance(t, go.Mesh3d) …)`, `isinstance(artists[-1], go.Mesh3d)`, the trailing mesh differs per frame, **and** the leading entries are asserted identical across frames *and* present in `fig.data` — which is what actually pins the shared/per-frame mixed contract |

**Suite arithmetic:** Task 7 **30 → 31**. Total **126 → 127**; final **2,678 passed / 13 skipped**. Checkpoints from Task 7 on: **2647, 2660, 2678**.

## Revision note (v4.1)

Two blockers from the second 2026-07-30 review of v4. **Both confirmed against source before editing**; no task added, removed, retitled or renumbered.

| # | blocker | verification | fix |
|-|-|-|-|
| **F1** *(High)* | plotly **spin** callbacks had no defined `artists` value | **Confirmed.** v4 said *"`artists` is `frame_traces`"* for all four branches, but the spin branch never builds a `frame_traces`. Its whole per-frame payload is `dict(name=str(k), layout=dict(scene_camera=...))` — **no `data` key at all** (`plotly_backend.py:2695-2699`), matching the branch's own comment: *"the FULL dataset is static in 'spin' mode (only the camera rotates)"*. A literal implementation raises `NameError`; `[]` would falsely claim a spin frame draws nothing. A `data` key exists only when surfaced, and then it is `surf_data`, the re-shaded `Mesh3d` updates (`:2711-2735`) | Step 6a now states the contract **per branch**: `frame_traces` for morph/serial/parallel; for spin, `tuple(fig.data[i] for i in trace_indices)` — the shared static traces the frame really renders — with `tuple(surf_data)` appended when surfaced. **Scope verified**: `fig` is a parameter (`:2517`), `trace_indices` binds at `:2602`, `surface_trace_indices` at `:2609`, all before the spin branch at `:2666`, so no new plumbing. The *shared-not-per-frame* consequence is documented rather than hidden, in `FrameContext.artists`, the guide and the CHANGELOG. Three new tests: spin artists are non-empty and identical across frames, surfaced spin appends per-frame mesh updates, and a spin mutation is retained **figure-wide**. The review is right that the parity test cannot catch this — it excludes `artists` by design, because artists were never a cross-backend guarantee |
| **F2** *(Medium)* | the guide presented post-construction registration as cross-backend | **Confirmed by running it.** `hyp.plot(..., animate=True)` returns `HyperAnimation` on matplotlib but a plain `plotly.graph_objects.Figure` on plotly, with its frames **already built** (measured: 2 frames present at return). `plot.py:4605-4612` says so directly — *"only animated matplotlib plots set `line_ani`; plotly and static plots leave it None"* — and `HyperAnimation` is constructed only under `if line_ani is not None`. So plotly has neither the method nor a window to use it. The neighbouring `ctx.axes` example was also matplotlib-specific while sitting in a backend-general section | The guide now leads with the portable form (`on_frame=` passed to `plot()`, both backends), gives **labelled** `# MATPLOTLIB ONLY` and `# PLOTLY ONLY` examples, and adds a *"Registering after construction is matplotlib-only"* section explaining **why** it cannot exist on plotly. The `plot()` docstring and CHANGELOG entry are qualified the same way. Three new guide tests pin the qualification, the labels, and spin's shared artists |

**Suite arithmetic:** Task 7 **27 → 30**, Task 9 **15 → 18**. Total **120 → 126**; final **2,677 passed / 13 skipped**. Checkpoints from Task 7 on: **2646, 2659, 2677**.

## Revision note (v4)

Four maintainer blockers from the 2026-07-30 review, plus three defects the review did not name that the same audit turned up. **No task was added, removed, retitled or renumbered** — sibling plans cite *animation-core* **Task 5** and **Task 7** by number and those remain load-bearing. Task 7 gains Step **6b**; Task 9 grows from 6 steps to 9.

| # | blocker | what was wrong | fix |
|-|-|-|-|
| **B1** | *No animation guide is actually planned* | **Confirmed.** Task 7 Step 8 and contract 3 both pointed at "the guide", and Task 9 was titled *"CHANGELOG, docs, and example cleanup"* — but no step created any `.rst`. The single `docs/` reference in the entire plan was `git add CHANGELOG.md docs/ examples/`, staging a directory nothing wrote to. Step 4's *"0 warnings"* gate made this self-contradictory in the other direction too: an unlinked new `.rst` would have **failed** the build | New **Step 2** writes `docs/animation.rst` in full (style/order, trails, title sequences, per-dataset styling, `simplify=`, `on_frame=`, the callback contract, backend scheduling, `_func`/`_args` migration); new **Step 3** links it into `docs/index.rst`'s toctree — mandatory, not cosmetic, because of the zero-warning standard; new **Step 4** adds `tests/test_animation_guide_docs.py` (**15** tests) pinning existence, toctree membership, topic coverage, both backend schedules, the contract sentence verbatim, and a regression guard against the word "pure" returning |
| **B2** | *The callback contract is still misstated* | **Confirmed, and the plan contradicted itself in adjacent lines**: the `plot()` docstring said *"must be a pure function"* and the example on the very next line called `ctx.axes.set_title(...)` — a mutation. `on_frame` exists to mutate artists; a literally pure callback would do nothing | Replaced at **all 8 sites** in this plan, both README sites, and the two *rendering* sites in Plan 3, with the maintainer's wording verbatim: *"Callbacks must be deterministic and idempotent for a given frame context. They must not depend on call count, call order, wall-clock time, or accumulated external state."* The distinction is now stated positively — **accumulation** is forbidden, **effects** are not — and the sanctioned precompute-then-index idiom is named. *"Output parity"* → *"context-metadata parity"* throughout; `test_on_frame_output_parity_across_backends` → `test_on_frame_context_metadata_parity_across_backends`. Two **mutation-retention** tests added (Step 1b), one per backend; the plotly one also pins the Step 6a dispatch **order**, since dispatching after `frames.append` would silently drop every mutation |
| **B3** | *`FrameContext` lacks a resolved public-API decision* | **Confirmed.** The plan called it public but exposed it only at `hypertools.plot.animation_context.FrameContext`, contradicting 1.0's curated `__all__` (`hypertools/__init__.py:43-52`) | New **Step 6b**: export `hypertools.FrameContext`, add to `__all__`, add to `docs/api.rst` beside `HyperAnimation`, and update the **hardcoded** `documented` literal at `tests/test_codeorg_licensing_audit_fixes.py:295-300` — all four **atomically**, since that literal is a hardcoded set and touching `__all__` alone turns the suite red (verified). `FrameHooks` stays internal; a new test asserts both directions. The README now separates the public callback API from the internal registry |
| **B4** | *Plan 1 / Plan 4 documentation ownership is ambiguous* | **Confirmed and mutual** — Plan 4 has a dedicated task per example (T2 market, T3 weather, T5 conversation, T6 morph) covering all four files Task 9 rewrites, 52 mentions total | Task 9 Step 5 is retitled **"MECHANICAL MIGRATION ONLY"** and carries a boundary table; Plan 4's contract 3 carries the reciprocal table. Plan 1 removes private reaches without changing behaviour; Plan 4 owns all narrative, visualization and notebook work, and **owns its own metrics** — Plan 1 must not enforce them against files Plan 4 has not rewritten yet |

**Found by the same audit, not in the review:**

| # | defect | fix |
|-|-|-|
| **E1** | **Every 1.1 plan's docs-verification step could not run.** All six `cd docs && make clean && make html` instructions failed: `make` invokes the `sphinx-build` **console script**, whose `sys.path[0]` is the venv `bin/`, so `docs/conf.py`'s `from _gallery_log_filter import install` raised `ModuleNotFoundError`. Reproduced twice. Worse, `make html` omits `-W`, so even a working invocation would not have enforced the zero-warning gate it claimed to | All six replaced across **all four plans** with the byte-for-byte CI command (`.github/workflows/test.yml:283-291`): `cd docs && MPLBACKEND=Agg ../.venv/bin/python -m sphinx -b html -W -E -a . _build/html`. **Superseded in part (2026-07-30):** the underlying import failure is now fixed at the source — `docs/conf.py` adds its own directory to `sys.path`, so `make html` works again (verified: builds clean, 0 warnings). The plans still mandate the CI command, because only it applies `-W` |
| **E2** | **This plan's *Decisions* list was numbered**, with three numeric back-references — the exact drift pattern that has now produced **five** stale citations across the plan set | De-numbered to named bullets with the rationale inline, matching the README. All three back-references re-pointed by name. Plan 4's list de-numbered too, and its two numeric citations re-pointed |
| **E3** | **Plan 4 stated the pre-`simplify=` morph behaviour** — *"animation-core Task 3 makes an uncapped morph above 2000 points **raise**"*. Under the resolved decision the default silently **caps**; only `simplify=False` raises | Corrected in place, with the better reason to keep `morph_samples=N` stated: explicitness and reproducibility, not error-avoidance |

**Suite arithmetic:** Task 7 **24 → 30** (+2 mutation-retention, +1 export, +3 plotly-spin artists); Task 9 **0 → 18** (the guide's tests). Total added **102 → 126**; final expected **2,677 passed / 13 skipped** against the verified 2,551 baseline. Every cumulative checkpoint from Task 7 onward was recomputed: 2646, 2659, 2677.

## Revision note (v3)

The maintainer **resolved all four** of v2's open decisions (2026-07-29). v3 folds the resolutions in. **No task was added, removed, retitled or renumbered** — sibling plans cite *animation-core* **Task 5** (`order=`) and **Task 7** (the frame hook), and those numbers are load-bearing. v2's *Decisions still needed* section is now **empty of open items** and retitled *Decisions (all resolved)*: it records the four resolutions instead of asking for them. Nothing in this plan is waiting on anyone.

| decision | what v2 said | what v3 does | verified evidence |
|-|-|-|-|
| **1. `morph_samples` above the cap** | Task 3 **raises** unconditionally; Contract 7 asserted *"`morph_samples` never silently drops data"* | New public `plot()` flag **`simplify=True`** (default). Below the cap `simplify` is a **no-op**. Above the cap: `simplify=True` **silently downsamples to the cap** — no warning, no print; `simplify=False` **raises**, naming the measured cost and suggesting `simplify=True`. Contract 7 rewritten to the conditional guarantee | Maintainer, verbatim: *"add a 'simplify' flag to control this behavior; if below cap, simplify does nothing. otherwise either silently drop with **no** warning if simplify=True (default), or print an informative message with a suggestion to set simplify=True and then raise an exception if simplify=False."* The cost that motivates it: uncapped morph over the built-in zoo was **killed at 10 min** (`duration=1, frame_rate=2`; `hyp.load` returns 30135–36022 pts), while **`morph_samples=2000` → 8.2 s** (`notes/audit/PLAN.md:260`, row B3; same figure at `notes/session_2026-07-26_demo-polish-and-tutorial-review.md:239`) |
| **2. `on_frame=` on plotly** | Task 7 raised `NotImplementedError`, *"a plotly animation is precomputed JSON played by a browser; there is no Python frame loop to call back into"* | **That premise is false.** The `NotImplementedError` and every claim of unreachability are **removed**; `on_frame=` ships on **both** backends. Contract 2 no longer carries an exception. Task 7 owns the plotly dispatch; Task 4 references it | `_add_animation` (`plotly_backend.py:2517`) builds every frame in a **Python loop at build time**: `frames = []` (`:2601`), then `frames.append(go.Frame(**frame_kwargs))` at `:2729` (spin), `:2819` (morph), `:2865` (serial) and `:2975` (the `else:` parallel/window branch at `:2866`) — **four** branches, one per style. Re-verified in this repo today; note the fourth site, which the directive's three-site list omitted. What plotly lacks is a Python loop **during playback**: once `fig.frames` is populated the browser plays it |
| **2a. call schedules differ; context metadata does not** | — | The new contract states the schedules explicitly and requires `on_frame` to be **deterministic and idempotent for a given `FrameContext`** — *not* "pure", since mutating artists is the point. **Context-metadata** parity is asserted on the backend-independent fields; rendered output is deliberately **not** claimed to match, and per-backend mutation *retention* is tested instead | matplotlib uses `FuncAnimation(..., blit=False)` (`matplotlib_backend.py:1935`, `:1957`, `:1968`), whose updater fires at **render** time — lazily during interactive playback, eagerly when saving (`animate.py:116`; the gif/apng/video writers save every frame). Every current use case is already idempotent per frame index, verified across all four `_func`-monkeypatching examples: `examples/animate_morph_zoo.py` does `label.set_text(shape_title(frame))` — a mutation, but an idempotent one; `animate_conversation.py` and `animate_weather_decades.py` read live artist state **0** times inside their wrappers; `animate_market_forecast.py`'s only **2** artist reads are one-time setup calibration (the uppercase `SLOPE`/`BLO`/`BHI` constants, computed after a full reveal), not per-frame. **Zero of the four accumulate inside the per-frame wrapper**, verified 2026-07-30 — so the contract costs the existing gallery nothing. Two of them *do* contain `+=`, and both are the idiom the guide should teach: `animate_market_forecast.py` accumulates at **module level** (`:255`) to precompute an `ACC` frame→value array, and its wrapper (`:323`) only does `acc = ACC[min(num, total - 1)]`, an idempotent lookup; `animate_conversation.py:254`'s `used += step` is a loop-local in the deterministic helper `caption_lines()`, reset on every call. **Precompute-then-index is the sanctioned pattern for anything that looks cumulative** |
| **3. animated continuous-hue default linewidth 1.5 → 1.0** | Listed as open decision #1, implementation already in Task 2 | **SHIP IT.** No implementation change. Recorded as decided, and the CHANGELOG now records it as a **visible change** to existing animated hue figures | Task 2 Step 1's measured red state `[1.5, 1.5, 1.5]` → green `[0.5, 0.5, 5.0]`; animated **no-hue** lines are already `1.0` (`matplotlib_backend.py:1603` `pop("linewidth", 1)`), so the fix makes hue and no-hue agree |
| **4. `order='serial'` with `spin`/`window`** | Listed as open decision #3; warn-and-ignore already implemented | **Unchanged**, and no longer labelled open. Warn-and-ignore matches the repo's established convention at `plot.py:3760-3781` | Measured: `animate='spin', chemtrails=True` → *"animate='spin' does not support trail styles; ignoring chemtrails for datasets [0, 1, 2]"* |

**On the v1 history, so a reader does not read decision 1 as drift.** The v2 table below lists *"Task 3 caps `morph_samples` by default"* as a **v1 ERROR**, corrected to a raise. Decision 1 **partially reinstates capping** — deliberately, and that is not a regression to the v1 defect. The v1 error was capping **silently, with no flag, no documentation and no way to opt out**, which contradicted a guarantee stated in the source. v3 adds an **explicit, documented, overridable** flag: the guarantee still holds unconditionally below the cap and whenever `simplify=False`, the flag is in the signature, the docstring, the validator and the CHANGELOG, and `morph.py:17-24`'s own wording is updated in the same task so no in-source guarantee outlives the behaviour it describes.

**Interpretation stated plainly, so it is visible and reversible.** The maintainer wrote *"print an informative message … and then raise an exception"*. v3 implements that as **a single `raise` carrying that message** — not a bare `print()` followed by a `raise`. A raised exception's message is already surfaced to the user (traceback, `pytest.raises(match=...)`, any logging handler), so printing it first would emit the same text twice and would also write to stdout in library code, which nothing else in `plot.py` does. To switch: prepend `print(_msg)` before the `raise` in Task 3 Step 3 and add a `capsys` assertion to `test_simplify_false_over_the_threshold_raises_naming_simplify`.

---

## Revision note (v2)

v1 of this plan was adversarially reviewed (`notes/audit/review_plan1_animation_core.md`) and **six of its eight tasks rested on false or unusable assumptions**. Every row below was re-verified against the source in this repo before the rewrite.

| v1 error | verified reality |
|-|-|
| Task 2's test builds `hue = np.linspace(0, 1, ds[0].shape[0])` (30 values) for 3×30-row datasets | The validator at `plot.py:3368-3375` counts `n_obs = sum(len(xi) for xi in xform)` = **90**. The v1 test errors before any assertion: `ValueError: hue has 30 entries but the data has 90 observations`. |
| Task 2's `_widths()` unions `ax.collections` **and** `ax.lines`, and asserts `max == 5.0`, `min == 0.5` | Measured today: the hidden head `Line2D`s already carry the correct `[0.5, 0.5, 5.0]` (created with `linewidth=linewidths[idx]`, `matplotlib_backend.py:1627`), so the assertions hold **on unfixed code**. The buggy artists are the 3 head `Line3DCollection`s at `1.5` (= `rcParams['lines.linewidth']`). Restricting to `ax.collections` instead makes `min == 1.0` (the six cube-plane collections), so `min == 0.5` would fail *after* a correct fix. |
| Task 2's patch is "in `matplotlib_backend.py:1602-1604` / `:2197-2199`" and names a local `dataset_kwargs` | Those line ranges are right, but the **reader** is `_apply_multicolor_animation._linewidth`, which lives in **`plot.py:5150-5153`**, and the collections are built in `plot.py:5161-5176`. `dataset_kwargs` does not exist — the real code is a list comprehension over all datasets. And merely un-popping `linewidth` makes the four surviving `**kwargs_list[idx]` expansions (`matplotlib_backend.py:1621-1631`, `1633-1643`, `1645-1654`, `1656-1665`, plus the 2-D twins at `2214-2223`, `2225-2234`) collide with the explicit `linewidth=linewidths[idx]` argument. |
| Task 3's snippet uses `mode` and `datasets` | Neither local exists. The mode is the **rebound** `animate` (`plot.py:3653`), and the post-pipeline arrays are `xform`. There is also no "where `morph_samples` is resolved": it is validated at `plot.py:2264-2274` and passed verbatim to the backends at `plot.py:4239` / `plot.py:4324`. |
| Task 3 caps `morph_samples` by default | `morph.py:17-24` states an explicit maintainer guarantee: *"Every dataset now keeps its FULL point count instead ... **No real data point is ever dropped**"*, echoed at `plot.py:1516-1518`. A silent default cap contradicts it. Every gallery example that morphs the zoo **already passes `morph_samples=` explicitly** (`examples/plot_shape_morph.py:73`, `examples/animate_morph_zoo.py:96`, `examples/animate_surface_morph.py:122`), so a hard error breaks nothing. |
| Task 4 gates serial capability on the **raw** `animate` argument, before list handling | `animate` may legitimately be a per-dataset list (`plot.py:480-505`), e.g. `['morph', None, 'morph']`, which resolves to `'morph'` — a style v1 itself declares serial-capable. `['morph', None, 'morph'] not in (True, 'parallel', 'serial', 'morph')` is `True`, so a **valid** combination would raise `NotImplementedError`. |
| Task 4 maps ordering onto the backend at one site (`backend_mode`) | `animate` is consumed at **four** semantically distinct sites after `plot.py:3653`: the trail-ignore check (`3760`), plotly draw (`4214`), matplotlib draw (`4299`), and `_apply_multicolor_animation(style=animate)` (`4379`), which branches on `if style == 'serial':` at `plot.py:5258` to recover the reveal position. A single substitution desyncs the hue overlay from the backend. |
| Task 4's `test_order_defaults_to_parallel` / `..._equals_the_legacy_alias` / `test_animate_serial_implies_order_serial` compare `len(ax.lines) + len(ax.collections)` | Measured: `animate=True` → 9, `animate='serial'` → 9, both `+chemtrails` → 12. All three pass if `order=` is accepted and then **completely ignored**. |
| Task 4 Step 4 says "Expected: 11 passed" | The v1 file defines **12** cases (5 plain + 4 + 2 parametrized + 1). |
| Task 5 promotes `alpha` to a named kwarg with no precedence rule | `mpl_kwargs["alpha"]` is already written as a per-dataset list by two internal paths — `plot.py:3056` (row-MultiIndex level fading) and `plot.py:3629` (nested-list depth fading) — and `plot.py:71-75` documents that *"named/internal styling always wins over a same-named extra kwarg"*. Promotion silently reverses that. `plot.py:242-244` also states verbatim that `**kwargs` values like `alpha=` *"never reach `mpl_kwargs`"*, an invariant promotion breaks for `_expand_styles_to_runs`. |
| Task 6: "`HyperAnimation` subclasses `tuple`, keep the callback list on the instance" | The tuple-subclass worry is **unfounded** — `hyper_animation.py:45` declares no `__slots__`; measured: `has __dict__: True`, instance attributes set in `__new__` work and survive. The real defect is **list identity**: the updater closure is created inside `_draw`/`animate_plot3D` long before `plot.py:4612-4614` wraps the result, so `self._frame_callbacks = []` in `__new__` creates a *fresh, unreferenced* list and `HyperAnimation.on_frame(cb)` can never fire. |
| Task 6 claims "exactly one implementation" of the reveal formula | It exists in **three** places: `matplotlib_backend.py:1316-1318` (3-D), `matplotlib_backend.py:2062-2064` (2-D twin), and re-derived at `plot.py:5265-5269`. No v1 test drives a 2-D animation, and every v1 helper does `[a for a in fig.axes if hasattr(a, 'zaxis')][0]`, which raises `IndexError` on a 2-D figure (measured: `hyp.plot(ds, ndims=2, animate='serial')` → `axes zaxis? [False]`). |
| Task 6's `test_revealed_counts_match_the_library_reveal_formula` uses lengths `[20, 20, 20]` | Worse than v1 knew: for a **line** format, `plot()` pre-interpolates every animated dataset onto the frame grid, so the arrays reaching `_draw` are all exactly `total_frames` rows (measured: input `[17, 23, 11]` → `[13, 13, 13]` at `duration=13, frame_rate=1`). Unequal lengths are only reachable with a **marker** format (measured: `'.'` → `[17, 23, 11]` preserved). |
| Task 7 blanks morph titles when `ctx.current_fraction not in (0.0, 1.0)` | Holds and transitions **both** sweep 0→1, so the fraction cannot discriminate them. Computed for the v1 test's own parameters (3 clouds, `duration=6, frame_rate=4`), `segment_frame_counts(3, 24)` = `[5, 5, 5, 5, 4]`: the v1 rule blanks 12 of the 15 hold frames and names 4 transition frames. The only discriminator is `seg_idx % 2` from `morph.frame_to_segment` (`morph.py:316-328`; even = hold, odd = transition, per `morph_positions`, `morph.py:342-346`). |
| Task 7's `test_morph_titles_blank_during_transitions` asserts `0.1 < blank_fraction < 0.9` | blank_fraction = 12/24 = **0.5** under the *inverted* behaviour. The test cannot fail. |
| Task 7 advertises per-segment titles with no backend caveat, and Task 4 leaves plotly's serial trails as a documented gap | **Maintainer directive (2026-07-28): plotly and matplotlib behaviour must be identical.** `order='serial'` must work on plotly, trail flags included. Measured today: plotly `animate='serial', chemtrails=True` emits *"animate='serial' does not support trail styles; ignoring chemtrails for datasets [0, 1, 2]"* (`plot.py:3757-3781`) and produces a figure byte-identical to plain serial (4 traces, no trail traces). The gap is now **closed by Task 4**, not documented. |

Two v1 claims the review **confirmed correct** and this plan keeps unchanged:

- `serial_reveal_counts` reproduces `matplotlib_backend.py:1316-1326` exactly (`max(0, min(length, remaining))` ≡ `np.clip`). Re-verified here against **real drawn artists**: simulated counts `[4,0,0] / [17,4,0] / [17,23,2]` at frames 1/5/10 match the measured vertex counts of a `fmt='.'`, lengths `[17,23,11]`, `duration=13, frame_rate=1` serial animation byte-for-byte.
- `ani._func(f, *ani._args)` is a valid way to drive frames for `animate=True`, `'serial'` and `'morph'`.

---

## Contracts this plan establishes

1. **`animate=` names a STYLE; `order=` names an ORDERING.** `_resolve_animate_mode` folds them into ONE resolved backend mode, so `animate` from `plot.py:3653` onward is already what every backend and every downstream consumer should see. There is no second "backend_mode" variable.
2. **Backend parity is a requirement, not an aspiration.** Any `hyp.plot(...)` call that draws on matplotlib must draw the equivalent thing on plotly. Where a Python-level capability genuinely cannot cross the browser boundary, the call raises `NotImplementedError` naming the backend — it never silently degrades. **This plan leaves no such case.** It closes the serial-trail gap (Task 4), the per-segment-title gap (Task 8) and — the one v2 listed as unreachable — `on_frame=` on plotly (Task 7): plotly *does* have a Python per-frame loop, at build time inside `_add_animation` (`plotly_backend.py:2517`, appends at `:2729`/`:2819`/`:2865`/`:2975`). No `plot()` argument in this plan raises `NotImplementedError` for being on the wrong backend.
3. **One shared frame-hook registry.** `plot()` creates a `FrameHooks` object; `_draw` closes over it; `HyperAnimation.__new__` **adopts** it. A callback registered after construction fires. There is exactly one place a `FrameContext` is built and dispatched, and it is the **outermost** wrapper of `line_ani._func`, so hooks always observe final artists (including hue collections). On plotly the same registry is dispatched once per frame inside `_add_animation`'s build loop.
   **`on_frame` MUST be deterministic and idempotent for a given `FrameContext`.** State it exactly this way, in the docstring, the guide and the CHANGELOG:

   > Callbacks must be deterministic and idempotent for a given frame context. They must not depend on call count, call order, wall-clock time, or accumulated external state.

   **Do not call this "purity."** The entire purpose of `on_frame` is to mutate backend artists — `label.set_text(...)`, `artist.set_alpha(...)` — so a literally pure callback would be useless. What the contract forbids is *accumulation*, not *effects*: `label.set_text(shape_title(frame))` is idempotent and fine; `counter += 1` or `alpha *= 0.9` is not, because re-delivery of the same frame changes the result.

   The requirement exists because the schedules differ, and the schedules are part of the contract: matplotlib calls back at **render** time (`FuncAnimation(..., blit=False)`, `matplotlib_backend.py:1935`/`:1957`/`:1968`), so a given frame index **may be called more than once** across a looping animation or a save (`animate.py:116`; gif/apng/video writers save every frame); plotly calls back **exactly once per frame index**, at build time. Idempotence is precisely what makes a repeated matplotlib call indistinguishable from plotly's single call.

   What is guaranteed and tested is **context-metadata parity**, *not* output parity: for the same `on_frame`, both backends yield the same backend-independent `FrameContext` fields per frame index (`frame`, `n_frames`, `style`, `order`, `current_index`, `current_fraction`, `revealed_counts`, `segment_index`, `segment_kind`, and the `datasets` shapes). Rendered output is explicitly **not** claimed to match: `figure`/`axes`/`artists` are backend-**native** — matplotlib `Figure`/`Axes`/artists, or the `go.Figure` and that frame's traces — so a callback that mutates them is not source-compatible across backends, and asserting output parity would be asserting something false. What each backend *does* guarantee is that a mutation the callback makes is **retained** in that backend's own rendered frame; that is tested per backend (Task 7 Step 1b), not across them.
4. **`FrameContext` reports segment structure explicitly.** `segment_index` and `segment_kind ∈ {'hold', 'transition'}` are fields, never inferred from `current_fraction`. `current_fraction` is documented as progress *within the current segment/dataset*, which for morph does **not** distinguish holds from transitions.
5. **`FrameContext.datasets` are the arrays the animation actually draws from** (`data_lines` in the backend), not the raw input. For line formats those are pre-interpolated onto the frame grid; `revealed_counts[i]` indexes into `datasets[i]`.
6. **Named-kwarg precedence is unchanged.** `plot.py:71-75` says internal styling wins over a same-named extra kwarg. Promoting `alpha` to a named parameter keeps that outcome: where an internal path already writes `mpl_kwargs['alpha']` (MultiIndex, nested-list depth), a user `alpha=` **warns and is ignored**, exactly mirroring the existing `linewidth=` precedent at `plot.py:3045-3050`.
7. **`morph_samples`' no-point-dropped guarantee is now conditional, and the condition is stated everywhere it is claimed.** The `morph.py:17-24` guarantee — *"No real data point is ever dropped"* — holds **unconditionally below the tractability cap, and at any size when `simplify=False`**. Above the cap with the default `simplify=True`, hypertools **downsamples silently, by design**, because hanging is worse than approximating: the uncapped morph over the built-in zoo was killed at **10 minutes**, while `morph_samples=2000` renders the same call in **8.2 s** (`notes/audit/PLAN.md:260`). Silently means silently: no `warnings.warn`, no print. `simplify=False` restores the absolute guarantee by **raising** instead, with a message that names the measured cost and suggests `simplify=True`. `morph.py:17-24`'s own docstring is rewritten in the same task (Task 3 Step 6) so no in-source guarantee outlives the behaviour it describes.
8. **Validation is fail-fast.** Anything checkable from the raw arguments is checked at `plot.py:2231` — after `animate` normalisation (`plot.py:2199-2230`), before `morph_samples` validation (`2264`), before `resolve_font` (`2428`) and before the `plot_stream` return (`2582`), per the principle stated at `plot.py:423-430`.

---

## Global Constraints

- Target release: **1.1**. Nothing here ships to users until the whole 1.1 line is working; the Bluesky announcement waits.
- Run everything with the repo venv: `.venv/bin/python -m pytest`. The base anaconda python is broken (numpy/matplotlib mismatch).
- Run pytest from the repo root; `pyproject.toml` sets `testpaths = ["tests"]` and `timeout = 1200` (`pytest-timeout` 2.4.0 **is** installed).
- **Verified baseline: `2567 collected`, `2554 passed, 13 skipped`.** Every task below ends by re-running the whole suite; the pass count may only grow.
- **Never simplify a test to make it pass.** If a test fails repeatedly, fix the code.
- **No mock objects.** Every test drives real `hyp.plot` calls and asserts on real matplotlib artists or real plotly frames. (`monkeypatch` of a hypertools function to *observe* it is not a mock and is used nowhere in this plan.)
- Matplotlib must be forced to `Agg` in every test module: `import matplotlib; matplotlib.use("Agg")`. There is **no** `conftest.py` in this repo — each module does it itself.
- Additive only: existing `animate=True/'parallel'/'spin'/'serial'/'window'/'morph'` behavior must not change. `animate='serial'` remains supported forever as an alias.
- When any behavior changes, update the docstring in the same commit (repo rule: docs travel with code).
- All `warnings.warn` calls in `plot.py` use `stacklevel=external_stacklevel()` (`hypertools/core/model.py:32`), never a literal `stacklevel=2`.
- Commit after every task. Branch off `dev-1.0`; do not commit to `master`.
- **`git add` any NEW file BEFORE the full-suite step, not after it.** Found by executing Task 1
  (2026-08-01): `tests/test_packaging_artifacts.py::test_sdist_contains_only_tracked_files_plus_allowlist`
  builds an sdist from the working tree and fails on any file that is present but untracked, so a
  new test module costs a **wasted ~9-minute suite run** if staged only at the commit step. This is
  the packaging gate working correctly — an untracked file really would ship in the sdist while
  being invisible to git — so stage first and let the suite see the final state. Affects every task
  that creates a file (1, 7 and 9 at minimum).
- **Run the suite as `cmd > log 2>&1; rc=$?; tail log; exit $rc`, never `cmd | tail`.** A pipeline
  reports *tail's* exit status, which has twice reported a green run for a suite that had failures.
- **Every `plot.py` line number in this plan is PRE-TASK-1 and will have drifted.** All nine tasks
  edit that one file, so each task shifts the citations of the tasks after it — Task 2 already found
  its target at `:5183-5186` where the plan said `:5150-5153`. **Locate by content or symbol, never
  by line number**, and confirm the surrounding code matches the plan's description before editing.
  When you WRITE a comment or docstring, cite the **symbol**, not a line number, for the same reason:
  stale line citations have misdirected readers in this project six times.
- **`pyflakes` is not a usable gate for `hypertools/plot/plot.py`** — `from .._shared.helpers import *`
  yields ~150 spurious "may be undefined" lines. Review that file by reading the diff.

---

## File Structure

| file | responsibility | change |
|-|-|-|
| `hypertools/plot/plot.py` | public `plot()` signature, fail-fast validation, mode/order resolution, hue overlay, hook dispatch | modify |
| `hypertools/plot/matplotlib_backend.py` | matplotlib drawing + the 7 per-frame updaters; the single reveal-schedule helper | modify |
| `hypertools/plot/plotly_backend.py` | serial trail traces + per-frame titles + the per-frame hook dispatch in `_add_animation`'s build loop (backend parity) | modify |
| `hypertools/plot/morph.py` | the no-point-dropped guarantee in its module docstring (`:17-24`), restated as conditional on `simplify=` | modify |
| `hypertools/plot/hyper_animation.py` | `HyperAnimation` gains `.on_frame()` over the **adopted** registry | modify |
| `hypertools/plot/animation_context.py` | **new** — `FrameContext` (**public**) + `FrameHooks` (**internal**) | create |
| `hypertools/__init__.py` | export `FrameContext`; add to `__all__` (Task 7 Step 6b) | modify |
| `docs/api.rst` | list `FrameContext` beside `HyperAnimation` (Task 7 Step 6b) | modify |
| `docs/animation.rst` | **new** — the animation guide (Task 9 Step 2) | create |
| `docs/index.rst` | add `animation` to the toctree (Task 9 Step 3) | modify |
| `tests/plot/test_title_validation.py` | `title=` type contract, fail-fast placement | create |
| `tests/plot/test_animated_hue_linewidth.py` | animated continuous-hue `linewidth=` | create |
| `tests/plot/test_morph_samples_guard.py` | morph tractability guard | create |
| `tests/plot/test_plotly_serial_parity.py` | plotly serial + trail parity with matplotlib | create |
| `tests/plot/test_order_kwarg.py` | `order=` axis, incl. cross-backend parity | create |
| `tests/plot/test_per_dataset_alpha.py` | per-dataset `alpha=` + precedence | create |
| `tests/plot/test_on_frame_hook.py` | public per-frame hook, 2-D and hue coverage | create |
| `tests/plot/test_serial_titles.py` | per-segment titles, both backends | create |
| `tests/test_animation_guide_docs.py` | the guide's content + navigation (Task 9 Step 4) | create |
| `tests/test_codeorg_licensing_audit_fixes.py` | add `FrameContext` to the hardcoded `documented` set (`:295-300`) | modify |

---

## Task 1: Reject non-string `title=`, validated before the pipeline

Today `hyp.plot(ds, title=['a','b','c'])` draws the literal text `"['a', 'b', 'c']"` with no error and no warning. Measured today:

```
TITLE list  -> "['a', 'b', 'c']"     TITLE tuple -> "('a', 'b', 'c')"
TITLE int   -> '3'                   TITLE dict  -> "{'a': 1}"
```

Task 8 will give sequences a real meaning for serial-style animations; this task closes the silent hole first. **Placement matters** (review G3): v1 put the check after `_resolve_animate_mode` at `plot.py:3653+`, i.e. after the whole analyze/reduce/align/cluster pipeline — which contradicts `plot.py:423-430` (*"Fail fast, BEFORE the analyze/reduce pipeline runs"*), lets `resolve_font` consume `title` first (`plot.py:2424`), and misses `plot_stream` entirely (it returns at `plot.py:2582`, and `title` is in its `_stream_forwarded` set at `plot.py:2555`). Measured today: a streaming plot with `title=['a','b']` renders the title `"['a', 'b']"` with no warning.

Nothing in the repo depends on the stringification: the only non-literal `title=` in tests is `tests/test_multibyte.py:711` (`title=text`, where `text` is a `str`), and every example uses f-strings or `.capitalize()`.

**Files:**
- Modify: `hypertools/plot/plot.py` (add `_validate_title`; call it at line 2231)
- Test: `tests/plot/test_title_validation.py` (create)

**Interfaces:**
- Produces: `_validate_title(title, style=None, order=None, n_datasets=None)` → `None` or `list[str]`; raises `TypeError`. Task 8 widens the *behaviour* of this same function without changing its signature.

- [ ] **Step 1: Write the failing test**

```python
# tests/plot/test_title_validation.py
import matplotlib
matplotlib.use("Agg")

import numpy as np
import pytest

import hypertools as hyp


def _datasets(n=3, rows=10, dims=4, seed=0):
    rng = np.random.default_rng(seed)
    return [rng.normal(size=(rows, dims)).cumsum(axis=0) for _ in range(n)]


def _ax(fig):
    return [a for a in fig.axes if hasattr(a, 'zaxis')][0]


def test_title_string_is_rendered():
    fig = hyp.plot(_datasets(), '-', title='My Title', show=False)
    assert _ax(fig).get_title() == 'My Title'


def test_title_none_leaves_axes_untitled():
    fig = hyp.plot(_datasets(), '-', title=None, show=False)
    assert _ax(fig).get_title() == ''


@pytest.mark.parametrize('bad', [['a', 'b', 'c'], ('a', 'b', 'c'), 3, {'a': 1}])
def test_non_string_title_raises_rather_than_stringifying(bad):
    with pytest.raises(TypeError, match='title must be a string'):
        hyp.plot(_datasets(), '-', title=bad, show=False)


def test_title_error_names_the_alternatives():
    with pytest.raises(TypeError, match='names='):
        hyp.plot(_datasets(), '-', title=['a', 'b', 'c'], show=False)


def test_title_is_rejected_before_the_analyze_pipeline_runs():
    """Fail-fast (plot.py:423-430): the title error must beat the reduce error.

    `reduce='NoSuchReducer'` raises `ValueError: unknown reduce model ...`
    from inside analyze(), which `plot()` calls at plot.py:2804 -- far after
    the validation anchor at plot.py:2231. If validation were placed after
    _resolve_animate_mode (plot.py:3653) this test would see the ValueError.
    """
    with pytest.raises(TypeError, match='title must be a string'):
        hyp.plot(_datasets(), '-', title=['a', 'b', 'c'],
                 reduce='NoSuchReducer', show=False)


def test_stream_input_also_rejects_a_list_title():
    """plot_stream returns at plot.py:2582 and forwards `title` verbatim
    (plot.py:2555), so validation placed after that line never sees it.
    Measured before this task: renders the title "['a', 'b']", no warning."""
    rng = np.random.default_rng(0)
    stream = (rng.normal(size=4) for _ in range(40))
    with pytest.raises(TypeError, match='title must be a string'):
        hyp.plot(stream, '-', title=['a', 'b'], stream_max=20, show=False)
```

- [ ] **Step 2: Run the test and confirm it fails**

Run: `.venv/bin/python -m pytest tests/plot/test_title_validation.py -v`

Expected: `test_title_string_is_rendered` and `test_title_none_leaves_axes_untitled` PASS. The four `test_non_string_title_raises_rather_than_stringifying` cases and `test_title_error_names_the_alternatives` FAIL with `DID NOT RAISE <class 'TypeError'>`. `test_title_is_rejected_before_the_analyze_pipeline_runs` FAILS with `ValueError: unknown reduce model 'NoSuchReducer'` (wrong exception type). `test_stream_input_also_rejects_a_list_title` FAILS with `DID NOT RAISE`. **9 collected, 2 passed, 7 failed.**

- [ ] **Step 3: Implement the validator**

Add to `hypertools/plot/plot.py`, next to the other private validators (immediately after `_validate_labels_length`, which ends at line 405):

```python
def _validate_title(title, style=None, order=None, n_datasets=None):
    """`title=` is a single string for the whole figure.

    A list/tuple used to be silently stringified onto the axes (a caller
    passing one title per dataset got the literal text "['a', 'b', 'c']"
    drawn on their figure). Reject anything that is not a string so the
    mistake is visible, and point at the kwargs that ARE per-dataset.

    Returns None for the scalar/None forms. Task 8 of the 1.1 animation-core
    plan widens this to return a list of per-segment titles for serial-style
    animations; `style`/`order`/`n_datasets` are accepted (and ignored) from
    the start so that widening never changes the signature or its call site.
    """
    if title is None or isinstance(title, str):
        return None
    raise TypeError(
        f"title must be a string (or None), not {type(title).__name__}. "
        "For a per-dataset legend entry use names=; for a per-observation "
        "annotation use labels=."
    )
```

Call it at `plot.py:2231` — the blank line directly after the `animate`-value normalisation block (`plot.py:2199-2230`) and before the `morph_samples` validation at `2264`:

```python
    # fail-fast on title= BEFORE the analyze/reduce pipeline (plot.py:423-430)
    # and before resolve_font (plot.py:2428) / the plot_stream return
    # (plot.py:2582) both consume it.
    _validate_title(title, style=animate)
```

- [ ] **Step 4: Run the test and confirm it passes**

Run: `.venv/bin/python -m pytest tests/plot/test_title_validation.py -v`
Expected: **9 passed.**

- [ ] **Step 5: Update the docstring**

In `plot()`'s docstring, replace the `title` entry at `plot.py:950-951` (`title : str` / `A title for the plot`) with:

```
    title : str
        A title for the plot. Must be a string; passing a list, tuple, int
        or dict raises ``TypeError`` (it used to be stringified onto the
        axes). Use ``names=`` for per-dataset legend entries, or ``labels=``
        for per-observation annotations. See ``order='serial'`` for
        per-segment titles during serial-style animations.
```

- [ ] **Step 6: Run the FULL suite (central dispatch changed)**

Run: `.venv/bin/python -m pytest -q`
Expected: `2563 passed, 13 skipped` (baseline 2554 + this task's 9). If an existing test passed a non-string title, fix the **test's** call to use a string — do not weaken the validator. (The audit found none; `tests/test_multibyte.py:711` passes a `str`.)

- [ ] **Step 7: Commit**

```bash
git add hypertools/plot/plot.py tests/plot/test_title_validation.py
git commit -m "fix(plot): reject non-string title= before the pipeline instead of stringifying it"
```

---

## Task 2: Honor `linewidth=` in animated continuous-hue line plots

`animate_plot3D` pops `linewidth` out of each per-dataset kwargs dict (`matplotlib_backend.py:1602-1606`; the 2-D twin at `2197-2201`) so it can be shared between each head line and its trail. `_apply_multicolor_animation._linewidth` (`plot.py:5150-5153`) then reads it back off `kwargs_list` — where it no longer is — and falls through to `plt.rcParams['lines.linewidth']`.

Measured today with `linewidth=[0.5, 0.5, 5.0]`, 3 datasets, `hue` of the correct length:

```
ANIM  collections: 3 x Line3DCollection label='_nolegend_' lw=1.5   <- THE BUG
      collections: 6 x Line3DCollection label='_childN'    lw=1.0   (cube planes)
      lines:       3 x Line2D lw=[0.5, 0.5, 5.0] visible=False      (hidden heads)
STATIC collections: 6 x lw=1.0 (cube planes), then 3 x lw=[0.5, 0.5, 5.0]
rcParams['lines.linewidth'] = 1.5
```

**The fix is one line in `plot.py`**, not in `matplotlib_backend.py`: read the width off the already-correct hidden head artist (`head_lines`, `plot.py:5138`). Verified by applying it: the three `_nolegend_` collections become `[0.5, 0.5, 5.0]`, and with `chemtrails=True` the six become `[0.5, 0.5, 5.0, 0.5, 0.5, 5.0]` (trails share the head's width, matching `matplotlib_backend.py:1639`). Un-popping `linewidth` in the backend instead would make the four surviving `**kwargs_list[idx]` expansions collide (`TypeError: got multiple values for keyword argument 'linewidth'`).

**Artist selector.** The head/trail collections are the only ones `_make_collection` labels `'_nolegend_'` (`plot.py:5172`); cube planes carry matplotlib's auto `_childN` labels. Verified stable both before and after driving a frame. The static control selects the last `len(datasets)` collections, since `_apply_multicolor_lines` (`plot.py:5075-5101`) removes every `Line2D` and appends its collections last.

**Known consequence, and why it is right.** After the fix, an animated hue plot with **no** explicit `linewidth=` renders at `1.0` (the backend's `pop("linewidth", 1)` default, matching the animated no-hue line artists) instead of `1.5` (rcParams). Measured today: animated no-hue lines are already `1.0` while the hue overlay was `1.5` — i.e. the overlay did **not** match the artist it replaces. The fix makes hue and no-hue animations agree. A test pins this invariant. **Maintainer decision 2026-07-29: SHIP IT** — settled, not open; recorded in the CHANGELOG (Task 9) as a visible change to existing animated hue figures. See the *Decisions (all resolved)* entry named **"Animated-hue default linewidth"**.

**Files:**
- Modify: `hypertools/plot/plot.py:5150-5153`
- Test: `tests/plot/test_animated_hue_linewidth.py` (**new file** — `tests/plot/test_matplotlib_backend_bugs.py` neither imports `pytest`/`hypertools` nor sets `Agg`, so appending would duplicate module setup at the bottom of an existing file; review T8)

**Interfaces:**
- Consumes: nothing from Task 1.
- Produces: no new public API; a behavior fix only.

- [ ] **Step 1: Write the failing test**

```python
# tests/plot/test_animated_hue_linewidth.py
"""The animated continuous-hue overlay must render at the width of the
artist it replaces (plot.py:5150-5153 read `linewidth` off kwargs_list after
matplotlib_backend.py:1602-1606 had already popped it, so every collection
fell back to rcParams['lines.linewidth'] == 1.5)."""
import matplotlib
matplotlib.use("Agg")

import numpy as np
import pytest

import hypertools as hyp


def _hue_datasets(n=3, rows=30, dims=4, seed=1):
    rng = np.random.default_rng(seed)
    return [rng.normal(size=(rows, dims)).cumsum(axis=0) for _ in range(n)]


def _hue_for(datasets):
    """One value per OBSERVATION -- plot.py:3368-3375 counts
    sum(len(xi) for xi in xform), not len(datasets[0])."""
    return np.linspace(0.0, 1.0, sum(d.shape[0] for d in datasets))


def _ax(fig):
    return [a for a in fig.axes if hasattr(a, 'zaxis')][0]


def _overlay_widths(ax):
    """Widths of the head/trail collections built by
    `_apply_multicolor_animation._make_collection`, which is the only place
    that labels a collection '_nolegend_' (plot.py:5172). The cube-plane
    collections carry matplotlib's auto '_childN' labels."""
    return [float(np.atleast_1d(c.get_linewidth())[0])
            for c in ax.collections if c.get_label() == '_nolegend_']


def _static_widths(ax, n):
    """`_apply_multicolor_lines` (plot.py:5075-5101) removes every Line2D and
    appends its collections last, so the multicolour ones are the final n."""
    return [float(np.atleast_1d(c.get_linewidth())[0])
            for c in ax.collections[-n:]]


def test_animated_continuous_hue_honors_per_dataset_linewidth():
    ds = _hue_datasets()
    fig, ani = hyp.plot(ds, '-', hue=_hue_for(ds), linewidth=[0.5, 0.5, 5.0],
                        animate=True, duration=1, frame_rate=2, show=False)
    ax = _ax(fig)
    ani._func(1, *ani._args)
    assert _overlay_widths(ax) == pytest.approx([0.5, 0.5, 5.0])


def test_animated_hue_trails_share_their_head_linewidth():
    """matplotlib_backend.py:1639 gives each trail its head's linewidth; the
    hue overlay must too (3 heads + 3 trails, in that order)."""
    ds = _hue_datasets()
    fig, ani = hyp.plot(ds, '-', hue=_hue_for(ds), linewidth=[0.5, 0.5, 5.0],
                        chemtrails=True, animate=True, duration=1,
                        frame_rate=2, show=False)
    ax = _ax(fig)
    ani._func(1, *ani._args)
    assert _overlay_widths(ax) == pytest.approx([0.5, 0.5, 5.0,
                                                 0.5, 0.5, 5.0])


def test_animated_hue_default_width_matches_the_artist_it_replaces():
    """With no linewidth=, the hidden head Line2Ds are 1.0 (the backend's
    pop default, matplotlib_backend.py:1603). The overlay must agree; it
    used to render at rcParams 1.5."""
    ds = _hue_datasets()
    fig, ani = hyp.plot(ds, '-', hue=_hue_for(ds), animate=True,
                        duration=1, frame_rate=2, show=False)
    ax = _ax(fig)
    ani._func(1, *ani._args)
    hidden = [ln.get_linewidth() for ln in ax.lines]
    assert hidden == pytest.approx([1.0, 1.0, 1.0])
    assert _overlay_widths(ax) == pytest.approx(hidden)


def test_static_continuous_hue_linewidth_still_correct():
    """Control: the bug is animation-only, so this passes before AND after."""
    ds = _hue_datasets()
    fig = hyp.plot(ds, '-', hue=_hue_for(ds), linewidth=[0.5, 0.5, 5.0],
                   show=False)
    assert _static_widths(_ax(fig), len(ds)) == pytest.approx([0.5, 0.5, 5.0])


def test_2d_animated_hue_honors_per_dataset_linewidth():
    """The 2-D twin pops linewidth at matplotlib_backend.py:2197-2201."""
    ds = _hue_datasets()
    fig, ani = hyp.plot(ds, '-', hue=_hue_for(ds), linewidth=[0.5, 0.5, 5.0],
                        ndims=2, animate=True, duration=1, frame_rate=2,
                        show=False)
    ax = fig.axes[0]
    assert not hasattr(ax, 'zaxis'), 'expected a 2-D axes'
    ani._func(1, *ani._args)
    assert _overlay_widths(ax) == pytest.approx([0.5, 0.5, 5.0])
```

- [ ] **Step 2: Run the test and confirm it fails**

Run: `.venv/bin/python -m pytest tests/plot/test_animated_hue_linewidth.py -v`

Expected: 5 collected. `test_static_continuous_hue_linewidth_still_correct` PASSES (the control). The four animated tests FAIL:
- `..._honors_per_dataset_linewidth`: `assert [1.5, 1.5, 1.5] == approx([0.5, 0.5, 5.0])`
- `..._trails_share_their_head_linewidth`: `assert [1.5]*6 == approx([0.5, 0.5, 5.0, 0.5, 0.5, 5.0])`
- `..._default_width_matches_the_artist_it_replaces`: `assert [1.5, 1.5, 1.5] == approx([1.0, 1.0, 1.0])`
- `test_2d_animated_hue_...`: `assert [1.5, 1.5, 1.5] == approx([0.5, 0.5, 5.0])`

**1 passed, 4 failed.**

- [ ] **Step 3: Read the width off the artist that carries it**

In `hypertools/plot/plot.py`, replace `_apply_multicolor_animation`'s `_linewidth` (currently `plot.py:5150-5153`):

```python
    def _linewidth(i):
        # the hidden head artist already carries the caller's linewidth=:
        # `animate_plot3D`/`animate_plot2D` pop it out of kwargs_list ONCE per
        # dataset (matplotlib_backend.py:1602-1606 / :2197-2201, so it cannot
        # also ride along in **kwargs_list[idx] and collide) and pass it
        # explicitly to ax.plot. Reading it back off kwargs_list here found
        # nothing and silently fell through to rcParams['lines.linewidth'],
        # so every animated multicolour collection rendered at 1.5 regardless
        # of what the caller asked for. Reading the artist also guarantees the
        # overlay always matches the artist it replaces.
        if i < len(head_lines):
            return head_lines[i].get_linewidth()
        tkwargs = kwargs_list[i] if i < len(kwargs_list) else {}
        return (tkwargs.get('linewidth')
                or plt.rcParams['lines.linewidth'])
```

`head_lines` is defined immediately above at `plot.py:5138` (`head_lines = list(ax.lines[:n])`), so no reordering is needed. `matplotlib_backend.py` is **not** modified by this task.

- [ ] **Step 4: Run the test and confirm it passes**

Run: `.venv/bin/python -m pytest tests/plot/test_animated_hue_linewidth.py -v`
Expected: **5 passed.**

- [ ] **Step 5: Run the FULL suite**

Run: `.venv/bin/python -m pytest -q`
Expected: `2568 passed, 13 skipped`. Pay particular attention to `tests/test_animation_hue.py`, `tests/test_animation_styles.py` and `tests/test_2d_animation.py`; any test asserting an animated-hue width of `1.5` with no explicit `linewidth=` is asserting the bug and must be updated to `1.0` with a comment pointing here.

- [ ] **Step 6: Commit**

```bash
git add hypertools/plot/plot.py tests/plot/test_animated_hue_linewidth.py
git commit -m "fix(plot): animated hue overlay uses the head artist's linewidth, not rcParams"
```

---

## Task 3: Make `animate='morph'` refuse to hang instead of hanging

`hyp.plot(shapes, animate='morph')` on the built-in zoo shapes does not finish: `hyp.load` returns 30135–36022 points for bunny/cube/sphere/vase and the point matching is a Hungarian assignment (`scipy.optimize.linear_sum_assignment`) costing roughly O(n³) with an n×n float64 cost matrix.

Measured here (`sample_and_match_clouds`, 2 structured clouds, uncapped):

```
n=1000   0.10 s   cost matrix 0.008 GB
n=2000   0.64 s   cost matrix 0.032 GB
n=4000   4.99 s   cost matrix 0.128 GB
```

(The review measured slower constants on its own data — `0.41 / 3.02 / 25.82 s` — but the same ~O(n³) scaling. Cost is data-dependent; both agree.) Extrapolating from either table, 30 000 points is tens of minutes to hours and a **7.2 GB** cost matrix.

**The behaviour is chosen by a new `simplify=` flag (maintainer decision, 2026-07-29).** `morph.py:17-24` records an explicit maintainer guarantee — *"Every dataset now keeps its FULL point count instead ... **No real data point is ever dropped**"* — restated at `plot.py:1516-1518`, and encoded by `tests/test_morph_animation.py:121-131 ::test_default_is_uncapped_target_is_largest_cloud`. v2 preserved that guarantee absolutely, by raising. The maintainer's resolution keeps the *choice* explicit but flips the *default* toward "still renders":

| situation | `simplify=True` (default) | `simplify=False` |
|-|-|-|
| largest morphing cloud **≤ `MORPH_SAMPLES_REQUIRED_ABOVE`** | **no-op** — no cap, no warning, no message, no behaviour change whatsoever | identical no-op |
| largest morphing cloud **> the cap**, no explicit `morph_samples=` | **silently** downsample to the cap. **No `warnings.warn`. No `print`.** | **raise**, with a message that names the measured cost and explicitly suggests `simplify=True` |
| explicit `morph_samples=` passed | `simplify` never engages — the caller already chose | same |

The title of this task still holds: `animate='morph'` never hangs — it now either **simplifies** or **refuses**, and which one is the caller's documented choice. Silent means silent: hanging is worse than approximating, and the maintainer was explicit that the default path must not nag. The uncapped morph over the built-in zoo was **killed at 10 minutes**; `morph_samples=2000` renders the same call in **8.2 s** (`notes/audit/PLAN.md:260`, row B3). The threshold is the size the docstring already calls out (`plot.py:1519-1522`: *"`morph_samples` is RECOMMENDED for clouds larger than ~2000 points"*). Every gallery example that morphs the zoo already passes `morph_samples=` explicitly (`examples/plot_shape_morph.py:73`, `examples/animate_morph_zoo.py:96`, `examples/animate_surface_morph.py:122`), so no example changes behaviour either way. See the *Decisions (all resolved)* entry named **"`morph_samples` above 2000"**.

**Why the raise carries the message instead of printing it first.** The maintainer wrote *"print an informative message with a suggestion to set simplify=True and then raise an exception"*. That is implemented as **one `raise` whose message is that informative message** — not a bare `print()` followed by a `raise`. A raised exception's message is already surfaced to the user, so printing it first duplicates the text and writes to stdout from library code, which nothing else in `plot.py` does. Stated here so the choice is visible and reversible (see *Revision note (v3)* for the one-line switch).

**Scope `simplify=` honestly.** Today it governs **morph tractability only**. It is documented as such — "controls whether hypertools may downsample to keep an `animate='morph'` render tractable" — not as a general "downsample everything" switch. If a later release gives it a second meaning, that is a docstring change made deliberately, not an implied promise cashed in.

**Hook point.** There is no "where `morph_samples` is resolved" — it is validated at `plot.py:2264-2274` and passed verbatim to the backends at `plot.py:4239` (plotly) and `plot.py:4324` (matplotlib). The row counts of the morphing datasets are only knowable from `xform` **and** `morph_tags`, i.e. after `plot.py:3653`. The tractability check therefore goes immediately after the morph ndims check that already lives there (`plot.py:3658-3663`), before `resolve_morph_rotations` at `3673`. The *type* check on `simplify` itself needs no data and so goes at `plot.py:2231` with the other raw-argument validation (Contract 8).

**Files:**
- Modify: `hypertools/plot/plot.py` (`simplify=` parameter + fail-fast validation at `plot.py:2231`; constant + guard after `plot.py:3663`; docstring at `plot.py:1512-1522`)
- Modify: `hypertools/plot/morph.py` (module docstring `:17-24` — the guarantee becomes conditional)
- Test: `tests/plot/test_morph_samples_guard.py` (create)

**Interfaces:**
- Produces: module constant `MORPH_SAMPLES_REQUIRED_ABOVE = 2000` in `hypertools/plot/plot.py`.
- Produces: public `plot(..., simplify=True)`.

- [ ] **Step 1: Write the failing test**

```python
# tests/plot/test_morph_samples_guard.py
import matplotlib
matplotlib.use("Agg")

import time

import numpy as np
import pytest

import hypertools as hyp
from hypertools.plot.plot import MORPH_SAMPLES_REQUIRED_ABOVE


def _clouds(n_points, seed=0):
    rng = np.random.default_rng(seed)
    return [rng.normal(size=(n_points, 3)) + off for off in (0.0, 5.0)]


def test_threshold_constant_is_2000():
    """Matches the ~2000-point recommendation already in plot()'s
    morph_samples docstring (plot.py:1519-1522)."""
    assert MORPH_SAMPLES_REQUIRED_ABOVE == 2000


def test_simplify_false_over_the_threshold_raises_naming_simplify():
    """Would otherwise be a >20-minute pytest-timeout kill, not an error.

    The message must name the escape hatch, not just the problem: the
    maintainer asked for "an informative message with a suggestion to set
    simplify=True", carried BY the exception (see the plan's Task 3 prose).
    """
    start = time.monotonic()
    with pytest.raises(ValueError, match='simplify=True'):
        hyp.plot(_clouds(12000), '.', animate='morph', simplify=False,
                 duration=1, frame_rate=2, show=False)
    assert time.monotonic() - start < 60, 'the guard must fire before matching'


def test_the_error_reports_the_actual_cloud_size_and_names_morph_samples():
    with pytest.raises(ValueError, match='12000'):
        hyp.plot(_clouds(12000), '.', animate='morph', simplify=False,
                 duration=1, frame_rate=2, show=False)
    with pytest.raises(ValueError, match='morph_samples'):
        hyp.plot(_clouds(12000), '.', animate='morph', simplify=False,
                 duration=1, frame_rate=2, show=False)


def test_default_simplify_downsamples_silently_above_the_threshold():
    """The DEFAULT path above the cap: it renders, it is capped at
    MORPH_SAMPLES_REQUIRED_ABOVE, and it says NOTHING.

    The maintainer was explicit that simplify=True must not warn. Any
    warnings.warn or print added here is a contract violation, so assert the
    absence of BOTH -- and assert the plot actually drew, so a silent
    no-render cannot pass this test.
    """
    import warnings
    start = time.monotonic()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        fig, ani = hyp.plot(_clouds(12000), '.', animate='morph',
                            duration=1, frame_rate=2, show=False)
    assert caught == [], f'simplify=True must be silent; got {caught}'
    assert time.monotonic() - start < 60, 'the cap must apply before matching'
    ani._func(0, *ani._args)
    ax = [a for a in fig.axes if hasattr(a, 'zaxis')][0]
    drawn = max(len(ln.get_data_3d()[0]) for ln in ax.lines)
    assert 0 < drawn <= MORPH_SAMPLES_REQUIRED_ABOVE


def test_default_simplify_prints_nothing_above_the_threshold(capsys):
    """"Silently" also means nothing on stdout/stderr: the informative
    message belongs to the simplify=False raise, not to this path."""
    hyp.plot(_clouds(12000), '.', animate='morph',
             duration=1, frame_rate=2, show=False)
    captured = capsys.readouterr()
    assert captured.out == '' and captured.err == ''


def test_simplify_is_a_no_op_below_the_threshold():
    """Below the cap, `simplify` has NO effect whatsoever -- this pins the
    ordinary-morph default path as untouched by this task.

    All three spellings must draw the same point counts, and none may warn.
    """
    import warnings
    drawn = {}
    for label, kwargs in (('default', {}),
                          ('true', {'simplify': True}),
                          ('false', {'simplify': False})):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            fig, ani = hyp.plot(_clouds(300), '.', animate='morph',
                                duration=1, frame_rate=2, show=False,
                                **kwargs)
        assert caught == [], f'{label} warned below the cap: {caught}'
        ani._func(0, *ani._args)
        ax = [a for a in fig.axes if hasattr(a, 'zaxis')][0]
        assert len(ax.lines) >= 1
        drawn[label] = sorted(len(ln.get_data_3d()[0]) for ln in ax.lines)
    assert drawn['default'] == drawn['true'] == drawn['false']


def test_clouds_at_or_below_the_threshold_keep_every_point():
    """The morph.py:17-24 full-sample guarantee holds unconditionally below
    the bar: no cap, no warning, and every one of the 300 points kept."""
    import warnings
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        fig, ani = hyp.plot(_clouds(300), '.', animate='morph',
                            duration=1, frame_rate=2, show=False)
    assert not [w for w in caught if 'morph_samples' in str(w.message)]
    ani._func(0, *ani._args)
    ax = [a for a in fig.axes if hasattr(a, 'zaxis')][0]
    assert len(ax.lines) >= 1
    assert max(len(ln.get_data_3d()[0]) for ln in ax.lines) == 300


def test_non_boolean_simplify_is_rejected_fail_fast():
    """Validated at plot.py:2231 with the other raw-argument checks
    (Contract 8), so it fires before the analyze/reduce pipeline."""
    with pytest.raises(TypeError, match='simplify'):
        hyp.plot(_clouds(50), '.', animate='morph', simplify='yes',
                 duration=1, frame_rate=2, show=False)


def test_explicit_morph_samples_is_respected_above_the_threshold():
    """The explicit opt-in still works and still draws. `simplify` never
    engages when the caller has already chosen a cap, so this holds for
    BOTH values of the flag."""
    for kwargs in ({}, {'simplify': False}):
        fig, ani = hyp.plot(_clouds(12000), '.', animate='morph',
                            morph_samples=400, duration=1, frame_rate=2,
                            show=False, **kwargs)
        ani._func(0, *ani._args)
        ax = [a for a in fig.axes if hasattr(a, 'zaxis')][0]
        drawn = max(len(ln.get_data_3d()[0]) for ln in ax.lines)
        assert 0 < drawn <= 400


def test_only_morph_tagged_datasets_are_measured():
    """A per-dataset animate list (plot.py:480-505) morphs only the tagged
    datasets, so a huge UNTAGGED backdrop must not trip the guard.

    Driven with simplify=False so the guard is in its RAISING mode: under the
    default it would not raise regardless, which would make this vacuous.
    """
    rng = np.random.default_rng(0)
    small = [rng.normal(size=(200, 3)) + off for off in (0.0, 5.0)]
    big_backdrop = rng.normal(size=(12000, 3)) + 10.0
    hyp.plot(small + [big_backdrop], '.',
             animate=['morph', 'morph', None], simplify=False,
             duration=1, frame_rate=2, show=False)


def test_plotly_backend_applies_the_same_guard():
    """Backend parity: the check lives in plot.py, above both dispatches
    (plot.py:4239 plotly / plot.py:4324 matplotlib), so both the raise and
    the silent cap behave identically."""
    pytest.importorskip('plotly')
    hyp.set_interactive_backend('plotly')
    try:
        with pytest.raises(ValueError, match='simplify=True'):
            hyp.plot(_clouds(12000), '.', animate='morph', simplify=False,
                     duration=1, frame_rate=2, show=False)
        import warnings
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            hyp.plot(_clouds(12000), '.', animate='morph',
                     duration=1, frame_rate=2, show=False)
        assert caught == []
    finally:
        hyp.set_interactive_backend('matplotlib')
```

- [ ] **Step 2: Run the test and confirm it fails**

Run: `.venv/bin/python -m pytest tests/plot/test_morph_samples_guard.py -v`

Expected: collection FAILS with `ImportError: cannot import name 'MORPH_SAMPLES_REQUIRED_ABOVE' from 'hypertools.plot.plot'`. (`pytest-timeout` 2.4.0 is installed and `pyproject.toml` sets `timeout = 1200`, so if you comment out the import to see the individual failures, every 12000-point test is a 20-minute kill rather than a true hang — including the `simplify=True` ones, which have no cap to apply yet.)

- [ ] **Step 3: Implement the flag and the guard**

In `hypertools/plot/plot.py`, next to the other module constants (e.g. beside `_STATIC_LINE_TARGET_VERTICES`, `plot.py:89`):

```python
#: Largest morphing-cloud size `animate='morph'` will accept without an
#: explicit `morph_samples=`. The one-to-one point matching is a Hungarian
#: assignment (`scipy.optimize.linear_sum_assignment`) over an n x n float64
#: cost matrix, costing roughly O(n^3): measured 0.10 s / 0.64 s / 4.99 s at
#: n = 1000 / 2000 / 4000, so the built-in zoo shapes (~30k points each,
#: 7.2 GB of cost matrix) do not finish in any usable time (measured: killed
#: at 10 min; `morph_samples=2000` renders the same call in 8.2 s). Above
#: this size `simplify=True` (the default) downsamples to it SILENTLY, and
#: `simplify=False` raises instead. Below it, nothing happens at all.
MORPH_SAMPLES_REQUIRED_ABOVE = 2000
```

Add `simplify=True` to `plot()`'s signature, and validate it fail-fast at `plot.py:2231` alongside Task 1's `_validate_title` call (Contract 8) — it needs no data, so it must not wait for the pipeline:

```python
    if not isinstance(simplify, bool):
        raise TypeError(
            f"simplify must be True or False, not {type(simplify).__name__}. "
            "It controls whether hypertools may downsample to keep an "
            "animate='morph' render tractable; it does not downsample "
            "anything else.")
```

Then, in `plot()`, immediately after the morph ndims check that ends at `plot.py:3663` and before `resolve_morph_rotations` at `plot.py:3673`:

```python
    # animate='morph' tractability guard: `morph_tags` marks which FINAL
    # (post cluster/hue-reshape) datasets join the morph sequence, so an
    # untagged static backdrop of any size is irrelevant here. An explicit
    # morph_samples= means the caller already chose, so `simplify` never
    # engages.
    if morph_tags is not None and morph_samples is None:
        _morph_sizes = [int(np.asarray(xform[i]).shape[0])
                        for i, _tagged in enumerate(morph_tags) if _tagged]
        _largest = max(_morph_sizes)
        if _largest > MORPH_SAMPLES_REQUIRED_ABOVE:
            if simplify:
                # SILENT by maintainer decision (2026-07-29): no warning, no
                # print. Hanging is worse than approximating, and the caller
                # who wants the guarantee back has simplify=False.
                morph_samples = MORPH_SAMPLES_REQUIRED_ABOVE
            else:
                raise ValueError(
                    f"animate='morph' received a cloud of {_largest} points. "
                    "The one-to-one point matching is a Hungarian assignment "
                    "(~O(n^3), with an n x n cost matrix), so this does not "
                    "finish in usable time or memory: measured, the built-in "
                    "zoo shapes were still running after 10 minutes, while "
                    f"morph_samples={MORPH_SAMPLES_REQUIRED_ABOVE} renders "
                    "the same call in 8.2 s. Set simplify=True (the default) "
                    "to let hypertools downsample to "
                    f"{MORPH_SAMPLES_REQUIRED_ABOVE} points per cloud "
                    "automatically, or pass morph_samples=<int> to choose "
                    "the cap yourself. With simplify=False and no "
                    "morph_samples=, every dataset keeps its full point "
                    "count and no real data point is ever dropped.")
```

Note there is no `warnings.warn` and no `print` on the `simplify` branch, and the `raise` carries the informative message itself. Both are contract, not style: `test_default_simplify_downsamples_silently_above_the_threshold` and `test_default_simplify_prints_nothing_above_the_threshold` fail if either is added.

- [ ] **Step 4: Run the test and confirm it passes**

Run: `.venv/bin/python -m pytest tests/plot/test_morph_samples_guard.py -v`
Expected: **11 passed.**

- [ ] **Step 5: Update `plot()`'s docstring**

In the `morph_samples` entry of `plot()`'s docstring (`plot.py:1512-1522`), replace the "Default `None`: no cap" sentence and the "RECOMMENDED" sentence with:

```
        Default `None`: no cap -- every dataset keeps its full point count,
        and the target count is simply the largest dataset's own size (no
        real data point is ever dropped; see `hypertools.plot.morph`). The
        Hungarian assignment's cost is roughly ``O(n^3)`` in the (post-cap)
        target point count, so above 2000 points per cloud an uncapped morph
        is intractable: with the default ``simplify=True`` hypertools caps it
        at 2000 for you, and with ``simplify=False`` it raises ``ValueError``
        naming this parameter rather than appearing to hang. Pass
        ``morph_samples=1000`` (or whatever cap you want) to choose for
        yourself; an explicit value always wins over ``simplify``. Measured
        matching cost: 0.10 s at 1000 points, 0.64 s at 2000, 4.99 s at
        4000; the built-in zoo shapes (~30k points) would need a 7.2 GB cost
        matrix and were still running after 10 minutes.
```

and add a new `simplify` entry beside it:

```
    simplify : bool, default True
        Whether hypertools may silently downsample to keep a render
        tractable. Today this governs ``animate='morph'`` **only**: a morph
        over clouds larger than 2000 points is downsampled to 2000 with no
        warning (see ``morph_samples``), because the alternative is a plot
        that never appears. Pass ``simplify=False`` to get an explanatory
        ``ValueError`` instead, so that no real data point is ever dropped
        without you asking. Below the threshold, and whenever you pass
        ``morph_samples=`` yourself, ``simplify`` does nothing at all.
```

- [ ] **Step 6: Update the guarantee in `morph.py`'s own docstring**

`hypertools/plot/morph.py:17-24` states the guarantee **absolutely** — *"No real data point is ever dropped."* — and after this task that is no longer true by default. A stale in-source guarantee contradicting shipped behaviour is exactly the defect class this plan set keeps catching, so it is fixed in the same commit. Replace the sentence *"No real data point is ever dropped."* with:

```
No real data point is dropped by the padding step itself. Whether one is
dropped EARLIER, by sampling, is the caller's documented choice: with an
explicit ``morph_samples=``, or with ``simplify=True`` (the default) over
clouds larger than ``plot.MORPH_SAMPLES_REQUIRED_ABOVE`` = 2000 points,
each cloud is first downsampled to that cap -- silently, because an
uncapped Hungarian match over ~30k-point clouds does not finish (measured:
still running after 10 minutes; capped at 2000 it renders in 8.2 s). With
``simplify=False`` and no ``morph_samples=``, the original guarantee holds
absolutely at any size: every dataset keeps its FULL point count, and an
intractable morph raises rather than approximating.
```

Also check `plot.py:1516-1518`, which restates the same guarantee, and give it the same conditional wording.

- [ ] **Step 7: Run the FULL suite**

Run: `.venv/bin/python -m pytest -q`
Expected: `2579 passed, 13 skipped`. `tests/test_morph_animation.py:121-131 ::test_default_is_uncapped_target_is_largest_cloud` calls `morph.sample_and_match_clouds` **directly**, below `plot()`, so it is unaffected by a `plot()`-level flag; confirm it still passes unmodified. If any existing test drives an *uncapped* `plot(..., animate='morph')` over more than 2000 points and asserts full point counts, it is now asserting `simplify=False` behaviour — add `simplify=False` to that call rather than weakening the assertion.

- [ ] **Step 8: Commit**

```bash
git add hypertools/plot/plot.py hypertools/plot/morph.py \
        tests/plot/test_morph_samples_guard.py
git commit -m "feat(plot): simplify= governs animate='morph' tractability (silent cap by default, raise when off)"
```

---

## Task 4: Backend parity — `animate='serial'` composes with trail flags on plotly

**Maintainer directive (2026-07-28): plotly and matplotlib behaviour must be identical.** This task closes the one real gap, for the *existing* spelling `animate='serial'`, so that Task 5's `order='serial'` inherits parity for free rather than inheriting a hole.

Measured today (3 datasets × 40 rows, `duration=3, frame_rate=4` ⇒ 12 frames):

| call | matplotlib (frame 3, vertex counts) | plotly (frame 3, point counts) |
|-|-|-|
| `animate=True` | heads `[247, 247, 247]` | traces `[247, 247, 247]` |
| `animate='serial'` | heads `[657, 0, 0]` | traces `[657, 0, 0]` |
| `animate=True, chemtrails=True` | heads `[247,247,247]`, trails `[0,0,0]` | 3 head + 3 trail traces, `[247,247,247, 1,1,1]` |
| `animate='serial', chemtrails=True` | heads `[247, 0, 0]`, trails `[657, 0, 0]` | **4 traces total, no trail traces**, plus `UserWarning: animate='serial' does not support trail styles; ignoring chemtrails for datasets [0, 1, 2]` |

So the plain serial reveal is **already** byte-identical across backends; only the trail composition is missing. Three edits close it.

1. `plotly_backend.py:946-949` — trail traces are only created for `animate in (True, 'parallel')`, so a serial animation has none to update.
2. `plotly_backend.py:2820-2865` — the `elif animate == 'serial':` frame loop draws `arr[:shown]` for every dataset and never computes a comet-head or a trail.
3. `plot.py:3757-3759` — `_trail_ignoring_modes` appends `"serial"` for plotly, producing the warn-and-drop above.

The matplotlib semantics to mirror are exactly `matplotlib_backend.py:1339-1366` (3-D) / `:2087-2095` (2-D twin), and the head width comes from `window_frames` (`matplotlib_backend.py:1915-1923`).

**Files:**
- Modify: `hypertools/plot/plotly_backend.py` (`:949`, and the serial branch at `:2820-2865`), `hypertools/plot/plot.py:3757-3759`
- Test: `tests/plot/test_plotly_serial_parity.py` (create)

**Interfaces:**
- Consumes: nothing from Tasks 1–3.
- Produces: no new public API. Establishes contract #2 (backend parity) for serial + trails.

- [ ] **Step 1: Write the failing test**

```python
# tests/plot/test_plotly_serial_parity.py
"""animate='serial' must mean the same thing on both backends, trail flags
included. Measured before this task: plotly warned "does not support trail
styles" and produced a figure identical to plain serial."""
import matplotlib
matplotlib.use("Agg")

import warnings

import numpy as np
import pytest

import hypertools as hyp

pytest.importorskip('plotly')

DURATION, FRAME_RATE = 3, 4
N_FRAMES = DURATION * FRAME_RATE          # 12
PROBE_FRAME = 3


def _datasets(n=3, rows=40, dims=4, seed=0):
    rng = np.random.default_rng(seed)
    return [rng.normal(size=(rows, dims)).cumsum(axis=0) for _ in range(n)]


def _mpl_counts(**kw):
    """(head vertex counts, trail vertex counts) at PROBE_FRAME."""
    fig, ani = hyp.plot(_datasets(), '-', duration=DURATION,
                        frame_rate=FRAME_RATE, show=False, **kw)
    ax = [a for a in fig.axes if hasattr(a, 'zaxis')][0]
    ani._func(PROBE_FRAME, *ani._args)
    n = 3
    heads = [len(ln.get_data_3d()[0]) for ln in ax.lines[:n]]
    trails = [len(ln.get_data_3d()[0]) for ln in ax.lines[n:2 * n]]
    return heads, trails


def _plotly_fig(**kw):
    hyp.set_interactive_backend('plotly')
    try:
        return hyp.plot(_datasets(), '-', duration=DURATION,
                        frame_rate=FRAME_RATE, show=False, **kw)
    finally:
        hyp.set_interactive_backend('matplotlib')


def _plotly_counts(fig, n=3):
    frame = fig.frames[PROBE_FRAME]
    npts = [0 if t.x is None else len(t.x) for t in frame.data]
    return npts[:n], npts[n:2 * n]


def _alpha_of(color):
    """rgba(r,g,b,a) -> a"""
    return float(color.rsplit(',', 1)[1].rstrip(') '))


def test_plotly_serial_with_chemtrails_emits_no_ignore_warning():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        _plotly_fig(animate='serial', chemtrails=True)
    assert not [w for w in caught if 'does not support trail' in str(w.message)]


def test_plotly_serial_with_chemtrails_creates_trail_traces():
    plain = _plotly_fig(animate='serial')
    trailed = _plotly_fig(animate='serial', chemtrails=True)
    assert len(trailed.data) == len(plain.data) + 3


def test_plotly_serial_trail_traces_are_faded():
    """Same 0.3 opacity the parallel trails already use
    (plotly_backend.py:953)."""
    fig = _plotly_fig(animate='serial', chemtrails=True)
    alphas = [_alpha_of(t.line.color) for t in fig.data
              if t.line is not None and t.line.color is not None
              and t.line.color.startswith('rgba')]
    assert alphas[:3] == pytest.approx([1.0, 1.0, 1.0])
    assert alphas[3:6] == pytest.approx([0.3, 0.3, 0.3])


@pytest.mark.parametrize('flags', [
    {'chemtrails': True},
    {'precog': True},
    {'bullettime': True},
    {'chemtrails': True, 'precog': True},
])
def test_serial_trail_geometry_matches_matplotlib_frame_for_frame(flags):
    """The strong parity assertion: identical head AND trail point counts."""
    mpl_heads, mpl_trails = _mpl_counts(animate='serial', **flags)
    ply_heads, ply_trails = _plotly_counts(
        _plotly_fig(animate='serial', **flags))
    assert ply_heads == mpl_heads
    assert ply_trails == mpl_trails


def test_plain_serial_parity_is_unchanged():
    """Regression guard: the no-trail serial reveal already matched."""
    mpl_heads, _ = _mpl_counts(animate='serial')
    ply_heads, _ = _plotly_counts(_plotly_fig(animate='serial'))
    assert ply_heads == mpl_heads == [657, 0, 0]


def test_parallel_trail_parity_is_unchanged():
    mpl_heads, mpl_trails = _mpl_counts(animate=True, chemtrails=True)
    ply_heads, ply_trails = _plotly_counts(
        _plotly_fig(animate=True, chemtrails=True))
    assert ply_heads == mpl_heads
    assert ply_trails == mpl_trails


def test_spin_and_window_still_warn_and_ignore_on_both_backends():
    """Only 'serial' leaves the ignore list; 'spin'/'morph'/'window' keep the
    established warn-and-ignore behaviour (plot.py:3757-3781)."""
    for backend, setter in (('matplotlib', 'matplotlib'), ('plotly', 'plotly')):
        hyp.set_interactive_backend(setter)
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always')
                hyp.plot(_datasets(), '-', animate='window', chemtrails=True,
                         duration=DURATION, frame_rate=FRAME_RATE, show=False)
            assert [w for w in caught
                    if 'does not support trail' in str(w.message)], backend
        finally:
            hyp.set_interactive_backend('matplotlib')
```

- [ ] **Step 2: Run the test and confirm it fails**

Run: `.venv/bin/python -m pytest tests/plot/test_plotly_serial_parity.py -v`

Expected: 10 collected (6 plain defs + 4 cases from the one parametrized def; v2 stated 11 by counting the parametrized def both as a def and as its 4 cases — see *Suite arithmetic*). PASS: `test_plain_serial_parity_is_unchanged`, `test_parallel_trail_parity_is_unchanged`, `test_spin_and_window_still_warn_and_ignore_on_both_backends`. FAIL: `..._emits_no_ignore_warning` (the warning fires), `..._creates_trail_traces` (`4 != 7`), `..._trail_traces_are_faded` (only 3 rgba line colors), and all four `test_serial_trail_geometry_matches_matplotlib_frame_for_frame` cases (`[657, 0, 0] != [247, 0, 0]` heads; `[0, 0, 0] != [657, 0, 0]` trails). **3 passed, 7 failed.**

- [ ] **Step 3: Create serial trail traces**

In `hypertools/plot/plotly_backend.py`, at line 949, extend the gate:

```python
    trail_dataset_indices = [
        i for i in range(len(data))
        if chemtrails[i] or precog[i] or bullettime[i]
    ] if animate in (True, 'parallel', 'serial') else []
```

and update the comment block at `:933-943` to say the trails now cover the serial reveal too (each currently-revealing dataset traces its own trail, matching `matplotlib_backend.update_lines_serial`).

- [ ] **Step 4: Compute serial head/trail bounds per frame**

Replace the body of the `elif animate == 'serial':` branch (`plotly_backend.py:2820-2865`) so it mirrors `matplotlib_backend.py:1339-1366` exactly. The head-window width must come from the same `window_frames` quantity matplotlib uses (`matplotlib_backend.py:1915-1923`), **not** from the point-based `window` the parallel branch computes at `plotly_backend.py:2888`:

```python
    elif animate == 'serial':
        # datasets appear one at a time, each growing into place while
        # earlier ones stay fully drawn (never connected to each other).
        # Trail composition mirrors `matplotlib_backend.update_lines_serial`
        # (matplotlib_backend.py:1339-1366): the ONE dataset currently being
        # revealed leads with a short opaque comet-head and trails the rest
        # at 0.3 opacity; already-revealed datasets stay fully drawn, and
        # not-yet-started ones stay empty.
        lengths = [np.atleast_2d(a).shape[0] for a in data]
        total_points = sum(lengths)
        starts = np.concatenate([[0], np.cumsum(lengths)[:-1]])
        has_trails = n_trail_traces > 0
        if has_trails:
            trace_indices = list(trace_indices) + list(range(
                trail_trace_start, trail_trace_start + n_trail_traces))
        # head length in FRAMES, resolved exactly as
        # matplotlib_backend.animate_plot3D does (:1915-1923)
        _focused = focused if focused is not None else tail_duration
        _uses_focus_window = (any(chemtrails) or any(precog)
                              or any(bullettime))
        _window_duration = _focused if _uses_focus_window else tail_duration
        window_frames = (1 if _window_duration == 0
                         else int(frame_rate * _window_duration))

        for k in range(n_frames):
            revealed = total_points * k / max(1, n_frames - 1)
            frame_traces = []
            trail_traces = []
            windows_by_index = {}
            window_colors_by_index = {}
            head_bounds_by_index = {}
            for idx, (arr, start) in enumerate(zip(data, starts)):
                arr = np.atleast_2d(np.asarray(arr, dtype=np.float64))
                n_pts = arr.shape[0]
                shown = int(np.clip(revealed - start, 0, n_pts))
                ct, pc, bt = chemtrails[idx], precog[idx], bullettime[idx]
                has_trail = has_trails and (ct or pc or bt)

                trail_bounds = None
                if not has_trail:
                    head_bounds = (0, shown)          # plain serial, UNCHANGED
                elif shown <= 0:
                    head_bounds = (0, 0)
                elif shown >= n_pts:
                    head_bounds = (0, n_pts)
                else:
                    w = max(1, int(round(window_frames * n_pts
                                         / max(1, total_points))))
                    head_bounds = (max(0, shown - 1 - w), shown)
                    if (ct and pc) or bt:
                        trail_bounds = (0, n_pts)              # bullettime
                    elif ct:
                        trail_bounds = (0, shown)              # chemtrails
                    else:
                        trail_bounds = (max(0, shown - 1), n_pts)  # precog
                head_bounds_by_index[idx] = head_bounds

                # surface/hue windows follow the FULL revealed portion, as in
                # matplotlib_backend.py:1379-1381 -- independent of the
                # comet-head trimming above
                windows_by_index[idx] = arr[:shown]
                cols = _window_colors(idx, 0, shown)
                if cols is not None:
                    window_colors_by_index[idx] = cols

                draw_seg = _aa_window(aa_curves, idx, *head_bounds)
                if ndims >= 3:
                    frame_traces.append(go.Scatter3d(
                        x=draw_seg[:, 0], y=draw_seg[:, 1], z=draw_seg[:, 2]))
                elif ndims == 2:
                    frame_traces.append(go.Scatter(x=draw_seg[:, 0],
                                                   y=draw_seg[:, 1]))
                else:
                    frame_traces.append(go.Scatter(
                        x=_aa_x(aa_curves[idx][1], head_bounds[0],
                                draw_seg.shape[0]),
                        y=draw_seg[:, 0]))

                if has_trail:
                    t0, t1 = trail_bounds if trail_bounds is not None else (0, 0)
                    trail = _aa_window(aa_curves, idx, t0, t1)
                    if ndims >= 3:
                        trail_traces.append(go.Scatter3d(
                            x=trail[:, 0], y=trail[:, 1], z=trail[:, 2]))
                    elif ndims == 2:
                        trail_traces.append(go.Scatter(x=trail[:, 0],
                                                       y=trail[:, 1]))
                    else:
                        trail_traces.append(go.Scatter(
                            x=_aa_x(aa_curves[idx][1], t0, trail.shape[0]),
                            y=trail[:, 0]))
                elif has_trails and idx in trail_dataset_indices:
                    # this dataset has a trail TRACE but no trail THIS frame
                    empty = np.zeros((0, max(2, min(3, ndims))))
                    trail_traces.append(
                        go.Scatter3d(x=empty[:, 0], y=empty[:, 0],
                                     z=empty[:, 0]) if ndims >= 3
                        else go.Scatter(x=empty[:, 0], y=empty[:, 0]))

            frame_traces.extend(trail_traces)
            frame_kwargs = dict(name=str(k), data=frame_traces,
                                traces=list(trace_indices))
            if ndims >= 3:
                angle = azim + 360.0 * rotations * k / n_frames
                frame_kwargs['layout'] = dict(scene_camera=dict(
                    eye=_camera_eye(elev, angle, r=_anim_zoom_r(zoom))))
            if surface_trace_indices:
                frame_kwargs['data'] = (list(frame_kwargs['data'])
                                        + _surface_frame_data(
                                            windows_by_index, angle,
                                            window_colors_by_index))
                frame_kwargs['traces'] = (list(frame_kwargs['traces'])
                                          + surface_trace_indices)
            frames.append(go.Frame(**frame_kwargs))
```

> **Ordering note for the implementer:** `trail_traces` must be appended in the same ascending `trail_dataset_indices` order `plotly_draw` used (`plotly_backend.py:946-966`), which the loop above preserves because it iterates `enumerate(data)` in order and only appends for datasets in that list.

> **Frame-hook note — this task does NOT dispatch `on_frame`.** The loop above is one of the four places `_add_animation` builds a frame (`plotly_backend.py:2729` spin, `:2819` morph, `:2865` serial, `:2975` parallel/window), and it is the natural place a plotly per-frame hook would fire. Adding it here would duplicate the dispatch, so **Task 7 Step 6a owns it** and applies the *same* three-line block to all four branches at once, after `FrameHooks`/`FrameContext` exist. Task 4 ships trails only; do not anticipate it.

- [ ] **Step 5: Stop ignoring serial trails on plotly**

In `hypertools/plot/plot.py`, replace `plot.py:3751-3759` (the comment plus the plotly special-case) with:

```python
    # 'serial' COMPOSES with the trail flags on BOTH backends
    # (chemtrails-serial / precog-serial / bullettime-serial -- the currently-
    # revealing dataset traces out its own trail; see
    # `matplotlib_backend.update_lines_serial` and the matching serial branch
    # in `plotly_backend._add_animation`), so it is never ignored. Only the
    # styles that draw no per-dataset reveal at all still ignore trails.
    _trail_ignoring_modes = ("spin", "morph", "window")
```

- [ ] **Step 6: Run the test and confirm it passes**

Run: `.venv/bin/python -m pytest tests/plot/test_plotly_serial_parity.py -v`
Expected: **10 passed.** (v2 said 11; see *Suite arithmetic*.)

- [ ] **Step 7: Document the parity**

In `plot()`'s docstring, in the `chemtrails`/`precog`/`bullettime` entries and the `animate` entry, remove any statement that serial trails are matplotlib-only and state that `animate='serial'` composes with all three trail flags on **both** backends. Do the same in `plotly_backend.plotly_draw`'s docstring (`plotly_backend.py:521-527`) and in `_add_animation`'s docstring (`plotly_backend.py:2530-2569`).

- [ ] **Step 8: Run the FULL suite**

Run: `.venv/bin/python -m pytest -q`
Expected: `2589 passed, 13 skipped`. Any existing test asserting the *"animate='serial' does not support trail styles"* warning on plotly is asserting the gap and must be updated to assert composition instead; grep for it: `grep -rn "does not support trail" tests/`.

- [ ] **Step 9: Commit**

```bash
git add hypertools/plot/plotly_backend.py hypertools/plot/plot.py \
        tests/plot/test_plotly_serial_parity.py
git commit -m "feat(plotly): serial reveal composes with chemtrails/precog/bullettime (backend parity)"
```

---

## Task 5: `order='parallel'|'serial'` as an axis orthogonal to `animate=`

`animate=` conflates a **style** (`spin`, `window`, `morph`) with an **ordering** (`parallel` vs `serial`), so `'serial'` occupies a slot `'window'`/`'morph'`/`'spin'` cannot share — which is why "the serial version of chemtrails" has no name. The *effect* already works and, after Task 4, works identically on both backends.

**Design (fixes review C5 + C6).** `order=` is resolved in two places, once each:

1. **`_resolve_order(animate, order)`**, called **fail-fast** at `plot.py:2231` (beside Task 1's `_validate_title`), returns the validated, resolved ordering string. It depends only on the raw arguments, so it needs no `n_datasets`. A per-dataset `animate` list is a morph (`plot.py:480-505`), which is determinable without resolution — that is what makes C5's unreachable gate reachable.
2. **`_resolve_animate_mode(animate, n_datasets, order)`** folds that ordering **into the returned mode**. From `plot.py:3653` onward, `animate` IS the backend mode, so all four consumers — the trail-ignore check (`3760`), plotly draw (`4214`), matplotlib draw (`4299`), and `_apply_multicolor_animation(style=animate)` (`4379`, which branches on `if style == 'serial':` at `5258`) — are correct with **no** per-site substitution. This is why v1's single `backend_mode` variable was insufficient.

**Behaviour table (all tested below):**

| call | resolved mode | note |
|-|-|-|
| `animate=True` (default order) | `True` | unchanged |
| `animate=True, order='serial'` | `'serial'` | the new spelling |
| `animate='serial'` | `'serial'` | permanent alias; implies `order='serial'` |
| `animate='serial', order='parallel'` | — | `ValueError` (contradiction) |
| `animate='morph'` | `'morph'` | inherently serial; `order` reports `'serial'` |
| `animate='morph', order='parallel'` | — | `ValueError` (contradiction) |
| `animate=['morph', None, 'morph'], order='serial'` | `'morph'` | **C5**: gate runs on the RESOLVED mode |
| `animate='spin' \| 'window'`, `order='serial'` | `'spin'`/`'window'` | `UserWarning`, ordering ignored — matches the repo's established warn-and-ignore convention at `plot.py:3760-3781`. **Settled** (maintainer, 2026-07-29): unchanged from v2, no longer an open question. See the *Decisions (all resolved)* entry named **"`order='serial'` with `animate='spin'` or `'window'`"**. |
| `order='serial'` with `animate=False` | — | `ValueError: order='serial' requires an animated plot` (mirrors the `on_frame` error shape, review G7) |
| `order=3` | — | `ValueError` that still offers the `zorder` hint (review G6) |

`order=`'s default is `None` (documented as "parallel, except where the style implies otherwise"), so an *explicit* `order='parallel'` is distinguishable from the default — which is what makes the contradiction errors possible.

**Files:**
- Modify: `hypertools/plot/plot.py` (`plot()` signature at `:553`-ish, `_resolve_animate_mode` at `:453-513`, its single caller at `:3653`, new `_resolve_order`/`_raw_animate_style`)
- Test: `tests/plot/test_order_kwarg.py` (create)

**Interfaces:**
- Consumes: `_resolve_animate_mode(animate, n_datasets)` from `plot.py:453` — **exactly one caller** (`plot.py:3653`) and no test calls it directly, so the 2-tuple → 3-tuple change is a one-line unpack update.
- Produces: `_raw_animate_style(animate)` → style; `_resolve_order(animate, order)` → `'parallel'|'serial'`; `_resolve_animate_mode(animate, n_datasets, order='parallel')` → `(mode, morph_tags, order)`. Tasks 7 and 8 both read the returned `order`.

- [ ] **Step 1: Write the failing test**

```python
# tests/plot/test_order_kwarg.py
import matplotlib
matplotlib.use("Agg")

import warnings

import numpy as np
import pytest

import hypertools as hyp

DURATION, FRAME_RATE = 3, 4
PROBE_FRAME = 3        # early: a serial reveal has only dataset 0 started


def _datasets(n=3, rows=40, dims=4, seed=0):
    rng = np.random.default_rng(seed)
    return [rng.normal(size=(rows, dims)).cumsum(axis=0) for _ in range(n)]


def _started(result, n=3, frame=PROBE_FRAME):
    """How many of the n head artists have any vertices at `frame`.

    This is the discriminator artist COUNTS cannot provide: measured at
    frame 3 of 12, parallel gives [247, 247, 247] (3 started) and serial
    gives [657, 0, 0] (1 started), while len(lines)+len(collections) is 9
    for BOTH.
    """
    fig, ani = result
    ax = [a for a in fig.axes if hasattr(a, 'zaxis')][0]
    ani._func(frame, *ani._args)
    return sum(1 for ln in ax.lines[:n] if len(ln.get_data_3d()[0]) > 0)


def _plot(**kw):
    return hyp.plot(_datasets(), '-', duration=DURATION,
                    frame_rate=FRAME_RATE, show=False, **kw)


# --- the ordering is actually honoured -------------------------------------

def test_default_order_reveals_every_dataset_together():
    assert _started(_plot(animate=True)) == 3


def test_order_serial_reveals_one_dataset_at_a_time():
    assert _started(_plot(animate=True, order='serial')) == 1


def test_explicit_order_parallel_matches_the_default():
    assert _started(_plot(animate=True, order='parallel')) == 3


def test_order_serial_matches_the_legacy_animate_serial_alias():
    assert (_started(_plot(animate=True, order='serial'))
            == _started(_plot(animate='serial')) == 1)


def test_order_serial_composes_with_chemtrails():
    """Trail artists appear AND the reveal stays serial -- artist counts
    alone cannot tell serial+chemtrails (12) from parallel+chemtrails (12)."""
    fig, ani = _plot(animate=True, order='serial', chemtrails=True)
    ax = [a for a in fig.axes if hasattr(a, 'zaxis')][0]
    ani._func(PROBE_FRAME, *ani._args)
    heads = [len(ln.get_data_3d()[0]) for ln in ax.lines[:3]]
    trails = [len(ln.get_data_3d()[0]) for ln in ax.lines[3:6]]
    assert sum(1 for h in heads if h) == 1, 'reveal must stay serial'
    assert sum(1 for t in trails if t) == 1, 'the revealing dataset trails'


def test_order_serial_matches_animate_serial_for_hue_overlays():
    """plot.py:4379 passes style=animate into _apply_multicolor_animation,
    which branches on `style == 'serial'` at plot.py:5258 to recover the
    reveal position. A one-site backend_mode substitution would desync."""
    ds = _datasets()
    hue = np.linspace(0.0, 1.0, sum(d.shape[0] for d in ds))

    def segments(**kw):
        fig, ani = hyp.plot(ds, '-', hue=hue, duration=DURATION,
                            frame_rate=FRAME_RATE, show=False, **kw)
        ax = [a for a in fig.axes if hasattr(a, 'zaxis')][0]
        ani._func(PROBE_FRAME, *ani._args)
        return [len(c._segments3d) for c in ax.collections
                if c.get_label() == '_nolegend_']

    assert segments(animate=True, order='serial') == segments(animate='serial')


# --- conflicts and errors ---------------------------------------------------

def test_conflicting_order_parallel_with_animate_serial_raises():
    with pytest.raises(ValueError, match="animate='serial'"):
        _plot(animate='serial', order='parallel')


def test_conflicting_order_parallel_with_animate_morph_raises():
    rng = np.random.default_rng(0)
    clouds = [rng.normal(size=(150, 3)) + off for off in (0.0, 4.0)]
    with pytest.raises(ValueError, match='inherently serial'):
        hyp.plot(clouds, '.', animate='morph', order='parallel',
                 duration=1, frame_rate=2, show=False)


def test_order_serial_without_animation_raises():
    with pytest.raises(ValueError, match="order='serial' requires an animated"):
        hyp.plot(_datasets(), '-', order='serial', show=False)


@pytest.mark.parametrize('bad', ['Serial', 'sequential', True, 1])
def test_invalid_order_raises(bad):
    with pytest.raises(ValueError, match="order must be"):
        _plot(animate=True, order=bad)


def test_numeric_order_still_offers_the_zorder_hint():
    """Before order= existed: TypeError "...did you mean 'zorder'?".
    That hint must survive the parameter's promotion (review G6)."""
    with pytest.raises(ValueError, match='zorder'):
        _plot(animate=True, order=3)


# --- styles with no serial analog ------------------------------------------

@pytest.mark.parametrize('style', ['spin', 'window'])
def test_serial_ordering_warns_and_is_ignored_for_spin_and_window(style):
    """Matches the established convention at plot.py:3760-3781 (warn, do not
    hard-error, when a flag has no meaning in the requested mode)."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        result = _plot(animate=style, order='serial')
    assert [w for w in caught
            if "order='serial'" in str(w.message) and style in str(w.message)]
    if style == 'window':
        assert _started(result) == 3, 'ordering ignored: still parallel-ish'


# --- C5: the list form of animate= -----------------------------------------

def test_per_dataset_morph_list_accepts_order_serial():
    """animate=['morph', None, 'morph'] resolves to 'morph' (plot.py:480-505),
    which IS serial-capable. Gating on the RAW argument would raise here."""
    rng = np.random.default_rng(0)
    clouds = [rng.normal(size=(150, 3)) + off for off in (0.0, 4.0, 8.0)]
    hyp.plot(clouds, '.', animate=['morph', None, 'morph'], order='serial',
             morph_samples=150, duration=1, frame_rate=2, show=False)


def test_morph_accepts_order_serial():
    rng = np.random.default_rng(0)
    clouds = [rng.normal(size=(150, 3)) + off for off in (0.0, 4.0)]
    hyp.plot(clouds, '.', animate='morph', order='serial',
             morph_samples=150, duration=1, frame_rate=2, show=False)


# --- backend parity ---------------------------------------------------------

def test_order_serial_is_identical_on_plotly():
    """Maintainer requirement: the same call must mean the same thing."""
    pytest.importorskip('plotly')
    fig, ani = _plot(animate=True, order='serial')
    ax = [a for a in fig.axes if hasattr(a, 'zaxis')][0]
    ani._func(PROBE_FRAME, *ani._args)
    mpl = [len(ln.get_data_3d()[0]) for ln in ax.lines[:3]]

    hyp.set_interactive_backend('plotly')
    try:
        pfig = _plot(animate=True, order='serial')
    finally:
        hyp.set_interactive_backend('matplotlib')
    ply = [0 if t.x is None else len(t.x)
           for t in pfig.frames[PROBE_FRAME].data][:3]
    assert ply == mpl


def test_order_serial_with_chemtrails_is_identical_on_plotly():
    pytest.importorskip('plotly')
    fig, ani = _plot(animate=True, order='serial', chemtrails=True)
    ax = [a for a in fig.axes if hasattr(a, 'zaxis')][0]
    ani._func(PROBE_FRAME, *ani._args)
    mpl = [len(ln.get_data_3d()[0]) for ln in ax.lines[:6]]

    hyp.set_interactive_backend('plotly')
    try:
        pfig = _plot(animate=True, order='serial', chemtrails=True)
    finally:
        hyp.set_interactive_backend('matplotlib')
    ply = [0 if t.x is None else len(t.x)
           for t in pfig.frames[PROBE_FRAME].data][:6]
    assert ply == mpl
```

- [ ] **Step 2: Run the test and confirm it fails**

Run: `.venv/bin/python -m pytest tests/plot/test_order_kwarg.py -v`

Expected: 20 collected (14 plain + 4 `test_invalid_order_raises` + 2 `test_serial_ordering_warns...`). `test_default_order_reveals_every_dataset_together` and `test_order_serial_matches_the_legacy_animate_serial_alias`'s legacy half aside, **every test that passes `order=` FAILS** with `TypeError: plot() got an unexpected keyword argument 'order'; did you mean 'zorder'?`. `test_default_order_reveals_every_dataset_together` PASSES. **1 passed, 19 failed.**

- [ ] **Step 3: Resolve ordering fail-fast, then fold it into the mode**

Add to `hypertools/plot/plot.py`, immediately before `_resolve_animate_mode` (`plot.py:453`):

```python
#: Animate STYLES that implement a dataset-by-dataset (serial) reveal.
#: Membership is tested against the RESOLVED style, never the raw argument:
#: `animate=['morph', None, 'morph']` resolves to 'morph', which is here.
_SERIAL_CAPABLE_STYLES = (True, 'parallel', 'serial', 'morph')
#: Styles that are serial by construction, so `order='parallel'` contradicts.
_INHERENTLY_SERIAL_STYLES = ('serial', 'morph')


def _raw_animate_style(animate):
    """The STYLE `animate=` names, before dataset-count-dependent resolution.

    A per-dataset list/tuple only ever tags datasets for a morph
    (`_resolve_animate_mode`, plot.py:480-505), so its style is 'morph'
    regardless of length or contents. Knowing this WITHOUT `n_datasets` is
    what lets ordering be validated fail-fast, before the pipeline.
    """
    if isinstance(animate, (list, tuple)):
        return 'morph'
    return animate


def _resolve_order(animate, order):
    """Validate and resolve ``order=`` against the requested animate STYLE.

    ``animate=`` names the STYLE ('spin'/'window'/'morph'/a parallel reveal)
    and ``order=`` names the ORDERING (all datasets at once, or one after
    another). ``animate='serial'`` predates this split and is a permanent
    alias for ``animate=True, order='serial'``.

    ``order=None`` (the default) means "whatever the style implies": parallel
    for the reveal styles, serial for 'serial'/'morph'. An EXPLICIT
    ``order='parallel'`` therefore contradicts an inherently serial style,
    and says so instead of being silently overridden.

    Returns 'parallel' or 'serial'. Called once, fail-fast, at plot.py:2231.
    """
    if order is not None and order not in ('parallel', 'serial'):
        hint = (" (for matplotlib's draw-order property, pass zorder=)"
                if isinstance(order, (int, float, np.integer, np.floating))
                and not isinstance(order, bool) else "")
        raise ValueError(
            f"order must be 'parallel' or 'serial' (or None); got "
            f"{order!r}{hint}.")

    style = _raw_animate_style(animate)

    if style in _INHERENTLY_SERIAL_STYLES:
        if order == 'parallel':
            if style == 'serial':
                raise ValueError(
                    "animate='serial' is an alias for animate=True, "
                    "order='serial', so it conflicts with order='parallel'. "
                    "Pass animate=True, order='parallel' for a parallel "
                    "reveal.")
            raise ValueError(
                "animate='morph' is inherently serial (one cloud eases into "
                "the next), so it conflicts with order='parallel'. Drop "
                "order=, or pass animate=True, order='parallel' for a "
                "parallel reveal.")
        return 'serial'

    if order is None:
        return 'parallel'
    if order == 'serial' and not style:
        raise ValueError(
            "order='serial' requires an animated plot; pass animate=True "
            "(or 'serial'/'morph') alongside it. A static plot draws every "
            "dataset at once by definition.")
    return order
```

Replace `_resolve_animate_mode`'s signature and return (`plot.py:453`, `:505`, `:512`, `:513`) so it takes and returns the ordering, folding it into the mode **after** the existing list/scalar resolution (so C5's gate sees the resolved style):

```python
def _resolve_animate_mode(animate, n_datasets, order='parallel'):
    """... (existing docstring, plus:)

    `order` is `_resolve_order`'s already-validated result. It is folded INTO
    the returned `mode`, so `animate` from plot.py:3653 onward is exactly what
    every backend and every downstream consumer should see -- the trail-ignore
    check (plot.py:3760), plotly_draw (:4214), _draw (:4299) and
    _apply_multicolor_animation(style=...) (:4379) all read that one value.
    `order` is ALSO returned, for consumers that need the ordering itself
    (FrameContext.order, per-segment titles).

    Returns
    -------
    (mode, morph_tags, order)
    """
```

- inside the list branch, replace `return "morph", tags` with `mode, morph_tags = "morph", tags` and fall through;
- replace `return "morph", [True] * n_datasets` with `mode, morph_tags = "morph", [True] * n_datasets` and fall through;
- replace the final `return animate, None` with `mode, morph_tags = animate, None`;

then append the single fold, which is the only new logic:

```python
    if order == 'serial':
        if mode in (True, 'parallel'):
            # the whole point: a serial ORDERING of a parallel STYLE is
            # exactly the existing 'serial' backend mode
            mode = 'serial'
        elif mode not in _SERIAL_CAPABLE_STYLES:
            # 'spin'/'window' have no dataset-by-dataset reveal. Warn and
            # ignore, matching the established convention for a flag with no
            # meaning in the requested mode (plot.py:3760-3781) rather than
            # introducing a new hard error class.
            warnings.warn(
                f"animate={mode!r} has no serial ordering (it does not "
                f"reveal datasets one at a time); ignoring order='serial'. "
                "Use animate=True, order='serial' for a serial reveal.",
                UserWarning,
                stacklevel=external_stacklevel(),
            )
            order = 'parallel'
    return mode, morph_tags, order
```

Add `order=None` to `plot()`'s signature immediately after `animate=False` (`plot.py:553`), resolve it fail-fast at `plot.py:2231` beside Task 1's call:

```python
    order = _resolve_order(animate, order)
```

and update the single caller at `plot.py:3653`:

```python
    animate, morph_tags, order = _resolve_animate_mode(animate, len(xform),
                                                       order=order)
```

- [ ] **Step 4: Run the test and confirm it passes**

Run: `.venv/bin/python -m pytest tests/plot/test_order_kwarg.py -v`
Expected: **20 passed.**

- [ ] **Step 5: Document the new axis**

Add an `order` entry to `plot()`'s docstring immediately after `animate`:

```
    order : {'parallel', 'serial'}, optional
        Whether animated datasets are revealed all at once ('parallel') or
        one after another ('serial'). This is ORTHOGONAL to ``animate=``,
        which names the style, so it composes with the trail flags:
        ``animate=True, order='serial', chemtrails=True`` is the serial
        version of chemtrails, and renders identically on the matplotlib and
        plotly backends. The default (``None``) means "whatever the style
        implies": parallel for the reveal styles, serial for
        ``animate='serial'`` (a permanent alias for ``animate=True,
        order='serial'``) and ``animate='morph'`` (inherently serial).
        Passing ``order='parallel'`` alongside either of those raises
        ``ValueError``. ``'spin'`` and ``'window'`` have no dataset-by-dataset
        reveal, so ``order='serial'`` warns and is ignored there.
        ``order='serial'`` without an animation raises ``ValueError``.
```

- [ ] **Step 6: Run the FULL suite (central dispatch changed)**

Run: `.venv/bin/python -m pytest -q`
Expected: `2609 passed, 13 skipped`.

- [ ] **Step 7: Commit**

```bash
git add hypertools/plot/plot.py tests/plot/test_order_kwarg.py
git commit -m "feat(plot): add order='parallel'|'serial', folded into the resolved animate mode"
```

---

## Task 6: Per-dataset `alpha=`

`hyp.plot(ds, '-', alpha=[0.1, 0.1, 1.0])` raises `TypeError: alpha must be numeric or None, not <class 'list'>` (from `matplotlib/artist.py`), because `alpha` is a `**kwargs` passthrough value and extra kwargs are explicitly never per-dataset (`plot.py:60-69`). Scalar `alpha=0.25` measured today reaches every `Line2D` correctly. This forces the weather example to re-apply `set_alpha` on every frame and the conversation example's entire recency-fade block.

**Two things v1 left undefined (review G1, G2), now specified:**

- **Precedence.** `mpl_kwargs["alpha"]` is already written as a per-dataset list by `plot.py:3056` (row-MultiIndex level fading) and `plot.py:3629` (nested-list depth fading). `plot.py:71-75` documents that internal styling wins over a same-named extra kwarg. Promoting `alpha` keeps that *outcome*: where those paths have written alpha, a user `alpha=` **warns and is ignored** — the exact shape of the existing `linewidth=` precedent at `plot.py:3045-3050`.
- **Which count.** `_validate_alpha(alpha, len(xform))` — the **final** (post cluster/hue-reshape) dataset count, the same count `surface_list`/`density_list` broadcast against at `plot.py:3637-3643`. Writing into `mpl_kwargs` at that point means `_expand_styles_to_runs` (`plot.py:231-263`) picks it up automatically for contiguous-run segmentation; its docstring at `plot.py:242-244` (which currently states that `alpha=` never reaches `mpl_kwargs`) must be corrected in the same commit.

**Backend note (verified):** plotly needs **no** change — `plotly_backend.py:776` already reads per-dataset alpha off `kwargs_list`: `color = _to_plotly_color(tkwargs.get('color'), tkwargs.get('alpha'))`.

**Files:**
- Modify: `hypertools/plot/plot.py` (signature, `_validate_alpha`, the write beside `plot.py:3637`, the `_expand_styles_to_runs` docstring at `:242-244`)
- Test: `tests/plot/test_per_dataset_alpha.py` (create)

**Interfaces:**
- Consumes: `len(xform)` (final dataset count).
- Produces: `_validate_alpha(alpha, n_datasets)` → `list[float] | None`; `alpha` accepted as scalar or per-dataset sequence.

- [ ] **Step 1: Write the failing test**

```python
# tests/plot/test_per_dataset_alpha.py
import matplotlib
matplotlib.use("Agg")

import warnings

import numpy as np
import pandas as pd
import pytest

import hypertools as hyp


def _datasets(n=3, rows=20, dims=4, seed=0):
    rng = np.random.default_rng(seed)
    return [rng.normal(size=(rows, dims)).cumsum(axis=0) for _ in range(n)]


def _ax(fig):
    return [a for a in fig.axes if hasattr(a, 'zaxis')][0]


def _alphas(fig):
    return [ln.get_alpha() for ln in _ax(fig).lines]


def test_scalar_alpha_still_applies_to_every_dataset():
    """Guards tests/test_gh206_extra_kwargs.py::test_alpha_kwarg_reaches_line
    _artists, which must keep passing after alpha leaves **kwargs."""
    fig = hyp.plot(_datasets(), '-', alpha=0.25, show=False)
    assert _alphas(fig) == pytest.approx([0.25, 0.25, 0.25])


def test_per_dataset_alpha_list():
    fig = hyp.plot(_datasets(), '-', alpha=[0.1, 0.5, 1.0], show=False)
    assert _alphas(fig) == pytest.approx([0.1, 0.5, 1.0])


def test_per_dataset_alpha_length_mismatch_raises():
    with pytest.raises(ValueError, match='alpha has 2 entries'):
        hyp.plot(_datasets(), '-', alpha=[0.1, 0.5], show=False)


def test_alpha_out_of_range_raises():
    with pytest.raises(ValueError, match='between 0 and 1'):
        hyp.plot(_datasets(), '-', alpha=[0.1, 0.5, 1.7], show=False)


def test_non_numeric_alpha_raises():
    with pytest.raises(ValueError, match='alpha'):
        hyp.plot(_datasets(), '-', alpha=['a', 'b', 'c'], show=False)


def test_per_dataset_alpha_survives_animation():
    fig, ani = hyp.plot(_datasets(), '-', alpha=[0.1, 0.5, 1.0],
                        animate=True, duration=1, frame_rate=2, show=False)
    ani._func(1, *ani._args)
    assert _alphas(fig) == pytest.approx([0.1, 0.5, 1.0])


def test_per_dataset_alpha_reaches_plotly_traces():
    """plotly_backend.py:776 already reads alpha off kwargs_list."""
    pytest.importorskip('plotly')
    hyp.set_interactive_backend('plotly')
    try:
        fig = hyp.plot(_datasets(), '-', alpha=[0.1, 0.5, 1.0], show=False)
    finally:
        hyp.set_interactive_backend('matplotlib')
    alphas = [float(t.line.color.rsplit(',', 1)[1].rstrip(') '))
              for t in fig.data
              if t.line is not None and t.line.color is not None
              and t.line.color.startswith('rgba')]
    assert alphas[:3] == pytest.approx([0.1, 0.5, 1.0])


# --- precedence (review G1) -------------------------------------------------

def _multiindex_frame(seed=0):
    idx = pd.MultiIndex.from_tuples(
        [('cond1', s) for s in range(3)] + [('cond2', s) for s in range(3)],
        names=['cond', 'subj'])
    rng = np.random.default_rng(seed)
    return pd.DataFrame(rng.normal(size=(6, 4)), index=idx)


def test_multiindex_level_fading_wins_and_says_so():
    """Mirrors the linewidth= precedent at plot.py:3045-3050: internal
    styling wins over a same-named user kwarg, with a warning."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        fig = hyp.plot(_multiindex_frame(), '-', alpha=0.9, show=False)
    assert [w for w in caught
            if 'alpha' in str(w.message) and 'MultiIndex' in str(w.message)]
    alphas = [ln.get_alpha() for ln in _ax(fig).lines]
    assert not all(a == pytest.approx(0.9) for a in alphas if a is not None)


def test_nested_list_depth_fading_wins_and_says_so():
    """plot.py:3629 writes a depth-derived alpha list for nested inputs."""
    rng = np.random.default_rng(0)
    nested = [[rng.normal(size=(10, 4)).cumsum(axis=0) for _ in range(2)],
              rng.normal(size=(10, 4)).cumsum(axis=0)]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        hyp.plot(nested, '-', alpha=0.9, show=False)
    assert [w for w in caught if 'alpha' in str(w.message)]


def test_alpha_survives_contiguous_run_segmentation():
    """A categorical hue turns N datasets into >= N runs
    (_expand_styles_to_runs, plot.py:231-263); a per-dataset alpha must be
    expanded, not length-checked against the run count."""
    ds = _datasets(n=2, rows=20)
    labels = np.array(['a'] * 10 + ['b'] * 10 + ['a'] * 10 + ['b'] * 10)
    fig = hyp.plot(ds, '-', hue=labels, alpha=[0.2, 0.8], show=False)
    alphas = [a for a in (ln.get_alpha() for ln in _ax(fig).lines)
              if a is not None]
    assert set(np.round(alphas, 6)) <= {0.2, 0.8}
    assert len(alphas) > 2, 'expected more runs than datasets'
```

- [ ] **Step 2: Run the test and confirm it fails**

Run: `.venv/bin/python -m pytest tests/plot/test_per_dataset_alpha.py -v`

Expected: 11 collected. `test_scalar_alpha_still_applies_to_every_dataset` PASSES. The list-form tests FAIL with `TypeError: alpha must be numeric or None, not <class 'list'>`; `test_alpha_out_of_range_raises`/`test_non_numeric_alpha_raises` FAIL with `TypeError` instead of `ValueError`; the two precedence tests FAIL with `DID NOT` find the warning (a scalar alpha silently loses today, with no message). **1 passed, 10 failed.**

- [ ] **Step 3: Promote `alpha` to a first-class per-dataset kwarg**

Add `alpha=None` to the `plot()` signature next to `linewidth=None` (`plot.py:523`) so it stops flowing through `**kwargs`, and add the validator beside `_validate_title`:

```python
def _validate_alpha(alpha, n_datasets):
    """`alpha=` is a scalar applied to every dataset, or one value per
    dataset. Returns a list of `n_datasets` floats, or None.

    Promoted out of the GH #206 `**kwargs` passthrough (where a list raised
    matplotlib's bare "alpha must be numeric or None") so callers can fade
    backdrops behind a highlighted dataset without re-applying `set_alpha`
    on every frame. `n_datasets` is the FINAL (post cluster/hue-reshape)
    count -- the same one surface=/density= broadcast against
    (plot.py:3637-3643) -- and `_expand_styles_to_runs` (plot.py:231-263)
    widens the result to run length when hue/cluster segmentation applies.
    """
    if alpha is None:
        return None
    values = [alpha] if np.isscalar(alpha) else list(alpha)
    try:
        values = [float(a) for a in values]
    except (TypeError, ValueError):
        raise ValueError(
            f"alpha must be a number, or one number per dataset; got "
            f"{alpha!r}.") from None
    if len(values) == 1:
        values = values * n_datasets
    if len(values) != n_datasets:
        raise ValueError(
            f"alpha has {len(values)} entries but there are {n_datasets} "
            "datasets to plot; pass a single value to apply it to every "
            "dataset, or one value per dataset.")
    for a in values:
        if not (0.0 <= a <= 1.0):
            raise ValueError(f"alpha values must be between 0 and 1; got {a}.")
    return values
```

Write it into `mpl_kwargs` immediately before the `surface_list` broadcast at `plot.py:3637`, i.e. after both internal alpha writers (`plot.py:3056` and `plot.py:3629`) have run, so precedence is explicit and in one place:

```python
    # alpha= (1.1): a first-class per-dataset style, resolved against the
    # FINAL dataset count like surface=/density= below. Internal per-trace
    # alpha (MultiIndex level fading at plot.py:3056, nested-list depth
    # fading at plot.py:3629) still wins -- the documented rule at
    # plot.py:71-75 -- but says so rather than losing silently, exactly as
    # the MultiIndex branch already does for linewidth= (plot.py:3045-3050).
    if alpha is not None:
        if "alpha" in mpl_kwargs:
            warnings.warn(
                "this input assigns alpha internally (MultiIndex level "
                "fading, or nested-list depth fading); ignoring alpha=. "
                "Flatten the input if you want to set alpha yourself.",
                UserWarning, stacklevel=external_stacklevel())
        else:
            mpl_kwargs["alpha"] = _validate_alpha(alpha, len(xform))
```

- [ ] **Step 4: Run the test and confirm it passes**

Run: `.venv/bin/python -m pytest tests/plot/test_per_dataset_alpha.py -v`
Expected: **10 passed.** (v2 said 11; see *Suite arithmetic*.)

- [ ] **Step 5: Document it**

Add an `alpha` entry to `plot()`'s docstring beside `linewidth`:

```
    alpha : float or list of float, optional
        Opacity in [0, 1], either one value for every dataset or one value
        per dataset (e.g. ``alpha=[0.1, 0.1, 1.0]`` to fade two backdrops
        behind a highlighted third). Inputs that assign alpha internally --
        a row MultiIndex (per-level fading) or a nested list (per-depth
        fading) -- keep their own values and warn that ``alpha=`` was
        ignored, matching how ``linewidth=`` already behaves there.
```

And correct `_expand_styles_to_runs`'s docstring at `plot.py:242-244`, which currently states that `alpha=` never reaches `mpl_kwargs`:

```
    ``alpha`` joined this set in 1.1 (it used to be a generic ``**kwargs``
    passthrough applied verbatim per trace); any REMAINING generic
    passthrough value still never reaches `mpl_kwargs` and is unaffected here.
```

- [ ] **Step 6: Run the FULL suite (central dispatch changed)**

Run: `.venv/bin/python -m pytest -q`
Expected: `2619 passed, 13 skipped`. Specifically confirm `tests/test_gh206_extra_kwargs.py::...::test_alpha_kwarg_reaches_line_artists` (`:71-78`) still passes; its *placement* in the passthrough suite is now misleading, so add a one-line comment there pointing at the named parameter.

- [ ] **Step 7: Commit**

```bash
git add hypertools/plot/plot.py tests/plot/test_per_dataset_alpha.py \
        tests/test_gh206_extra_kwargs.py
git commit -m "feat(plot): accept per-dataset alpha= alongside color=/linewidth=, with explicit precedence"
```

---

## Task 7: Public per-frame hook on `HyperAnimation`

Four of the five new gallery examples monkeypatch matplotlib's private `FuncAnimation._func` (`examples/animate_conversation.py`, `animate_market_forecast.py`, `animate_morph_zoo.py`, `animate_weather_decades.py`), and three read `ani._args[...]` to recover the drawn arrays and artists. The conversation example **re-derives hypertools' own reveal formula by hand** — the real one is at `matplotlib_backend.py:1316-1318` — and its comments document off-by-one bugs hit while doing so.

**Design (fixes review C7, G5, T4, T7).**

- **One shared registry, created before the closure exists.** `plot()` builds a `FrameHooks` object and threads it into `_draw`, whose nested updaters close over it. `HyperAnimation.__new__` **adopts that same object** — it never creates a list. That is the whole of C7: v1's `self._frame_callbacks = []` in `__new__` produced a fresh, unreferenced list, so a callback registered after construction could never fire.
- **One dispatch site, installed last.** `_apply_multicolor_animation` wraps `line_ani._func` at `plot.py:5289`. The hook dispatcher is installed **after** that (and after any other wrapping), so a hue animation's callbacks observe the final, re-sliced collections rather than the pre-multicolor state. The updaters only *record* state; they never fire callbacks.
- **One reveal-schedule implementation.** `serial_reveal_counts` / `serial_current_index` become module-level helpers in `matplotlib_backend.py`, used by `update_lines_serial` (3-D), `update_lines_serial_2d`, and the recorded frame state. `plot.py:5265-5269`'s third copy inside `_apply_multicolor_animation` is replaced by a call to the same helper.
- **2-D is covered.** All seven updaters record state: `update_lines_parallel` (`:1118`), `update_lines_spin` (`:1229`), `update_lines_serial` (`:1283`), `update_morph` (`:1398`), `update_lines_parallel_2d` (`:2009`), `update_lines_serial_2d` (`:2048`), `update_morph_2d` (`:2107`). Tests drive both dimensionalities.
- **`return_model=True`.** That path hands back the **raw** `FuncAnimation` and never constructs a `HyperAnimation` (`plot.py:4584-4586`, `4612-4614`), so `.on_frame()` is unavailable there. `on_frame=` passed to `plot()` still fires (the dispatcher is on `line_ani._func`). Both facts are tested and documented.
- **Both backends (maintainer decision, 2026-07-29 — v2's `NotImplementedError` is REMOVED).** v2 asserted that "a plotly animation is precomputed JSON played by a browser; there is no Python frame loop to call back into". **That premise is false.** `_add_animation` (`plotly_backend.py:2517`) builds every frame in a **Python loop at build time** — `frames = []` (`:2601`), then `frames.append(go.Frame(**frame_kwargs))` at `:2729` (spin), `:2819` (morph), `:2865` (serial) and `:2975` (the `else:` parallel/window branch at `:2866`). What plotly lacks is a Python loop **during playback**: once `fig.frames` is populated the browser plays it. So `on_frame=` ships on both backends and Contract 2 has no exception left.
- **The call SCHEDULES differ; the CONTEXT METADATA does not.** `on_frame` is called **once per frame**, with a `FrameContext` carrying the frame index and that frame's data, at the point that frame is produced by each backend's natural loop:
  - **matplotlib — at render time.** `FuncAnimation(..., blit=False)` (`matplotlib_backend.py:1935`, `:1957`, `:1968`) fires its updater lazily during interactive playback and eagerly when saving (`animate.py:116`; the gif/apng/video writers save every frame). A given frame index **may therefore be called more than once** across a loop or a save.
  - **plotly — exactly once per frame index**, at build time inside `_add_animation`.
  - **Therefore `on_frame` MUST be deterministic and idempotent for a given `FrameContext`.** The binding sentence, used verbatim in the docstring (Step 8), the guide (Task 9) and the CHANGELOG:

    > Callbacks must be deterministic and idempotent for a given frame context. They must not depend on call count, call order, wall-clock time, or accumulated external state.

    **Never call this "purity."** `on_frame` exists to mutate artists; a pure callback would do nothing. Idempotence — not absence of effects — is what makes matplotlib's possible re-delivery of a frame index indistinguishable from plotly's single delivery. This costs nothing today: verified 2026-07-30 across all four `_func`-monkeypatching gallery examples, **no per-frame wrapper accumulates**. `examples/animate_morph_zoo.py` does `label.set_text(shape_title(frame))` — a mutation, and idempotent; `animate_conversation.py` and `animate_weather_decades.py` read live artist state **0** times inside their wrappers; `animate_market_forecast.py`'s only **2** artist reads are one-time setup calibration (the uppercase `SLOPE`/`BLO`/`BHI` constants computed after a full reveal), not per-frame. Where an example genuinely needs a running quantity it **precomputes at module level and indexes by frame** (`animate_market_forecast.py:255` builds `ACC`; the wrapper at `:323` only reads `ACC[min(num, total - 1)]`). Teach that idiom in the guide.
  - **Context-metadata parity is the testable guarantee — output parity is NOT claimed**, and asserting it would be asserting something false. `test_on_frame_context_metadata_parity_across_backends` pins the former: for the same `on_frame`, both backends yield the same backend-independent `FrameContext` fields per frame index. `figure`/`axes`/`artists` are backend-**native** (matplotlib `Figure`/`Axes`/artists, or the `go.Figure` and that frame's traces), so a callback that mutates them is **not source-compatible across backends**; they are documented as such, not faked, and `test_plotly_frame_context_carries_backend_native_objects` pins that. What each backend separately guarantees is that a mutation the callback performs is **retained in that backend's own rendered frame** — one test per backend, Step 1b. See the *Decisions (all resolved)* entry named **"`FrameContext` exposes backend-native objects"**.

**Files:**
- Create: `hypertools/plot/animation_context.py`
- Modify: `hypertools/plot/hyper_animation.py`, `hypertools/plot/plot.py`, `hypertools/plot/matplotlib_backend.py`, `hypertools/plot/plotly_backend.py` (Step 6a — the four frame-build sites)
- Modify (Step 6b, public surface — atomic): `hypertools/__init__.py` (import + `__all__`), `docs/api.rst` (Plot autosummary)
- Test: `tests/plot/test_on_frame_hook.py` (create), `tests/test_codeorg_licensing_audit_fixes.py:295-300` (modify the hardcoded `documented` set — **required**, or Step 6b turns the suite red)

**Interfaces:**
- Consumes: `order` from `_resolve_animate_mode` (Task 5).
- Produces:
  - `FrameContext` — frozen dataclass, fields `frame`, `n_frames`, `figure`, `axes`, `artists`, `datasets`, `style`, `order`, `current_index`, `current_fraction`, `revealed_counts`, `segment_index`, `segment_kind`. The three sequence fields are **tuples** on every backend and style (`revealed_counts` is `None` or a tuple), canonicalized by `__post_init__` — index and iterate them, never `.append`.
  - `FrameHooks` — `.callbacks` (list), `.record(**state)`, `.dispatch(figure, axes)`.
  - `matplotlib_backend.serial_reveal_counts(lengths, num, total_frames)`, `serial_current_index(counts, lengths)`.
  - `plot(..., on_frame=callable)`; `HyperAnimation.on_frame(callable)`.

- [ ] **Step 1: Write the failing test**

```python
# tests/plot/test_on_frame_hook.py
import matplotlib
matplotlib.use("Agg")

import numpy as np
import pytest

import hypertools as hyp
from hypertools.plot.animation_context import FrameContext


def _datasets(n=3, rows=20, dims=4, seed=0):
    rng = np.random.default_rng(seed)
    return [rng.normal(size=(rows, dims)).cumsum(axis=0) for _ in range(n)]


def _axes_of(fig):
    """Works for 2-D and 3-D: an animated 2-D figure has exactly one axes
    and it has no `zaxis` (measured)."""
    threed = [a for a in fig.axes if hasattr(a, 'zaxis')]
    return threed[0] if threed else fig.axes[0]


def _drive(ani, n):
    for f in range(n):
        ani._func(f, *ani._args)


# --- basics -----------------------------------------------------------------

def test_on_frame_is_called_once_per_frame():
    seen = []
    fig, ani = hyp.plot(_datasets(), '-', animate=True, duration=2,
                        frame_rate=4, on_frame=seen.append, show=False)
    _drive(ani, 8)
    assert len(seen) == 8
    assert all(isinstance(ctx, FrameContext) for ctx in seen)


def test_frame_context_reports_frame_and_total():
    seen = []
    fig, ani = hyp.plot(_datasets(), '-', animate=True, duration=2,
                        frame_rate=4, on_frame=seen.append, show=False)
    _drive(ani, 8)
    assert [ctx.frame for ctx in seen] == list(range(8))
    assert {ctx.n_frames for ctx in seen} == {8}


def test_frame_context_exposes_figure_axes_artists_and_datasets():
    seen = []
    fig, ani = hyp.plot(_datasets(), '-', animate=True, duration=1,
                        frame_rate=2, on_frame=seen.append, show=False)
    _drive(ani, 2)
    ctx = seen[-1]
    assert ctx.figure is fig
    assert ctx.axes is _axes_of(fig)
    assert len(ctx.artists) >= 3
    assert len(ctx.datasets) == 3


def test_parallel_mode_reports_no_serial_position():
    seen = []
    fig, ani = hyp.plot(_datasets(), '-', animate=True, duration=1,
                        frame_rate=2, on_frame=seen.append, show=False)
    _drive(ani, 2)
    assert all(ctx.current_index is None for ctx in seen)
    assert all(ctx.revealed_counts is None for ctx in seen)
    assert all(ctx.order == 'parallel' for ctx in seen)


# --- the serial schedule ----------------------------------------------------

def test_serial_schedule_is_exposed_so_callers_need_not_re_derive_it():
    seen = []
    fig, ani = hyp.plot(_datasets(), '-', animate=True, order='serial',
                        duration=4, frame_rate=4, on_frame=seen.append,
                        show=False)
    _drive(ani, 16)
    indices = [ctx.current_index for ctx in seen]
    assert indices[0] == 0, 'frame 0 must report the FIRST dataset'
    assert max(indices) == 2
    assert indices == sorted(indices), 'serial reveal advances monotonically'
    for ctx in seen:
        assert ctx.order == 'serial'
        assert 0.0 <= ctx.current_fraction <= 1.0
        assert len(ctx.revealed_counts) == 3
        assert sum(ctx.revealed_counts) <= sum(d.shape[0] for d in ctx.datasets)


def test_revealed_counts_match_the_drawn_artists_with_unequal_lengths():
    """Exercises the UNEQUAL-length branch of the reveal split.

    A LINE format pre-interpolates every animated dataset onto the frame grid
    (measured: input [17, 23, 11] -> [13, 13, 13]), so only a MARKER format
    keeps them unequal. Asserted against the artists themselves, not against
    a second copy of the formula.

    `revealed_counts` is a TUPLE (FrameContext.__post_init__ canonicalizes
    it), so `drawn` is compared as a tuple -- `(17, 4, 0) == [17, 4, 0]` is
    False and this assertion would fail for the wrong reason otherwise.
    """
    seen = []
    ds = [np.random.default_rng(s).normal(size=(n, 4)).cumsum(axis=0)
          for s, n in enumerate((17, 23, 11))]
    fig, ani = hyp.plot(ds, '.', animate=True, order='serial', duration=13,
                        frame_rate=1, on_frame=seen.append, show=False)
    _drive(ani, 13)
    ax = _axes_of(fig)
    assert [d.shape[0] for d in seen[-1].datasets] == [17, 23, 11]
    for ctx in seen:
        ani._func(ctx.frame, *ani._args)
        drawn = tuple(len(ln.get_data_3d()[0]) for ln in ax.lines[:3])
        assert ctx.revealed_counts == drawn


def test_serial_current_fraction_completes_each_dataset_before_the_next():
    seen = []
    ds = [np.random.default_rng(s).normal(size=(n, 4)).cumsum(axis=0)
          for s, n in enumerate((17, 23, 11))]
    fig, ani = hyp.plot(ds, '.', animate=True, order='serial', duration=13,
                        frame_rate=1, on_frame=seen.append, show=False)
    _drive(ani, 13)
    by_index = {}
    for ctx in seen:
        by_index.setdefault(ctx.current_index, []).append(ctx.current_fraction)
    for idx in (0, 1):
        assert max(by_index[idx]) == pytest.approx(1.0)


# --- morph segments ---------------------------------------------------------

def test_morph_reports_segment_index_and_kind():
    """C8: holds and transitions BOTH sweep current_fraction 0->1, so the
    kind must be an explicit field, derived from morph.frame_to_segment."""
    from hypertools.plot.morph import segment_frame_counts, frame_to_segment
    rng = np.random.default_rng(0)
    clouds = [rng.normal(size=(120, 3)) + off for off in (0.0, 4.0, 8.0)]
    seen = []
    fig, ani = hyp.plot(clouds, '.', animate='morph', morph_samples=120,
                        duration=6, frame_rate=4, on_frame=seen.append,
                        show=False)
    counts = segment_frame_counts(3, 24)
    assert counts == [5, 5, 5, 5, 4]
    _drive(ani, sum(counts))
    assert len(seen) == 24
    for ctx in seen:
        seg, _step, _n = frame_to_segment(counts, ctx.frame)
        assert ctx.segment_index == seg
        assert ctx.segment_kind == ('hold' if seg % 2 == 0 else 'transition')
        assert ctx.current_index == seg // 2


def test_morph_holds_and_transitions_are_not_separable_by_fraction_alone():
    """Documents WHY segment_kind exists: both kinds span the same range."""
    rng = np.random.default_rng(0)
    clouds = [rng.normal(size=(120, 3)) + off for off in (0.0, 4.0, 8.0)]
    seen = []
    fig, ani = hyp.plot(clouds, '.', animate='morph', morph_samples=120,
                        duration=6, frame_rate=4, on_frame=seen.append,
                        show=False)
    _drive(ani, 24)
    holds = {round(c.current_fraction, 3) for c in seen
             if c.segment_kind == 'hold'}
    moves = {round(c.current_fraction, 3) for c in seen
             if c.segment_kind == 'transition'}
    assert holds & moves, 'fractions overlap, so they cannot discriminate'


# --- 2-D --------------------------------------------------------------------

def test_hook_fires_for_2d_animations():
    """Every v1 helper did `[a for a in fig.axes if hasattr(a, 'zaxis')][0]`,
    which raises IndexError on a 2-D figure (measured: zaxis? [False])."""
    seen = []
    fig, ani = hyp.plot(_datasets(), '-', ndims=2, animate=True,
                        order='serial', duration=2, frame_rate=4,
                        on_frame=seen.append, show=False)
    _drive(ani, 8)
    assert len(seen) == 8
    assert not hasattr(seen[0].axes, 'zaxis')
    assert seen[-1].revealed_counts is not None


# --- hue overlays (review T7) -----------------------------------------------

def test_hook_sees_post_multicolor_artists():
    """_apply_multicolor_animation WRAPS line_ani._func (plot.py:5289), so
    the hook must be installed OUTSIDE that wrapper or it observes empty
    collections."""
    ds = _datasets()
    hue = np.linspace(0.0, 1.0, sum(d.shape[0] for d in ds))
    seen = []
    fig, ani = hyp.plot(ds, '-', hue=hue, animate=True, duration=2,
                        frame_rate=4, on_frame=seen.append, show=False)
    _drive(ani, 8)
    assert len(seen) == 8
    ax = _axes_of(fig)
    overlay = [c for c in ax.collections if c.get_label() == '_nolegend_']
    assert overlay and any(len(c._segments3d) for c in overlay)


# --- registry identity (review C7) ------------------------------------------

def test_hook_can_be_attached_after_construction():
    """The defect v1 could not have caught: a fresh list in __new__ is
    invisible to the closure created inside _draw."""
    seen = []
    result = hyp.plot(_datasets(), '-', animate=True, duration=1,
                      frame_rate=2, show=False)
    result.on_frame(seen.append)
    _drive(result[1], 2)
    assert len(seen) == 2


def test_on_frame_returns_self_for_chaining():
    a, b = [], []
    result = hyp.plot(_datasets(), '-', animate=True, duration=1,
                      frame_rate=2, show=False)
    assert result.on_frame(a.append).on_frame(b.append) is result
    _drive(result[1], 2)
    assert len(a) == len(b) == 2


def test_constructor_and_post_construction_callbacks_both_fire():
    first, second = [], []
    result = hyp.plot(_datasets(), '-', animate=True, duration=1,
                      frame_rate=2, on_frame=first.append, show=False)
    result.on_frame(second.append)
    _drive(result[1], 2)
    assert len(first) == len(second) == 2


# --- errors and limits ------------------------------------------------------

def test_hook_exception_is_not_swallowed():
    def boom(ctx):
        raise RuntimeError('hook failed')

    fig, ani = hyp.plot(_datasets(), '-', animate=True, duration=1,
                        frame_rate=2, on_frame=boom, show=False)
    with pytest.raises(RuntimeError, match='hook failed'):
        ani._func(0, *ani._args)


def test_on_frame_rejects_non_callable():
    with pytest.raises(TypeError, match='on_frame must be callable'):
        hyp.plot(_datasets(), '-', animate=True, duration=1, frame_rate=2,
                 on_frame='not callable', show=False)


def test_on_frame_without_animation_raises():
    with pytest.raises(ValueError, match='on_frame requires an animated plot'):
        hyp.plot(_datasets(), '-', on_frame=lambda ctx: None, show=False)


# --- backend parity ---------------------------------------------------------

def test_on_frame_fires_once_per_frame_on_plotly():
    """plotly DOES have a Python per-frame loop -- at BUILD time, inside
    `_add_animation` (plotly_backend.py:2517; frames appended at :2729 spin,
    :2819 morph, :2865 serial, :2975 parallel). No driving is needed: the
    callbacks have all fired by the time plot() returns.
    """
    pytest.importorskip('plotly')
    seen = []
    hyp.set_interactive_backend('plotly')
    try:
        hyp.plot(_datasets(), '-', animate=True, duration=2, frame_rate=4,
                 on_frame=seen.append, show=False)
    finally:
        hyp.set_interactive_backend('matplotlib')
    assert [ctx.frame for ctx in seen] == list(range(8)), (
        'exactly once per frame index, in order, at build time')
    assert all(isinstance(ctx, FrameContext) for ctx in seen)


@pytest.mark.parametrize('style,order', [
    (True, 'parallel'),
    (True, 'serial'),
    ('spin', 'parallel'),
    ('morph', 'serial'),
])
def test_on_frame_context_metadata_parity_across_backends(style, order):
    """THE parity guarantee: same `on_frame`, same per-frame CONTEXT METADATA.

    Deliberately NOT "output parity". Parity holds over the backend-
    INDEPENDENT fields only. `figure`/`axes`/`artists` are backend-native by
    design (matplotlib artists vs. plotly traces), so a callback that mutates
    them is not source-compatible across backends and rendered output is not
    claimed to match. Those fields are excluded here and documented as such
    -- see the next test, and the per-backend retention pair in Step 1b.
    """
    pytest.importorskip('plotly')

    def _portable(ctx):
        return (ctx.frame, ctx.n_frames, ctx.style, ctx.order,
                ctx.current_index,
                None if ctx.current_fraction is None
                else round(ctx.current_fraction, 9),
                ctx.revealed_counts, ctx.segment_index, ctx.segment_kind,
                [d.shape for d in ctx.datasets])

    kwargs = dict(animate=style, order=order, duration=2, frame_rate=4,
                  show=False)
    if style == 'morph':
        kwargs['morph_samples'] = 50

    mpl_seen = []
    fig, ani = hyp.plot(_datasets(), '.', on_frame=mpl_seen.append, **kwargs)
    _drive(ani, mpl_seen[0].n_frames if mpl_seen else 8)

    ply_seen = []
    hyp.set_interactive_backend('plotly')
    try:
        hyp.plot(_datasets(), '.', on_frame=ply_seen.append, **kwargs)
    finally:
        hyp.set_interactive_backend('matplotlib')

    # matplotlib may repeat a frame index across a loop/save; plotly may not.
    # Compare the per-index CONTENT, which is what the contract guarantees.
    mpl_by_index = {ctx.frame: _portable(ctx) for ctx in mpl_seen}
    ply_by_index = {ctx.frame: _portable(ctx) for ctx in ply_seen}
    assert sorted(ply_by_index) == sorted(mpl_by_index)
    assert ply_by_index == mpl_by_index


def test_plotly_frame_context_carries_backend_native_objects():
    """Documented, not faked: on plotly `figure` is the go.Figure and
    `artists` are that frame's traces. A caller that touches these is
    writing backend-specific code and the docstring says so."""
    pytest.importorskip('plotly')
    import plotly.graph_objects as go

    seen = []
    hyp.set_interactive_backend('plotly')
    try:
        hyp.plot(_datasets(), '-', animate=True, duration=1, frame_rate=2,
                 on_frame=seen.append, show=False)
    finally:
        hyp.set_interactive_backend('matplotlib')
    ctx = seen[-1]
    assert isinstance(ctx.figure, go.Figure)
    assert len(ctx.artists) >= 3
    assert all(hasattr(a, 'x') for a in ctx.artists), 'traces, not artists'


# --- mutation retention: the per-backend guarantee (Step 1b) ----------------
# Cross-backend OUTPUT parity is deliberately NOT asserted anywhere: artists
# and traces are backend-native, so a mutating callback is not source-
# compatible across backends. What each backend owes the caller is that a
# mutation it was handed is RETAINED in the frame that backend renders.

def test_matplotlib_callback_mutation_is_retained_in_the_rendered_frame():
    """The hook exists to mutate. Setting a title from the callback must
    survive into the frame matplotlib actually renders -- and, because a
    frame index may be re-delivered, re-running the same index must land on
    the same title rather than compounding."""
    captured = {}

    def retitle(ctx):
        ctx.axes.set_title(f'f{ctx.frame}')
        captured['ax'] = ctx.axes

    fig, ani = hyp.plot(_datasets(), '-', animate=True, duration=1,
                        frame_rate=4, on_frame=retitle, show=False)
    _drive(ani, 3)
    assert captured['ax'].get_title() == 'f2', (
        'the mutation made during the last driven frame is still on the axes')

    # idempotence: re-delivering an earlier index reproduces that index's
    # state exactly, which is what makes matplotlib's repeat harmless.
    # (`_func` here is the TEST HARNESS standing in for matplotlib's own
    # renderer, exactly as `_drive` does -- it is not the user-facing reach
    # into private internals this plan removes.)
    ani._func(1, *ani._args)
    assert captured['ax'].get_title() == 'f1'


def test_plotly_callback_mutation_is_retained_in_the_stored_frame():
    """Same guarantee on plotly, and it pins the DISPATCH ORDER: Step 6a puts
    the hook immediately BEFORE `frames.append(go.Frame(**frame_kwargs))`, so
    a trace the callback mutates is captured by the stored frame. Dispatching
    after the append would silently drop every mutation and this test is what
    catches that."""
    pytest.importorskip('plotly')

    def rename(ctx):
        ctx.artists[0].name = f'frame-{ctx.frame}'

    hyp.set_interactive_backend('plotly')
    try:
        fig = hyp.plot(_datasets(), '-', animate=True, duration=1,
                       frame_rate=4, on_frame=rename, show=False)
    finally:
        hyp.set_interactive_backend('matplotlib')
    assert fig.frames[2].data[0].name == 'frame-2'
    assert fig.frames[0].data[0].name == 'frame-0'


def test_return_model_bundle_hands_back_a_raw_funcanimation():
    """Documented limitation: the bundle never constructs a HyperAnimation
    (plot.py:4584-4586, :4612-4614), so .on_frame() is not available there --
    but on_frame= passed to plot() still fires."""
    seen = []
    bundle = hyp.plot(_datasets(), '-', animate=True, duration=1,
                      frame_rate=2, on_frame=seen.append,
                      return_model=True, show=False)
    ani = bundle['animation']
    with pytest.raises(AttributeError, match='on_frame'):
        ani.on_frame(seen.append)
    _drive(ani, 2)
    assert len(seen) == 2
```

- [ ] **Step 2: Run the test and confirm it fails**

Run: `.venv/bin/python -m pytest tests/plot/test_on_frame_hook.py -v`
Expected: collection FAILS with `ModuleNotFoundError: No module named 'hypertools.plot.animation_context'`. **26 tests (22 plain defs + 4 parametrized cases), 0 collected.**

- [ ] **Step 3: Create the context object and the shared registry**

```python
# hypertools/plot/animation_context.py
#!/usr/bin/env python
"""The per-frame context handed to a `hyp.plot(..., on_frame=...)` callback,
and the single shared registry those callbacks live in.

Before this existed, callers reached into matplotlib's private
`FuncAnimation._func`/`._args` to run code per frame, and re-derived
hypertools' own serial-reveal schedule by hand (4 of the 5 animated gallery
examples did exactly that). `FrameContext` publishes the state those callers
were reconstructing.

`FrameHooks` exists because of an ordering problem: the per-frame updater
closure is created inside `matplotlib_backend._draw`, long before `plot()`
wraps the result in a `HyperAnimation`. A callback list created in
`HyperAnimation.__new__` would therefore be a fresh, unreferenced object that
the closure never sees. `plot()` creates ONE `FrameHooks`, threads it into
`_draw`, and `HyperAnimation` ADOPTS it -- so `anim.on_frame(cb)` after
construction reaches the same list the dispatcher reads.
"""

from dataclasses import dataclass
from typing import Any, Optional, Tuple


@dataclass(frozen=True)
class FrameContext:
    """State of one drawn animation frame.

    Backend note
    ------------
    The same `on_frame` runs on both backends and sees the same values in
    every field below EXCEPT `figure`, `axes` and `artists`, which are
    backend-native: matplotlib `Figure`/`Axes`/artists, or the `go.Figure`
    and that frame's traces. The two backends also call back on different
    SCHEDULES -- matplotlib at render time (so a frame index may recur across
    a loop or a save), plotly exactly once per index at build time -- so
    callbacks must be deterministic and idempotent for a given frame
    context. They must not depend on call count, call order, wall-clock
    time, or accumulated external state.

    Mutating artists is expected and supported -- that is what the hook is
    for. What is unsupported is ACCUMULATION: ``label.set_text(title(ctx))``
    is fine, ``count += 1`` or ``alpha *= 0.9`` is not, because matplotlib
    may deliver the same frame index more than once. If you need a running
    quantity, precompute it once and index it by ``ctx.frame``.

    Attributes
    ----------
    frame : int
        Index of the frame just drawn, counting from 0.
    n_frames : int
        Total frames in the animation. For most styles this is
        ``round(duration * frame_rate)``; for ``animate='morph'`` it is
        ``sum(segment_frame_counts(...))``, which may differ by rounding.
    figure : matplotlib.figure.Figure or plotly.graph_objects.Figure
        The figure being animated. BACKEND-NATIVE.
    axes : matplotlib.axes.Axes or None
        The axes the trajectories are drawn on -- an ``Axes3D`` for 3-D
        plots, a plain ``Axes`` for 2-D ones (which have no ``zaxis``).
        BACKEND-NATIVE: ``None`` on plotly, which has no axes object (its
        equivalent state lives on the figure's ``layout``).
    artists : tuple
        The drawn artists, in dataset order: the head artists first, then
        any trail artists, matching the backend's own bookkeeping.
        The CONTAINER is always a tuple -- see "Container types" below --
        while the artists inside it are the backend's own live objects and
        are meant to be mutated.
        BACKEND-NATIVE: on plotly these are that frame's ``go.Scatter``/
        ``go.Scatter3d`` traces, in the same order.

        ARTIST LIFETIME -- read this before writing a callback. Whether
        ``artists`` holds fresh objects per frame or the same objects
        re-delivered depends on the backend and style:

        ==================================  ===========================
        backend / style                     lifetime
        ==================================  ===========================
        matplotlib, ALL styles              shared live artists,
                                            mutated in place each render
        plotly spin (no surfaces)           shared figure traces
        plotly spin (surfaced)              shared traces, then that
                                            frame's Mesh3d updates
        plotly parallel/serial/window/      per-frame trace payloads
        morph
        ==================================  ===========================

        Matplotlib never hands you a fresh artist: ``FuncAnimation``'s
        updater mutates the same ``Line2D``/collection objects every
        frame, so ``ctx.artists[0]`` on frame 1 and on frame 2 are the
        SAME object in different states.

        THE PORTABLE RULE, on both backends: ASSIGN the complete desired
        value on EVERY invocation, including the default. What breaks is
        not a per-frame DECISION, it is a per-frame ASSIGNMENT -- writing
        the attribute on some frames and leaving it untouched on others.
        The rule is the same on both backends but the reason is NOT, and
        the failure modes are opposite:

        * Where artists are SHARED (matplotlib all styles, plotly spin),
          anything you set persists until something overwrites it, so
          ``if ctx.frame == 0: artist.set_color('red')`` colours the
          ENTIRE animation rather than frame 0.
        * Where they are PER-FRAME (plotly parallel/serial/window/morph),
          the same callback colours ONLY frame 0 -- each frame carries an
          independent trace payload that the callback mutates before it
          is stored. Measured 2026-07-30: ``fig.frames[0].data[0] is not
          fig.frames[1].data[0]`` for every one of those four styles.

        So a skipped assignment does not merely misbehave -- it misbehaves
        DIFFERENTLY per backend. Highlighting exactly one frame is
        perfectly legitimate; just put the condition in the VALUE, not
        around the call::

            # correct everywhere -- assigns on every frame
            artist.set_color('red' if ctx.frame == target else DEFAULT)

            # correct everywhere -- also assigns on every frame
            artist.set_color(COLOURS[ctx.frame])

            # BROKEN -- assigns on one frame, leaves the rest to chance
            if ctx.frame == target:
                artist.set_color('red')

        Note that "a mutation is retained in the rendered frame" does NOT
        mean artists are isolated per frame. It means the backend renders
        what you set; where artists are shared, it renders it for every
        subsequent frame too.
    datasets : tuple of numpy.ndarray
        The arrays the animation actually DRAWS FROM, in dataset order --
        not the raw input. For a line format `plot()` pre-interpolates every
        animated dataset onto the frame grid, so these may be denser or
        coarser than what you passed in; `revealed_counts[i]` indexes into
        ``datasets[i]``.
    style : bool or str
        The resolved backend animate mode (``True``/``'serial'``/``'spin'``/
        ``'window'``/``'morph'``) -- i.e. after ``order=`` has been folded in.
    order : {'parallel', 'serial'}
        The resolved ordering.
    current_index : int or None
        For serial-style animations, the index of the dataset currently
        being revealed. For ``animate='morph'`` it is the dataset the
        current segment belongs to (``segment_index // 2``: the shape being
        held, or the SOURCE of the transition). ``None`` for parallel
        animations, where every dataset advances together.
    current_fraction : float or None
        Progress through the current dataset (serial) or the current
        SEGMENT (morph), in [0, 1]. ``None`` when `current_index` is
        ``None``. **This does not distinguish a morph hold from a morph
        transition** -- both sweep 0 -> 1 over their own segment. Use
        `segment_kind` for that.
    revealed_counts : tuple of int or None
        Number of rows of each dataset currently drawn. ``None`` for
        parallel and morph animations -- ``None`` is preserved as
        ``None``, never normalized to an empty tuple.
    segment_index : int or None
        For ``animate='morph'``, the index into the hold/morph schedule
        (``hypertools.plot.morph.frame_to_segment``). ``None`` otherwise.
    segment_kind : {'hold', 'transition'} or None
        ``'hold'`` for even `segment_index` (a fully-formed cloud is being
        held) and ``'transition'`` for odd (one cloud is easing into the
        next) -- the parity rule `morph.morph_positions` implements.
        ``None`` for non-morph animations.

    Container types
    ---------------
    ``artists``, ``datasets`` and ``revealed_counts`` are always TUPLES
    (``revealed_counts`` is ``None`` or a tuple). This is a public
    guarantee, not an accident of whichever branch built the frame.

    Eleven separate call sites record frame state -- seven matplotlib
    updaters and four plotly frame-build branches -- and each has a
    different sequence in hand: ``list(lines) + [...]``, a list
    comprehension, ``tuple(fig.data[i] for i in trace_indices)``. Left
    alone, ``type(ctx.artists)`` would vary by backend and style, which is
    not something a public field may do.

    All eleven funnel through the SINGLE construction site --
    `FrameHooks.dispatch`'s ``FrameContext(figure=..., axes=...,
    **self.state)`` -- so normalizing in `__post_init__` covers every one
    of them, and covers any branch added later without that branch having
    to know. This is why the coercion lives here and not at the recorders.

    Tuples rather than lists because the dataclass is ``frozen=True``: a
    list would make that promise half-true, letting a caller
    ``ctx.artists.append(...)`` or ``ctx.revealed_counts.sort()`` and
    corrupt the context. The CONTAINED artists stay mutable on purpose --
    mutating them is what the hook is for. What is fixed is MEMBERSHIP.
    """

    frame: int
    n_frames: int
    figure: Any
    axes: Any
    artists: Tuple[Any, ...] = ()
    datasets: Tuple[Any, ...] = ()
    style: Any = None
    order: str = 'parallel'
    current_index: Optional[int] = None
    current_fraction: Optional[float] = None
    revealed_counts: Optional[Tuple[int, ...]] = None
    segment_index: Optional[int] = None
    segment_kind: Optional[str] = None

    def __post_init__(self):
        """Canonicalize the container types. `object.__setattr__` is the
        documented way to assign inside a frozen dataclass."""
        object.__setattr__(self, 'artists', tuple(self.artists))
        object.__setattr__(self, 'datasets', tuple(self.datasets))
        if self.revealed_counts is not None:
            object.__setattr__(self, 'revealed_counts',
                               tuple(self.revealed_counts))


class FrameHooks:
    """The ONE mutable callback registry for an animated plot.

    Created in `plot()` (so it exists before the backend builds its updater
    closures), threaded into `matplotlib_backend._draw`, and adopted -- never
    re-created -- by `HyperAnimation.__new__`.

    Backend updaters call `record(...)` with whatever they know about the
    frame; they never invoke callbacks. On matplotlib, `plot()` installs
    `dispatch` as the OUTERMOST wrapper of `line_ani._func`, after any other
    wrapping (notably `_apply_multicolor_animation`'s, plot.py:5289), so
    callbacks always see final artists. On plotly, `_add_animation` records
    and dispatches inside its own frame-building loop (one call per
    `frames.append(go.Frame(...))` site) -- the same registry, the same
    `FrameContext` fields, a different schedule; see `FrameContext`'s
    backend note.
    """

    __slots__ = ('callbacks', 'state')

    def __init__(self, callbacks=None):
        self.callbacks = list(callbacks or [])
        self.state = {}

    def add(self, callback):
        if not callable(callback):
            raise TypeError(
                f"on_frame must be callable; got {type(callback).__name__}.")
        self.callbacks.append(callback)
        return self

    def record(self, **state):
        """Store this frame's state. Cheap and unconditional: a no-callback
        animation pays one dict assignment per frame."""
        self.state = state

    def dispatch(self, figure, axes):
        """Build a FrameContext from the recorded state and run every
        callback. Exceptions propagate -- a broken hook must be visible, not
        swallowed into a silently-wrong animation."""
        if not self.callbacks or not self.state:
            return
        ctx = FrameContext(figure=figure, axes=axes, **self.state)
        for callback in self.callbacks:
            callback(ctx)
```

- [ ] **Step 4: Publish the reveal schedule from ONE implementation**

Add to `hypertools/plot/matplotlib_backend.py` at module level (above `_draw`, `:443`):

```python
def serial_reveal_counts(lengths, num, total_frames):
    """Rows revealed per dataset at frame `num` of a serial animation.

    THE reveal schedule. `update_lines_serial` (3-D), `update_lines_serial_2d`
    and `FrameContext.revealed_counts` all read it, and
    `plot._apply_multicolor_animation` recovers its hue window from it, so the
    formula exists once. Equivalent to the historical inline code
    (`revealed = total_points * num / max(1, total_frames - 1)`; per dataset
    `shown = int(np.clip(revealed - start, 0, n_pts))`).
    """
    total_points = sum(lengths)
    revealed = total_points * num / max(1, total_frames - 1)
    counts, remaining = [], revealed
    for length in lengths:
        counts.append(int(max(0, min(length, remaining))))
        remaining -= length
    return counts


def serial_current_index(counts, lengths):
    """``(index, fraction)`` of the dataset mid-reveal at these counts."""
    done = -1
    for i, (shown, length) in enumerate(zip(counts, lengths)):
        if 0 < shown < length:
            return i, (shown - 1) / max(1, length - 1)
        if shown >= length:
            done = i
    if done < 0:
        return 0, 0.0
    return done, 1.0
```

Rewrite `update_lines_serial` (`matplotlib_backend.py:1316-1326`) and `update_lines_serial_2d` (`:2062-2071`) to call `serial_reveal_counts` instead of recomputing `revealed`/`shown` inline, keeping `revealed` itself for `_sync_anim_labels` (`:1395`, `:2104`). Replace `plot.py:5265-5269`'s third copy with:

```python
                from .matplotlib_backend import serial_reveal_counts
                _lengths = [_points(j).shape[0] for j in range(n)]
                shown = serial_reveal_counts(_lengths, num,
                                             int(total_frames))[i]
                end = shown
                start = max(0, shown - head_len)
```

- [ ] **Step 5: Record frame state in all seven updaters**

Add `frame_hooks=None` to `_draw`'s signature (`matplotlib_backend.py:443-482`) and, at the end of each updater (before its `return`), record what it knows. For `update_lines_serial`:

```python
        if frame_hooks is not None:
            _counts = serial_reveal_counts(lengths, num, total_frames)
            _idx, _frac = serial_current_index(_counts, lengths)
            frame_hooks.record(
                frame=int(num), n_frames=int(total_frames),
                artists=list(lines) + [t for t in trail_lines if t is not None],
                datasets=list(data_lines), style='serial', order='serial',
                current_index=_idx, current_fraction=_frac,
                revealed_counts=_counts)
```

For `update_lines_parallel` / `update_lines_spin` / their 2-D twins, record with `order='parallel'` and `current_index=current_fraction=revealed_counts=None`. For `update_morph` / `update_morph_2d`, record from the schedule already computed at `matplotlib_backend.py:1413-1414`:

```python
        if frame_hooks is not None:
            frame_hooks.record(
                frame=int(num),
                n_frames=int(sum(morph_state["frame_counts"])),
                artists=[morph_state["artist"]],
                datasets=list(morph_state["sampled"]),
                style='morph', order='serial',
                current_index=seg_idx // 2,
                current_fraction=step / max(1, n_steps - 1),
                revealed_counts=None,
                segment_index=seg_idx,
                segment_kind='hold' if seg_idx % 2 == 0 else 'transition')
```

- [ ] **Step 6: Wire `on_frame` through `plot()` and `HyperAnimation`**

Add `on_frame=None` to `plot()`'s signature. Validate fail-fast at `plot.py:2231`, beside Task 1's and Task 5's calls:

```python
    if on_frame is not None:
        if not callable(on_frame):
            raise TypeError(
                f"on_frame must be callable; got {type(on_frame).__name__}.")
        if not animate:
            raise ValueError(
                "on_frame requires an animated plot; pass animate=True "
                "(or 'spin'/'serial'/'window'/'morph').")
```

There is **no backend check here.** v2 raised `NotImplementedError` for plotly at this point; that is deleted, not relocated (see the *Revision note (v3)*). `resolve_backend` is not consulted by this validation at all.

Create the registry once, before dispatch, and pass it into `_draw`:

```python
    _frame_hooks = FrameHooks([on_frame] if on_frame is not None else [])
```
```python
                frame_hooks=_frame_hooks,
```

Install the dispatcher as the **outermost** wrapper, immediately after the `_apply_multicolor_animation` / `_apply_multicolor_lines` block that ends around `plot.py:4399`:

```python
            # the hook dispatcher goes on LAST, so callbacks observe the
            # final artists -- _apply_multicolor_animation wraps
            # line_ani._func itself (plot.py:5289) and would otherwise run
            # after them, handing hooks pre-multicolor collections.
            if line_ani is not None:
                _orig_frame_func = line_ani._func

                def _hyp_frame_with_hooks(num, *fargs,
                                          _orig=_orig_frame_func):
                    result = _orig(num, *fargs)
                    _frame_hooks.dispatch(fig, ax)
                    return result

                line_ani._func = _hyp_frame_with_hooks
```

Finally, have `HyperAnimation` **adopt** the registry. In `hypertools/plot/hyper_animation.py`:

```python
    def __new__(cls, figure, animation, frame_hooks=None):
        self = super().__new__(cls, (figure, animation))
        # ADOPT the registry plot() already threaded into the backend -- do
        # NOT create one here. The per-frame updater closure was built inside
        # `_draw` long before this wrapper existed, so a list created here
        # would be a fresh, unreferenced object and `on_frame()` could never
        # fire.
        self._frame_hooks = frame_hooks
        return self

    def on_frame(self, callback):
        """Register `callback` to run after every drawn frame.

        The callback receives a
        :class:`~hypertools.plot.animation_context.FrameContext`. Returns
        `self`, so calls chain. Exceptions from a callback propagate.

        Not available on the ``return_model=True`` bundle, which hands back
        the raw ``FuncAnimation``; pass ``on_frame=`` to ``plot()`` instead
        on that path.
        """
        if self._frame_hooks is None:
            raise RuntimeError(
                "this HyperAnimation carries no frame-hook registry (it was "
                "constructed directly rather than by hyp.plot); pass "
                "on_frame= to hyp.plot instead.")
        self._frame_hooks.add(callback)
        return self
```

and update the construction site at `plot.py:4612-4614`:

```python
        return HyperAnimation(fig, line_ani, frame_hooks=_frame_hooks)
```

- [ ] **Step 6a: Dispatch the same registry from plotly's build loop**

This step — **not Task 4** — owns the plotly side, so the dispatch block exists once. Thread `frame_hooks=_frame_hooks` into `plotly_draw` (`plot.py:4206-4246`) and on into `_add_animation` (`plotly_backend.py:2517`), exactly as Task 8 threads `segment_titles` through the same two signatures. Then, in `_add_animation`, add the same block immediately **before** each of the four `frames.append(go.Frame(**frame_kwargs))` calls — `:2729` (spin), `:2819` (morph), `:2865` (serial, as rewritten by Task 4), `:2975` (the `else:` parallel/window branch at `:2866`):

```python
            if frame_hooks is not None:
                frame_hooks.record(**_frame_state)   # per-branch, see below
                frame_hooks.dispatch(fig, None)      # plotly has no Axes
```

`_frame_state` is built per branch from quantities each branch already has in hand, and must name the **same fields with the same values** as the matplotlib updater for the same style (Step 5) — that identity is what `test_on_frame_context_metadata_parity_across_backends` asserts:

| branch | `style` / `order` | serial position | segment fields |
|-|-|-|-|
| spin (`:2729`) | `'spin'` / `'parallel'` | all `None` | `None` |
| morph (`:2819`) | `'morph'` / `'serial'` | `current_index=seg_idx // 2`, `current_fraction=step / max(1, n_steps - 1)` | `segment_index=seg_idx`, `segment_kind='hold' if seg_idx % 2 == 0 else 'transition'` |
| serial (`:2865`) | `'serial'` / `'serial'` | `revealed_counts=_shown`, then `serial_current_index(_shown, lengths)` — the **same** helper, imported from `matplotlib_backend` (Step 4), which Task 8 Step 4 also imports here | `None` |
| parallel/window (`:2975`) | the resolved `animate` / `'parallel'` | all `None` | `None` |

`datasets` is `data`; `n_frames` is `n_frames`. `figure` is the `go.Figure` and `axes` is `None`, per `FrameContext`'s backend note.

**`artists` — and the spin branch is NOT `frame_traces`.** Three of the four branches build a per-frame trace list named `frame_traces`; **spin does not build one at all**, so `artists=frame_traces` would raise `NameError` for `animate='spin'`. Verified against source 2026-07-30: the spin loop's payload is

```python
frame_kwargs = dict(
    name=str(k),
    layout=dict(scene_camera=dict(eye=_camera_eye(elev, angle, r=_anim_zoom_r(zoom)))))
```

with **no `data` key at all** (`plotly_backend.py:2695-2699`) — the branch's own comment says *"the FULL dataset is static in 'spin' mode (only the camera rotates)"*. A `data` key appears **only** when surfaces are enabled, and then it is `surf_data`, the re-shaded `Mesh3d` updates addressed by `surface_trace_indices` (`plotly_backend.py:2711-2735`).

Substituting `[]` is **not** an acceptable fallback: `FrameContext.artists` is documented as the drawn artists, and an empty tuple would say a spin frame draws nothing, which is false.

**The contract, stated explicitly per branch:**

| branch | `artists` | shared or per-frame? |
|-|-|-|
| morph, serial, parallel/window | `frame_traces` — that frame's traces, head then trail, in the order the loop already builds them | **per-frame**: mutating one affects only that frame |
| **spin, no surfaces** | the figure's **static data traces**, `tuple(fig.data[i] for i in trace_indices)` — the traces the frame actually renders, which spin re-uses rather than re-sending | **shared**: every frame renders the same trace objects, so a mutation applies to the whole animation |
| **spin, surfaced** | the same static data traces **followed by** that frame's `surf_data` mesh updates | **mixed**: the leading static traces are shared; the trailing `surf_data` entries are per-frame |
| *(matplotlib, all styles — for contrast)* | the live `Line2D`/collection artists the updater mutates | **shared**: matplotlib never hands out a fresh artist, so `ctx.artists[0]` is the same object every frame. Do **not** read this table as "plotly is shared, matplotlib is per-frame" |

Implement it as an explicit per-branch assignment, not a single shared expression:

```python
# spin branch, immediately before frames.append(go.Frame(**frame_kwargs)):
_frame_artists = tuple(fig.data[i] for i in trace_indices)
if surface_trace_indices:
    _frame_artists = _frame_artists + tuple(surf_data)
```

**Scope check (verified 2026-07-30, do not skip):** all three names this needs are already in scope at the spin loop — `fig` is `_add_animation`'s first parameter (`plotly_backend.py:2517`), `trace_indices` is bound at `:2602` as `list(range(data_trace_start, data_trace_start + n_data_traces))`, and `surface_trace_indices` at `:2609`. Both bindings precede the spin branch at `:2666`, so no new plumbing is required. `surf_data` is local to the surfaced sub-branch, which is why the `tuple(surf_data)` append sits inside the `if`.

**Why "shared" is the honest answer rather than a wart to hide.** Spin genuinely does not redraw the data — only the camera moves. A caller who recolours `ctx.artists[0]` on frame 3 of a spin will see that colour on every frame, and that is what the figure really does. The docstring, the guide and the CHANGELOG all say so; the retention test asserts the shared semantics rather than pretending it is per-frame.

This is also why the parity test can keep excluding `artists` (it compares only backend-independent fields) — parity across backends was never the guarantee for artists. What pins spin is the dedicated pair of tests below.

Add these to `tests/plot/test_on_frame_hook.py`, in the backend-parity section:

```python
def test_matplotlib_artists_are_shared_across_frame_deliveries():
    """Matplotlib hands out the SAME artist objects every frame -- the
    FuncAnimation updater mutates them in place. Verified against the real
    backend: line identities are unchanged across frames 0/1/2.

    This is why the contract is "set the complete state for this frame":
    a conditional mutation persists into every later frame. The plan
    previously claimed matplotlib was per-frame throughout; it is not.
    """
    seen = []
    fig, ani = hyp.plot(_datasets(), '-', animate=True, duration=1,
                        frame_rate=4, on_frame=seen.append, show=False)
    _drive(ani, 3)
    assert len(seen) == 3
    first = tuple(id(a) for a in seen[0].artists)
    assert all(tuple(id(a) for a in ctx.artists) == first for ctx in seen), (
        'matplotlib re-delivers the same artist objects, not copies')


def test_plotly_spin_artists_are_the_static_data_traces():
    """Regression: the spin branch builds no `frame_traces` (its frame payload
    is camera-layout only, plotly_backend.py:2695-2699), so a literal
    `artists=frame_traces` raises NameError there. Spin publishes the traces it
    actually renders -- the figure's static ones -- never an empty tuple."""
    pytest.importorskip('plotly')
    seen = []
    hyp.set_interactive_backend('plotly')
    try:
        fig = hyp.plot(_datasets(), '-', animate='spin', duration=1,
                       frame_rate=4, on_frame=seen.append, show=False)
    finally:
        hyp.set_interactive_backend('matplotlib')
    assert len(seen) == 4
    assert all(len(ctx.artists) > 0 for ctx in seen), 'never empty'
    assert all(hasattr(a, 'x') for a in seen[0].artists), 'traces, not artists'
    # shared, not per-frame: every frame publishes the SAME trace objects
    assert all(ctx.artists[0] is seen[0].artists[0] for ctx in seen)
    assert seen[0].artists[0] in tuple(fig.data)


def test_plotly_surface_spin_artists_include_the_per_frame_mesh_updates():
    """Surfaced spin DOES send per-frame data (`surf_data`, the re-shaded
    Mesh3d updates at plotly_backend.py:2711-2735). Those trail the static
    traces, so a caller can reach both."""
    pytest.importorskip('plotly')
    rng = np.random.default_rng(0)
    cloud = rng.normal(size=(40, 3))
    seen = []
    hyp.set_interactive_backend('plotly')
    try:
        fig = hyp.plot([cloud], animate='spin', surface=True, duration=1,
                       frame_rate=4, on_frame=seen.append, show=False)
    finally:
        hyp.set_interactive_backend('matplotlib')
    import plotly.graph_objects as go

    # NOT `hasattr(t, 'x')` -- go.Mesh3d has an .x too (verified), so that
    # predicate matches every trace and the assertion can never fail.
    # Discriminate on TYPE.
    assert any(isinstance(t, go.Mesh3d) for t in seen[0].artists), (
        'the frame\'s re-shaded mesh updates are appended')
    assert isinstance(seen[0].artists[-1], go.Mesh3d)
    # the trailing mesh entries are PER-FRAME: different objects each frame
    assert seen[0].artists[-1] is not seen[1].artists[-1]
    # ...while the LEADING entries are the shared figure traces themselves,
    # which is what makes this the documented mixed case
    assert seen[0].artists[0] is seen[1].artists[0]
    assert seen[0].artists[0] in tuple(fig.data)


def test_plotly_spin_mutation_is_retained_and_is_figure_wide():
    """Spin's documented consequence: because the traces are shared, a
    mutation is figure-wide rather than per-frame. Asserted, not hidden."""
    pytest.importorskip('plotly')

    def rename(ctx):
        if ctx.frame == 1:
            ctx.artists[0].name = 'touched-on-frame-1'

    hyp.set_interactive_backend('plotly')
    try:
        fig = hyp.plot(_datasets(), '-', animate='spin', duration=1,
                       frame_rate=4, on_frame=rename, show=False)
    finally:
        hyp.set_interactive_backend('matplotlib')
    assert fig.data[0].name == 'touched-on-frame-1', (
        'the mutation lands on the shared figure trace')


@pytest.mark.parametrize('style', [True, 'serial', 'window', 'morph'])
def test_plotly_non_spin_frames_are_isolated_per_frame(style):
    """The MIRROR IMAGE of the spin test above, and the reason the guide
    documents two opposite failure modes rather than one.

    plotly's parallel/serial/window/morph branches each build their own
    `frame_traces`, so a callback that mutates on frame 1 only affects
    frame 1 -- the exact opposite of spin and of matplotlib, where the same
    callback would affect the whole animation. Measured against the real
    backend 2026-07-30 (before this plan changes anything):
    `fig.frames[0].data[0] is not fig.frames[1].data[0]` for all four.

    A caller who writes a conditional mutation therefore gets DIFFERENT
    wrong behaviour per backend, which is why the portable contract is
    "set the complete state every frame" and not "mutations persist".
    """
    pytest.importorskip('plotly')

    def rename(ctx):
        if ctx.frame == 1:
            ctx.artists[0].name = 'touched-on-frame-1'

    kwargs = dict(morph_samples=40) if style == 'morph' else {}
    hyp.set_interactive_backend('plotly')
    try:
        fig = hyp.plot(_datasets(), '-', animate=style, duration=1,
                       frame_rate=4, on_frame=rename, show=False, **kwargs)
    finally:
        hyp.set_interactive_backend('matplotlib')

    assert fig.frames[1].data[0].name == 'touched-on-frame-1'
    assert fig.frames[0].data[0].name != 'touched-on-frame-1', (
        'frame 0 carries its own payload; the mutation must NOT leak back')
    assert fig.frames[2].data[0].name != 'touched-on-frame-1', (
        'nor forward -- this is what makes these styles per-frame')
    assert fig.frames[0].data[0] is not fig.frames[1].data[0]


@pytest.mark.parametrize('backend,style', [
    ('matplotlib', True),        # revealed_counts is None
    ('matplotlib', 'serial'),    # revealed_counts from serial_reveal_counts
    ('matplotlib', 'morph'),     # artists=[morph_state["artist"]]
    ('plotly', True),            # artists=frame_traces
    ('plotly', 'serial'),        # artists=frame_traces, revealed_counts=_shown
    ('plotly', 'spin'),          # artists=tuple(fig.data[i] ...) -- the odd one
])
def test_frame_context_containers_are_canonical_tuples(backend, style):
    """`artists`, `datasets` and `revealed_counts` are TUPLES on every
    backend and every style -- a public field may not change type
    according to which branch built it.

    This is the regression guard for `FrameContext.__post_init__`. Eleven
    call sites record frame state and each has a different sequence in
    hand; before the normalizer, matplotlib passed lists and plotly's spin
    branch passed a tuple, so `type(ctx.artists)` varied by style.
    """
    if backend == 'plotly':
        pytest.importorskip('plotly')
    seen = []
    kwargs = dict(morph_samples=40) if style == 'morph' else {}
    hyp.set_interactive_backend(backend)
    try:
        result = hyp.plot(_datasets(), '.', animate=style, duration=1,
                          frame_rate=4, on_frame=seen.append, show=False,
                          **kwargs)
    finally:
        hyp.set_interactive_backend('matplotlib')
    if backend == 'matplotlib':
        _drive(result[1], 3)

    assert seen, 'the hook must have fired'
    for ctx in seen:
        assert type(ctx.artists) is tuple, type(ctx.artists)
        assert type(ctx.datasets) is tuple, type(ctx.datasets)
        assert (ctx.revealed_counts is None
                or type(ctx.revealed_counts) is tuple), ctx.revealed_counts
        # frozen means MEMBERSHIP is fixed; the artists inside stay mutable
        with pytest.raises(AttributeError):
            ctx.artists.append(None)
```

- [ ] **Step 6b: Put `FrameContext` on the public surface — and keep `FrameHooks` off it**

**The decision:** `FrameContext` is **public** and exported as `hypertools.FrameContext`. `FrameHooks` is **internal** and is not exported, not documented, and not named in any public docstring.

The rationale is the split between the two: a user *receives* a `FrameContext` on every callback and will reasonably want to type-annotate it, `isinstance`-check it, or build one in their own tests. They never construct or touch a `FrameHooks` — it is the registry `plot()` creates and `HyperAnimation` adopts, an implementation detail of contract 3. Exporting the receipt type without the registry is the smallest surface that makes the hook usable.

Leaving it at `hypertools.plot.animation_context.FrameContext` was **not** acceptable: 1.0 deliberately curates `__all__` so that `from hypertools import *` yields exactly the documented names (`hypertools/__init__.py:43-52`), and a public API reachable only by a three-segment private-looking path contradicts that.

These four edits are **one atomic change** — do all of them in this step. `tests/test_codeorg_licensing_audit_fixes.py:294-305` compares `__all__` against a **hardcoded literal set**, so touching `__all__` without touching that literal turns the suite red:

1. In `hypertools/__init__.py`, beside the existing `HyperAnimation` import:

```python
from .plot.animation_context import FrameContext
```

2. Add `'FrameContext'` to `__all__` (`hypertools/__init__.py:46-52`), next to `'HyperAnimation'`:

```python
    'set_interactive_backend', 'HyperAnimation', 'FrameContext', 'io',
```

3. In `docs/api.rst`, add it under the existing **Plot** `autosummary` block beside `HyperAnimation` (`docs/api.rst:111-115`):

```rst
  plot
  HyperAnimation
  FrameContext
```

> **Docstring-rendering check (done for you, 2026-07-30).** This is the step that puts `FrameContext`'s docstring through `autodoc` + `numpydoc` under the CI build's `-W`, where an unrecognized section header would become a build **error**, not a warning. `FrameContext` carries two non-standard numpydoc sections — *Backend note* and *Container types* — so this was built before the plan shipped: a minimal `sphinx -b html -W -E -a` project with this repo's exact extension list (`sphinx.ext.autodoc`, `numpydoc`, `sphinx.ext.autosummary`, `numpydoc_class_members_toctree = False`) and a `FrameContext` stub carrying both headers **built clean, zero warnings**. Keep the headers as written; if you add another custom section, re-run that check rather than assuming.

4. In `tests/test_codeorg_licensing_audit_fixes.py`, add `'FrameContext'` to the `documented` literal inside `test_all_names_resolve_and_cover_documented_api` (`:295-300`), so the curated-surface assertion still describes reality:

```python
                  'Pipeline', 'set_interactive_backend', 'HyperAnimation',
                  'FrameContext',
                  'io', 'HypertoolsError', 'HypertoolsBackendError',
```

Then add this test to `tests/plot/test_on_frame_hook.py`, in the basics section:

```python
def test_frame_context_is_exported_at_top_level_but_frame_hooks_is_not():
    """`FrameContext` is public: users receive one per callback and will
    annotate and isinstance-check it. `FrameHooks` is the internal registry
    from contract 3 -- users never construct one, so it stays off the
    curated surface that `hypertools/__init__.py:43-52` maintains."""
    assert hyp.FrameContext is FrameContext
    assert 'FrameContext' in hyp.__all__
    assert not hasattr(hyp, 'FrameHooks')
    assert 'FrameHooks' not in hyp.__all__
```

- [ ] **Step 7: Run the test and confirm it passes**

Run: `.venv/bin/python -m pytest tests/plot/test_on_frame_hook.py -v`
Expected: **31 passed** (27 plain + 4 parametrized parity cases) — the 23 defs from Step 1, plus Step 6a's four artist-lifetime tests (three plotly-spin, one matplotlib shared-identity) and Step 6b's export test.

Then confirm the public-surface tests still pass, since Step 6b touched them:

Run: `.venv/bin/python -m pytest tests/test_codeorg_licensing_audit_fixes.py -v -k "star_import or documented"`
Expected: **3 passed** (`test_star_import_yields_exactly_all`, `test_star_import_does_not_leak_internal_submodules`, `test_all_names_resolve_and_cover_documented_api`).

- [ ] **Step 8: Document the hook**

Add an `on_frame` entry to `plot()`'s docstring:

```
    on_frame : callable, optional
        Called after each animation frame is drawn, with a single
        ``FrameContext`` argument exposing the frame index, the axes and
        drawn artists, the arrays being animated, and -- for serial-style
        animations -- which dataset is being revealed, how far through it,
        and the exact per-dataset reveal counts. For ``animate='morph'`` it
        also reports ``segment_index`` and ``segment_kind`` ('hold' or
        'transition'). Use this instead of reaching into matplotlib's
        private ``FuncAnimation._func``. On MATPLOTLIB, callbacks may also
        be attached afterwards via ``HyperAnimation.on_frame()``; this is
        not available on plotly, whose animated return is a plain
        ``go.Figure`` with its frames already built, so pass ``on_frame=``
        here for backend-portable code.

        Supported on BOTH backends, with the same per-frame context
        metadata but different call schedules: matplotlib calls back at
        render time, so a frame index may recur across a looping animation
        or a save; plotly calls back exactly once per frame index, while
        the frames are built. **Callbacks must be deterministic and
        idempotent for a given frame context. They must not depend on call
        count, call order, wall-clock time, or accumulated external
        state.**

        Mutating what the context hands you is the point of the hook and is
        fully supported -- the example below sets a title every frame.
        What is unsupported is accumulation (``count += 1``,
        ``alpha *= 0.9``), because a repeated frame would change the
        result. Precompute running quantities and index them by
        ``ctx.frame``.

        ``ctx.figure``, ``ctx.axes`` and ``ctx.artists`` are backend-native
        (``ctx.axes`` is ``None`` on plotly, whose ``ctx.artists`` are that
        frame's traces), so a callback that touches them is **not**
        portable across backends; every other field is identical across
        backends.

        >>> def annotate(ctx):
        ...     ctx.axes.set_title(f'frame {ctx.frame} of {ctx.n_frames}')
        >>> anim = hyp.plot(data, animate=True, on_frame=annotate)
```

Note the `return_model=True` limitation in the `return_model` entry (`plot.py:1920`). Say the same thing about determinism/idempotence and schedules in the animation guide (the prose that Task 9 Step 2 rewrites the examples against), so a reader who never opens the docstring still gets it.

- [ ] **Step 9: Run the FULL suite (central dispatch changed)**

Run: `.venv/bin/python -m pytest -q`
Expected: `2672 passed, 13 skipped`. Grep for any test asserting that `on_frame=` is unavailable on plotly — there is none in the repo today, but if one appears it is asserting v2's removed premise, not a contract.

- [ ] **Step 10: Commit**

```bash
git add hypertools/plot/animation_context.py hypertools/plot/hyper_animation.py \
        hypertools/plot/plot.py hypertools/plot/matplotlib_backend.py \
        hypertools/plot/plotly_backend.py tests/plot/test_on_frame_hook.py
git commit -m "feat(plot): public on_frame hook over one shared frame-callback registry, on both backends"
```

---

## Task 8: Per-segment titles during serial-style animations

The morph example imports the private `hypertools.plot.morph.frame_to_segment` to title each shape while it is fully formed and blank it through transitions (`examples/animate_morph_zoo.py:106-115`). With Task 7's schedule published, `title=` can accept one entry per dataset and do this natively.

**The discriminator is segment parity, not a fraction (review C8).** Computed for the exact test parameters below (3 clouds, `duration=6, frame_rate=4` ⇒ 24 frames), `segment_frame_counts(3, 24)` = `[5, 5, 5, 5, 4]`:

```
frames  0- 4  seg 0  hold        -> 'alpha'
frames  5- 9  seg 1  transition  -> ''
frames 10-14  seg 2  hold        -> 'beta'
frames 15-19  seg 3  transition  -> ''
frames 20-23  seg 4  hold        -> 'gamma'
```

v1's `ctx.current_fraction not in (0.0, 1.0)` blanks 12 of the 15 hold frames and names 4 transition frames, because holds and transitions both sweep 0→1. `FrameContext.segment_kind` (Task 7) is the contract.

**Backend parity.** matplotlib sets `ax.set_title` from an internal callback on the same registry (so there is exactly one per-frame path). plotly carries the title on each `go.Frame`'s `layout` — verified: `go.Frame(layout=dict(title=dict(text='alpha')))` round-trips, the same mechanism already used for `scene_camera`. Both are tested.

**Files:**
- Modify: `hypertools/plot/plot.py` (widen `_validate_title`; register the internal title callback), `hypertools/plot/plotly_backend.py` (per-frame `layout.title`)
- Test: `tests/plot/test_serial_titles.py` (create)

**Interfaces:**
- Consumes: `_validate_title(title, style, order, n_datasets)` (Task 1 — the signature was written for this from the start); `FrameContext.current_index`/`segment_kind` (Task 7); `order` (Task 5).
- Produces: `title=` accepts a sequence of `n_datasets` strings when the animation is serial-style.

- [ ] **Step 1: Write the failing test**

```python
# tests/plot/test_serial_titles.py
import matplotlib
matplotlib.use("Agg")

import numpy as np
import pytest

import hypertools as hyp
from hypertools.plot.morph import segment_frame_counts, frame_to_segment


def _datasets(n=3, rows=20, dims=4, seed=0):
    rng = np.random.default_rng(seed)
    return [rng.normal(size=(rows, dims)).cumsum(axis=0) for _ in range(n)]


def _clouds(n=3, pts=120, seed=0):
    rng = np.random.default_rng(seed)
    return [rng.normal(size=(pts, 3)) + off for off in np.arange(n) * 4.0]


def _titles_over(ani, fig, n):
    threed = [a for a in fig.axes if hasattr(a, 'zaxis')]
    ax = threed[0] if threed else fig.axes[0]
    seen = []
    for f in range(n):
        ani._func(f, *ani._args)
        seen.append(ax.get_title())
    return seen


# --- serial reveal ----------------------------------------------------------

def test_title_list_tracks_the_revealed_dataset():
    fig, ani = hyp.plot(_datasets(), '-', animate=True, order='serial',
                        title=['first', 'second', 'third'],
                        duration=4, frame_rate=4, show=False)
    seen = _titles_over(ani, fig, 16)
    assert seen[0] == 'first', 'frame 0 must title the FIRST dataset'
    assert set(seen) <= {'first', 'second', 'third'}
    assert seen.index('second') < seen.index('third')


def test_title_list_matches_the_published_current_index():
    """The title must be driven by the same schedule on_frame publishes."""
    names = ['first', 'second', 'third']
    seen_ctx = []
    fig, ani = hyp.plot(_datasets(), '-', animate=True, order='serial',
                        title=names, duration=4, frame_rate=4,
                        on_frame=seen_ctx.append, show=False)
    titles = _titles_over(ani, fig, 16)
    assert [names[c.current_index] for c in seen_ctx] == titles


def test_title_list_length_must_match_dataset_count():
    with pytest.raises(ValueError, match='title has 2 entries'):
        hyp.plot(_datasets(), '-', animate=True, order='serial',
                 title=['a', 'b'], duration=2, frame_rate=2, show=False)


def test_scalar_title_is_constant_across_a_serial_animation():
    fig, ani = hyp.plot(_datasets(), '-', animate=True, order='serial',
                        title='constant', duration=2, frame_rate=4,
                        show=False)
    assert set(_titles_over(ani, fig, 8)) == {'constant'}


def test_title_list_still_rejected_for_parallel_animations():
    with pytest.raises(TypeError, match='title must be a string'):
        hyp.plot(_datasets(), '-', animate=True, title=['a', 'b', 'c'],
                 duration=2, frame_rate=2, show=False)


def test_title_list_still_rejected_for_static_plots():
    with pytest.raises(TypeError, match='title must be a string'):
        hyp.plot(_datasets(), '-', title=['a', 'b', 'c'], show=False)


def test_title_list_works_for_2d_serial_animations():
    fig, ani = hyp.plot(_datasets(), '-', ndims=2, animate=True,
                        order='serial', title=['first', 'second', 'third'],
                        duration=4, frame_rate=4, show=False)
    seen = _titles_over(ani, fig, 16)
    assert seen[0] == 'first' and 'third' in seen


# --- morph: holds named, transitions blank ---------------------------------

def test_morph_titles_follow_the_hold_transition_schedule_exactly():
    """C9: derived from frame_to_segment, so it CANNOT pass under v1's
    fraction rule (which blanked 12 of 15 hold frames and named 4 transition
    frames while still landing at blank_fraction == 0.5)."""
    names = ['alpha', 'beta', 'gamma']
    fig, ani = hyp.plot(_clouds(), '.', animate='morph', title=names,
                        morph_samples=120, duration=6, frame_rate=4,
                        show=False)
    counts = segment_frame_counts(3, 24)
    assert counts == [5, 5, 5, 5, 4]
    seen = _titles_over(ani, fig, sum(counts))
    for frame, title in enumerate(seen):
        seg, step, n_steps = frame_to_segment(counts, frame)
        expected = names[seg // 2] if seg % 2 == 0 else ''
        assert title == expected, (frame, seg, step, n_steps, title)


def test_every_interior_transition_frame_is_blank():
    """The weaker property stated on its own, so intent stays legible."""
    fig, ani = hyp.plot(_clouds(), '.', animate='morph',
                        title=['alpha', 'beta', 'gamma'], morph_samples=120,
                        duration=6, frame_rate=4, show=False)
    counts = segment_frame_counts(3, 24)
    seen = _titles_over(ani, fig, sum(counts))
    interiors = [f for f in range(len(seen))
                 if frame_to_segment(counts, f)[0] % 2 == 1
                 and 0 < frame_to_segment(counts, f)[1]
                 < frame_to_segment(counts, f)[2] - 1]
    assert interiors, 'the schedule must contain interior transition frames'
    assert all(seen[f] == '' for f in interiors)


def test_every_hold_frame_is_named():
    fig, ani = hyp.plot(_clouds(), '.', animate='morph',
                        title=['alpha', 'beta', 'gamma'], morph_samples=120,
                        duration=6, frame_rate=4, show=False)
    counts = segment_frame_counts(3, 24)
    seen = _titles_over(ani, fig, sum(counts))
    holds = [f for f in range(len(seen))
             if frame_to_segment(counts, f)[0] % 2 == 0]
    assert len(holds) == 14
    assert all(seen[f] != '' for f in holds)


# --- backend parity ---------------------------------------------------------

def test_serial_titles_render_on_plotly_frames():
    pytest.importorskip('plotly')
    hyp.set_interactive_backend('plotly')
    try:
        fig = hyp.plot(_datasets(), '-', animate=True, order='serial',
                       title=['first', 'second', 'third'],
                       duration=4, frame_rate=4, show=False)
    finally:
        hyp.set_interactive_backend('matplotlib')
    titles = [f.layout.title.text for f in fig.frames]
    assert titles[0] == 'first'
    assert set(titles) <= {'first', 'second', 'third'}
    assert titles.index('second') < titles.index('third')


def test_morph_titles_match_across_backends():
    """The same call must produce the same per-frame title sequence."""
    pytest.importorskip('plotly')
    names = ['alpha', 'beta', 'gamma']
    fig, ani = hyp.plot(_clouds(), '.', animate='morph', title=names,
                        morph_samples=120, duration=6, frame_rate=4,
                        show=False)
    mpl_titles = _titles_over(ani, fig, 24)

    hyp.set_interactive_backend('plotly')
    try:
        pfig = hyp.plot(_clouds(), '.', animate='morph', title=names,
                        morph_samples=120, duration=6, frame_rate=4,
                        show=False)
    finally:
        hyp.set_interactive_backend('matplotlib')
    ply_titles = [f.layout.title.text for f in pfig.frames]
    assert ply_titles == mpl_titles


def test_serial_titles_compose_with_chemtrails():
    fig, ani = hyp.plot(_datasets(), '-', animate=True, order='serial',
                        chemtrails=True, title=['first', 'second', 'third'],
                        duration=4, frame_rate=4, show=False)
    seen = _titles_over(ani, fig, 16)
    assert seen[0] == 'first' and 'third' in seen
```

- [ ] **Step 2: Run the test and confirm it fails**

Run: `.venv/bin/python -m pytest tests/plot/test_serial_titles.py -v`

Expected: 14 collected. `test_title_list_still_rejected_for_parallel_animations`, `test_title_list_still_rejected_for_static_plots` and `test_scalar_title_is_constant_across_a_serial_animation` PASS (Task 1 already enforces the first two). Every test passing a `title=` list to a serial-style animation FAILS with `TypeError: title must be a string`. **3 passed, 11 failed.**

- [ ] **Step 3: Allow sequences for serial-style modes**

Widen `_validate_title`'s **behaviour** (its Task 1 signature is unchanged) in `hypertools/plot/plot.py`:

```python
#: Resolved animate modes for which a per-dataset `title=` sequence means
#: "name each segment while it is the one being shown".
_SERIAL_TITLE_STYLES = ('serial', 'morph')


def _validate_title(title, style=None, order=None, n_datasets=None):
    """`title=` is one string for the whole figure, or -- for serial-style
    animations -- one string per dataset, shown while that dataset is the one
    being revealed (and blanked through morph transitions, so only fully
    formed clouds are named).

    Returns None for the scalar/None forms, or a list of per-segment strings.
    """
    if title is None or isinstance(title, str):
        return None
    serial_style = (_raw_animate_style(style) in _SERIAL_TITLE_STYLES
                    or order == 'serial')
    if not serial_style:
        raise TypeError(
            f"title must be a string (or None), not {type(title).__name__}. "
            "Per-dataset titles are only meaningful for serial-style "
            "animations (order='serial' or animate='morph'). For a "
            "per-dataset legend entry use names=; for a per-observation "
            "annotation use labels=."
        )
    titles = [str(t) for t in title]
    if n_datasets is not None and len(titles) != n_datasets:
        raise ValueError(
            f"title has {len(titles)} entries but there are {n_datasets} "
            "datasets to plot; pass a single string for a fixed title, or "
            "one string per dataset.")
    return titles
```

The fail-fast call at `plot.py:2231` now passes the resolved order and gets the type check for free (`n_datasets` is unknown there, so the length check is deferred):

```python
    _segment_titles = _validate_title(title, style=animate, order=order)
```

and a second call after `plot.py:3653`, where `len(xform)` exists, performs the length check and re-resolves against the folded mode:

```python
    _segment_titles = _validate_title(title, style=animate, order=order,
                                      n_datasets=len(xform))
    if _segment_titles is not None:
        title = None      # the axes title is driven per frame, not statically
```

- [ ] **Step 4: Drive the title from the published schedule**

Register an internal callback on the **same** `FrameHooks` registry Task 7 created, so there is exactly one per-frame path. Add beside `_apply_multicolor_animation` in `plot.py`:

```python
def _make_title_updater(titles, axes):
    """Set the axes title from the frame context.

    Morph transitions are blanked so only fully-formed clouds are named. The
    discriminator is `segment_kind` (from `morph.frame_to_segment`'s segment
    PARITY), never `current_fraction`: holds and transitions both sweep 0->1
    over their own segment, so a fraction cannot tell them apart.
    """
    def _update(ctx):
        if ctx.segment_kind == 'transition':
            axes.set_title('')
            return
        idx = ctx.current_index
        if idx is None:
            return
        axes.set_title(titles[min(idx, len(titles) - 1)])
    return _update
```

and, immediately after the hook dispatcher is installed in Step 6 of Task 7:

```python
            if _segment_titles is not None and line_ani is not None:
                _frame_hooks.add(_make_title_updater(_segment_titles, ax))
```

For plotly, pass `segment_titles=_segment_titles` into `plotly_draw` (`plot.py:4206-4246`) and on into `_add_animation`, and set each frame's title from the same rule. In the `'serial'` branch (Task 4's rewrite) add, after `head_bounds_by_index` is complete:

```python
            if segment_titles is not None:
                _shown = [int(np.clip(revealed - s, 0, L))
                          for s, L in zip(starts, lengths)]
                _idx, _ = serial_current_index(_shown, lengths)
                frame_kwargs.setdefault('layout', {})['title'] = dict(
                    text=segment_titles[min(_idx, len(segment_titles) - 1)])
```

and in the `'morph'` branch (`plotly_backend.py:2773-2819`), where `seg_idx` is already in hand:

```python
            if segment_titles is not None:
                _text = ('' if seg_idx % 2 else
                         segment_titles[min(seg_idx // 2,
                                            len(segment_titles) - 1)])
                frame_kwargs.setdefault('layout', {})['title'] = dict(text=_text)
```

`serial_current_index` is imported from `matplotlib_backend` (the same helper Task 7 Step 4 introduced, under exactly that name — **no leading underscore**; it is the same import Task 7 Step 6a already adds to this branch) so the two backends share one rule. `lengths` and `starts` are already bound in this scope — `plotly_backend.py:2823-2825` computes `lengths = [np.atleast_2d(a).shape[0] for a in data]` and `starts = np.concatenate([[0], np.cumsum(lengths)[:-1]])` immediately above the frame loop (verified against source 2026-07-30), so do **not** recompute them.

- [ ] **Step 5: Run the test and confirm it passes**

Run: `.venv/bin/python -m pytest tests/plot/test_serial_titles.py -v`
Expected: **13 passed.** (v2 said 14; see *Suite arithmetic*.)

- [ ] **Step 6: Confirm the earlier contracts still hold**

Run: `.venv/bin/python -m pytest tests/plot/test_title_validation.py tests/plot/test_order_kwarg.py tests/plot/test_on_frame_hook.py tests/plot/test_plotly_serial_parity.py -v`
Expected: **9 + 20 + 24 + 10 = 63 passed.** `test_non_string_title_raises_rather_than_stringifying` must still pass — those calls are static.

- [ ] **Step 7: Update the docstring**

Extend the `title` entry written in Task 1:

```
    title : str or list of str
        A title for the plot. Normally a single string. For serial-style
        animations (``order='serial'``, ``animate='serial'`` or
        ``animate='morph'``) you may pass one string per dataset: each is
        shown while its dataset is the one being revealed, and morph
        TRANSITIONS show a blank title so only fully-formed clouds are named
        (a hold and a transition both progress 0 -> 1, so the distinction is
        the segment itself, not how far through it you are). Anywhere else a
        non-string raises ``TypeError``: use ``names=`` for per-dataset
        legend entries, or ``labels=`` for per-observation annotations.
        Rendered identically on the matplotlib and plotly backends.
```

- [ ] **Step 8: Run the FULL suite (central dispatch changed)**

Run: `.venv/bin/python -m pytest -q`
Expected: `2685 passed, 13 skipped`.

- [ ] **Step 9: Commit**

```bash
git add hypertools/plot/plot.py hypertools/plot/plotly_backend.py \
        tests/plot/test_serial_titles.py
git commit -m "feat(plot): per-segment titles for serial-style animations, on both backends"
```

---

## Task 9: CHANGELOG, the animation guide, and example cleanup

- [ ] **Step 1: Add the 1.1 entries to CHANGELOG.md**

```markdown
## 1.1.0 (unreleased)

### Added
- `order='parallel'|'serial'` on `plot()`, orthogonal to `animate=`, so trail
  styles compose with a serial reveal (`animate=True, order='serial',
  chemtrails=True`). `animate='serial'` remains a permanent alias, and
  `animate='morph'` is inherently serial. Resolved into the backend mode, so
  hue overlays and trail handling stay in sync.
- Per-dataset `alpha=`, alongside the existing per-dataset `color=`/
  `linewidth=`. Inputs that assign alpha internally (row MultiIndex, nested
  lists) keep their own values and now say so instead of losing silently.
- Public `on_frame=` hook (**both backends**) and, on matplotlib,
  `HyperAnimation.on_frame()` for attaching after construction — not
  available on plotly, whose animated return is a plain `go.Figure` whose
  frames are already built. Both give a
  `FrameContext` with the frame index, axes, drawn artists, animated arrays,
  the serial-reveal counts, and -- for morphs -- `segment_index`/
  `segment_kind`. Replaces reaching into `FuncAnimation._func`. Works on
  **both** backends and yields the same per-frame context metadata on each,
  but the call schedules differ -- matplotlib calls back at render time (a
  frame index may recur across a loop or a save), plotly exactly once per
  frame while the frames are built. Callbacks must be deterministic and
  idempotent for a given frame context. They must not depend on call count,
  call order, wall-clock time, or accumulated external state. Mutating artists is supported and expected; accumulating is not.
  `ctx.figure`/`ctx.axes`/`ctx.artists` are backend-native (`ctx.axes` is
  `None` on plotly, whose `ctx.artists` are traces), so a callback that
  mutates them is not portable across backends.
- Per-segment `title=` for serial-style animations, blanking morph
  transitions, on both backends.
- `simplify=` on `plot()` (default `True`). Today it governs
  `animate='morph'` tractability only: over clouds larger than 2000 points
  an uncapped morph is downsampled to 2000 **silently**, because the
  alternative is a render that never finishes (measured: killed at 10
  minutes uncapped; 8.2 s at `morph_samples=2000`). Pass `simplify=False`
  for an explanatory `ValueError` instead, which restores the guarantee that
  no real data point is ever dropped. An explicit `morph_samples=` always
  wins, and below the threshold `simplify` does nothing at all.

### Changed
- Animated continuous-hue line plots with no explicit `linewidth=` now
  render at `1.0` instead of `1.5`. This is a **visible change to existing
  animated hue figures**: the overlay now matches the width of the artist it
  replaces, which is what animated no-hue lines already used, so hue and
  no-hue animations finally agree. Pass `linewidth=1.5` to keep the old
  look.

### Fixed
- `animate='serial'` now composes with `chemtrails=`/`precog=`/`bullettime=`
  on the **plotly** backend, matching matplotlib frame for frame. It used to
  warn and drop the trails.
- `title=` no longer stringifies a list onto the axes; non-strings raise, and
  the check runs before the analyze pipeline (so streaming plots get it too).
- `linewidth=` is honored in animated continuous-hue line plots; the overlay
  now always renders at the width of the artist it replaces (previously
  `rcParams['lines.linewidth']`).
- `animate='morph'` over clouds larger than 2000 points no longer appears to
  hang: it is capped at 2000 by default, or raises naming `morph_samples=`
  and `simplify=True` when you pass `simplify=False`. See `simplify=` above
  for which of your data actually reaches the plot.
```

- [ ] **Step 2: Write the animation guide — `docs/animation.rst`**

This is a **new file** and a real deliverable of this task. Until now the plan referred to "the guide" in several places (the `on_frame` contract, Task 7 Step 8) without anything creating it; those references resolve here.

Create `docs/animation.rst`. It follows `docs/pipeline_order.rst`'s shape — a `.. _label:` anchor, a title, narrative prose with runnable snippets:

```rst
.. _animation:

Animating plots
===============

Every animation in HyperTools comes out of one call: ``hypertools.plot``
with ``animate=``. This guide covers what you can vary -- the animation
style, the order data is revealed in, trails, titles, per-dataset styling,
and per-frame callbacks -- and what differs between the matplotlib and
plotly backends.

Style and order are independent
-------------------------------

``animate=`` names a *style*; ``order=`` names the *ordering*. They are
orthogonal, which is new in 1.1:

.. code-block:: python

    import hypertools as hyp

    data = hyp.load('weights')

    hyp.plot(data, '-', animate=True)                      # parallel reveal
    hyp.plot(data, '-', animate=True, order='serial')      # one at a time
    hyp.plot(data, '-', animate='spin')                    # rotate, no reveal

``animate='serial'`` remains a permanent alias for
``animate=True, order='serial'``, and ``animate='morph'`` is inherently
serial. ``order='serial'`` has no meaning for ``'spin'`` or ``'window'``;
passing it there warns and is ignored, rather than raising.

Trails
------

``chemtrails=``, ``precog=`` and ``bullettime=`` leave a visible history
behind (or ahead of) the moving head. As of 1.1 they compose with a serial
reveal on **both** backends:

.. code-block:: python

    hyp.plot(data, '-', animate=True, order='serial', chemtrails=True)

Titles that change with the animation
-------------------------------------

Pass a **list** of strings as ``title=`` to name each segment of a
serial-style animation. For a morph, the holds are named and the
transitions are left blank automatically:

.. code-block:: python

    hyp.plot([a, b, c], '-', animate='morph',
             title=['first', 'second', 'third'])

Anywhere else a non-string ``title=`` raises ``TypeError``. Use ``names=``
for per-dataset legend entries and ``labels=`` for per-observation
annotations.

Per-dataset styling
-------------------

``color=``, ``linewidth=`` and -- new in 1.1 -- ``alpha=`` accept one value
per dataset:

.. code-block:: python

    hyp.plot([a, b, c], '-', animate=True,
             color=['red', 'blue', 'green'], alpha=[1.0, 0.6, 0.3])

Some inputs assign alpha internally (row-MultiIndex frames, nested lists).
Those keep their own values and say so with a warning rather than silently
discarding yours.

Large morphs and ``simplify=``
------------------------------

Morphing clouds larger than about 2000 points is intractable to render.
``simplify=True`` (the default) silently downsamples them so the render
finishes. Pass ``simplify=False`` to get a ``ValueError`` instead, which
restores the guarantee that no real data point is ever dropped:

.. code-block:: python

    hyp.plot(big_clouds, animate='morph')                   # downsampled
    hyp.plot(big_clouds, animate='morph', simplify=False)   # raises
    hyp.plot(big_clouds, animate='morph', morph_samples=500) # you decide

An explicit ``morph_samples=`` always wins, and below the threshold
``simplify`` does nothing at all.

Per-frame callbacks
-------------------

``on_frame=`` runs your function once per frame with a
:class:`~hypertools.FrameContext`. **Passing it to** ``plot()`` **works on
both backends** and is the portable form:

.. code-block:: python

    def label_frame(ctx):
        # ctx.frame and ctx.n_frames are backend-independent
        print(f'frame {ctx.frame} of {ctx.n_frames}')

    hyp.plot(data, '-', animate=True, on_frame=label_frame)

The context carries the frame index and total, the resolved ``style`` and
``order``, the arrays being drawn, the serial-reveal counts, and -- for
morphs -- ``segment_index`` and ``segment_kind``. All of those are the same
on either backend.

What you *do* with the context is usually backend-specific, because
``ctx.figure``, ``ctx.axes`` and ``ctx.artists`` are backend-native. The
matplotlib form:

.. code-block:: python

    # MATPLOTLIB ONLY -- ctx.axes is None on plotly
    def annotate(ctx):
        ctx.axes.set_title(f'frame {ctx.frame} of {ctx.n_frames}')

    fig, ani = hyp.plot(data, '-', animate=True, on_frame=annotate)

and the plotly equivalent, which reaches the frame's traces instead:

.. code-block:: python

    # PLOTLY ONLY -- ctx.artists are that frame's traces
    def rename(ctx):
        ctx.artists[0].name = f'frame {ctx.frame}'

    hyp.set_interactive_backend('plotly')
    fig = hyp.plot(data, '-', animate=True, on_frame=rename)

.. _animation-artist-lifetime:

Artist lifetime: what ``ctx.artists`` actually hands you
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Whether ``ctx.artists`` holds fresh objects each frame or the *same*
objects re-delivered depends on the backend and the style:

.. list-table::
   :header-rows: 1

   * - backend / style
     - lifetime
   * - matplotlib, **all** styles
     - shared live artists, mutated in place on every render
   * - plotly ``animate='spin'`` (no surfaces)
     - shared figure traces
   * - plotly ``animate='spin'`` (surfaced)
     - shared traces, then that frame's ``Mesh3d`` updates
   * - plotly parallel / serial / window / morph
     - per-frame trace payloads

Matplotlib never hands you a fresh artist. ``FuncAnimation``'s updater
mutates the same ``Line2D`` and collection objects every frame, so
``ctx.artists[0]`` on frame 1 and on frame 2 are the *same* object in two
different states. Plotly's spin is the same story for a different reason:
it moves only the camera and re-sends no point data, so its frames share
the figure's traces.

**The rule that follows applies to both backends: assign the complete
value you want on every invocation, including the default** -- never
write the attribute on some frames and leave it untouched on others. The
rule is portable; the *reason* is not, and the two failure modes are
opposite.

Where artists are **shared**, anything you set persists until something
overwrites it::

    # MATPLOTLIB ONLY (set_color is a matplotlib Artist method).
    # Shared artists, so this colours the WHOLE animation, not frame 0.
    def broken(ctx):
        if ctx.frame == 0:
            ctx.artists[0].set_color('red')

    def correct(ctx):
        ctx.artists[0].set_color(COLOURS[ctx.frame])   # set it every frame

Where they are **per-frame**, the very same conditional does the opposite
-- it touches an independent payload that only that frame keeps::

    # PLOTLY ONLY -- ctx.artists are that frame's traces, and
    # parallel/serial/window/morph frames are independent, so this
    # colours ONLY frame 0.
    def also_broken(ctx):
        if ctx.frame == 0:
            ctx.artists[0].line.color = 'red'

    def also_correct(ctx):
        ctx.artists[0].line.color = COLOURS[ctx.frame]

Writing a callback as though each frame had its own artists is the common
mistake, and writing one as though they were shared is the mirror image of
it. Under matplotlib and under plotly spin there is only ever one object,
so a conditional mutation looks like it "sticks" -- because it does. Under
plotly's other styles it silently does not.

This is also why *"a mutation is retained in the rendered frame"* does not
mean artists are isolated per frame. It means the backend renders what you
set; where artists are shared it renders it for every later frame too. A
surfaced spin is the mixed case: its ``Mesh3d`` updates trail the shared
traces in ``ctx.artists`` and those trailing entries *are* per-frame.

Highlighting exactly one frame is a perfectly good thing to want, and none
of this forbids it. Put the condition in the **value**, not around the
call, so the attribute is still assigned on every frame::

    HIGHLIGHT, DEFAULT = 'red', 'steelblue'

    def highlight_one_frame(ctx):                   # correct on both backends
        colour = HIGHLIGHT if ctx.frame == TARGET else DEFAULT
        ctx.artists[0].set_color(colour)            # matplotlib spelling

Assign on every invocation and none of this can bite you.

.. _animation-post-construction:

Registering after construction is matplotlib-only
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

On matplotlib you can attach a callback to an animation you already have::

    anim = hyp.plot(data, '-', animate=True)   # a HyperAnimation
    anim.on_frame(annotate)                     # fires on subsequent draws

**This is not available on plotly, and cannot be.** An animated matplotlib
plot returns a :class:`~hypertools.HyperAnimation`, whose frames are drawn
lazily at render time -- so there is still a window in which to register.
An animated plotly plot returns a plain ``plotly.graph_objects.Figure``:
its frames are *already built* by the time ``plot()`` returns, so there is
no later frame to call back into, and the returned object has no
``.on_frame()`` method.

If you are writing backend-portable code, **pass the callback to**
``plot()``. That is the form that works everywhere:

.. code-block:: python

    hyp.plot(data, '-', animate=True, on_frame=my_callback)   # both backends

    anim = hyp.plot(data, '-', animate=True)                  # matplotlib only
    anim.on_frame(my_callback)

.. _animation-callback-contract:

The callback contract
~~~~~~~~~~~~~~~~~~~~~

**Callbacks must be deterministic and idempotent for a given frame
context. They must not depend on call count, call order, wall-clock time,
or accumulated external state.**

Mutating what the context hands you is the *point* of the hook and is fully
supported -- the example above sets a title on every frame. What is
unsupported is **accumulation**::

    def ok(ctx):                     # idempotent: same frame, same result
        label.set_text(TITLES[ctx.frame])

    def broken(ctx):                 # accumulates: a repeated frame drifts
        ctx.artists[0].set_alpha(ctx.artists[0].get_alpha() * 0.9)

If you need a running quantity, precompute it once and index it by
``ctx.frame``::

    ACC = compute_running_accuracy(...)     # once, before plotting

    def show_accuracy(ctx):
        label.set_text(f'{ACC[ctx.frame]:.0f}%')

Backend scheduling
~~~~~~~~~~~~~~~~~~

The two backends call back on different schedules, and that is why the
contract exists:

.. list-table::
   :header-rows: 1

   * - backend
     - when it calls
     - how often per frame index
   * - matplotlib
     - at render time
     - **one or more times** -- a looping animation or a save replays frames
   * - plotly
     - at build time, before ``plot()`` returns
     - exactly once

Both backends deliver the same *context metadata* for a given frame index.
They do **not** produce interchangeable rendered output from a mutating
callback: ``ctx.figure``, ``ctx.axes`` and ``ctx.artists`` are
backend-native (on plotly ``ctx.axes`` is ``None`` and ``ctx.artists`` are
that frame's traces), so a callback that touches them is backend-specific
code. Each backend does guarantee that a mutation you make is retained in
the frame it renders.

Migrating from ``_func``/``_args``
----------------------------------

Before 1.1 the only way to run code per frame was to monkeypatch
matplotlib's private ``FuncAnimation._func`` and read ``_args``. That
reached into matplotlib internals, worked on one backend only, and broke
whenever the private signature changed. Replace it:

.. code-block:: python

    # before -- private, matplotlib-only
    _orig = ani._func

    def _wrapped(num, *args):
        out = _orig(num, *args)
        label.set_text(TITLES[num])
        return out

    ani._func = _wrapped

    # after -- public, both backends
    fig, ani = hyp.plot(data, '-', animate=True,
                        on_frame=lambda ctx: label.set_text(TITLES[ctx.frame]))

If you were re-deriving the serial reveal counts by hand from ``_args``,
use ``ctx.revealed_counts``; if you were computing which morph segment a
frame belonged to, use ``ctx.segment_index`` and ``ctx.segment_kind``
rather than thresholding ``ctx.current_fraction`` -- a hold and a
transition are not separable by fraction alone.
```

- [ ] **Step 3: Link the guide into the site navigation**

An unreferenced `.rst` in the source tree makes Sphinx warn *"document isn't included in any toctree"*, and Step 7 holds the repo's zero-warning standard — so this step is **not optional**. Add `animation` to the toctree in `docs/index.rst:41-48`, after `pipeline_order`:

```rst
.. toctree::
   :maxdepth: 2
   :caption: Contents:

   api
   pipeline_order
   animation
   tutorials
   auto_examples/index
```

- [ ] **Step 4: Write the guide's content and navigation tests**

Create `tests/test_animation_guide_docs.py`. These pin the guide against drift: they fail if the guide stops covering a documented feature, if it falls out of the toctree, or if it reintroduces the "pure function" framing this plan removed.

```python
"""The animation guide (docs/animation.rst) exists, is reachable, and
covers every animation feature 1.1 documents."""
import pathlib

import pytest

DOCS = pathlib.Path(__file__).resolve().parents[1] / 'docs'
GUIDE = DOCS / 'animation.rst'


def test_animation_guide_exists():
    assert GUIDE.is_file(), 'docs/animation.rst is a Task 9 deliverable'


def test_animation_guide_is_in_the_toctree():
    """Not decorative: an unreferenced .rst makes Sphinx warn, and the repo
    holds a zero-warning build standard."""
    index = (DOCS / 'index.rst').read_text()
    toctree = index.split('.. toctree::', 1)[1]
    entries = [ln.strip() for ln in toctree.split('\n\n')[1].splitlines()]
    assert 'animation' in entries, f'not in the toctree: {entries}'


@pytest.mark.parametrize('topic', [
    "order='serial'",      # ordering as its own axis
    'chemtrails',          # trails
    'title=',              # per-segment title sequences
    'simplify=',           # morph tractability
    'alpha=',              # per-dataset styling
    'on_frame=',           # the hook itself
    'FrameContext',        # the public context type
    'revealed_counts',     # the serial schedule it exposes
    'segment_kind',        # morph segment structure
    '_func',               # migration away from the private internals
])
def test_animation_guide_covers(topic):
    assert topic in GUIDE.read_text(), f'guide does not mention {topic}'


def test_animation_guide_documents_both_backend_schedules():
    text = GUIDE.read_text()
    assert 'matplotlib' in text and 'plotly' in text
    assert 'render time' in text
    assert 'build time' in text


def test_animation_guide_states_the_callback_contract_verbatim():
    """The one sentence that has to be identical in the guide, the plot()
    docstring and the CHANGELOG."""
    text = ' '.join(GUIDE.read_text().split())
    assert ('Callbacks must be deterministic and idempotent for a given '
            'frame context.') in text
    assert ('must not depend on call count, call order, wall-clock time, '
            'or accumulated external state.') in text


def test_animation_guide_does_not_call_the_contract_purity():
    """Regression guard. Callbacks mutate artists by design -- calling the
    contract 'purity' is the misstatement this plan's v4 removed, and the
    guide's own example sets a title every frame."""
    text = GUIDE.read_text().lower()
    assert 'pure function' not in text


def test_animation_guide_marks_post_construction_registration_matplotlib_only():
    """`HyperAnimation.on_frame()` cannot exist on plotly: animated plotly
    returns a plain go.Figure whose frames are already built when plot()
    returns (plot.py:4605-4612 -- only animated matplotlib sets line_ani).
    The guide must not present post-construction registration as portable."""
    text = ' '.join(GUIDE.read_text().split())
    assert 'Registering after construction is matplotlib-only' in text
    assert 'This is not available on plotly, and cannot be.' in text
    # and it must say what to do instead
    assert 'pass the callback to' in text.lower()


def test_animation_guide_labels_its_backend_specific_examples():
    """ctx.axes is None on plotly and ctx.artists are traces, so neither
    example is portable. Each must be labelled rather than sitting
    unmarked in a backend-general section."""
    text = GUIDE.read_text()
    assert '# MATPLOTLIB ONLY' in text
    assert '# PLOTLY ONLY' in text


def test_animation_guide_documents_artist_lifetime_for_both_backends():
    """Artists are SHARED on matplotlib (FuncAnimation mutates the same
    Line2D objects every render) and on plotly spin (camera-only frames).
    Only plotly's reveal/morph styles hand out per-frame trace payloads. A
    caller who assumes per-frame artists writes a conditional mutation that
    silently applies to the whole animation."""
    text = ' '.join(GUIDE.read_text().split()).lower()
    assert 'artist lifetime' in text
    assert 'matplotlib, **all** styles' in text or 'matplotlib, all' in text
    assert 'whole animation' in text or 'figure-wide' in text
    assert 'spin' in text
    # the corrected claim must not come back
    assert 'every style on matplotlib, is per-frame' not in text


def test_animation_guide_gives_both_failure_modes_not_just_persistence():
    """The guide must not say persistence applies to both backends.

    Measured 2026-07-30: plotly's parallel/serial/window/morph frames are
    INDEPENDENT payloads (`fig.frames[0].data[0] is not
    fig.frames[1].data[0]`), so a frame-0-only mutation there affects only
    frame 0 -- the opposite of matplotlib and plotly spin, where it affects
    everything. An earlier draft stated the shared behaviour as universal.
    Both modes must be present, and the persistence claim must be scoped.

    The rule must also stay stated as an ASSIGNMENT rule, not a ban on
    per-frame decisions: v4.3 said "never write a mutation that fires on
    one frame only", which forbids highlighting a single frame -- a
    legitimate thing to want, and portable when the condition sits in the
    value rather than around the call.
    """
    raw = GUIDE.read_text()
    # collapse whitespace AND strip rst emphasis, so the assertions below
    # survive `**shared**` being bolded or re-wrapped
    text = ' '.join(raw.replace('*', '').split()).lower()
    # both failure modes are described, not just the shared one
    assert 'whole animation' in text
    assert 'only frame 0' in text
    # and persistence is scoped to shared artists rather than to "both backends"
    assert 'where artists are shared' in text
    # the plotly example uses a real plotly API, not matplotlib's set_color
    assert '.line.color' in raw
    # the rule is about assigning every invocation, and single-frame
    # highlighting is shown as supported rather than forbidden
    assert 'assign the complete value' in text
    assert 'highlighting exactly one frame' in text
```

- [ ] **Step 5: Simplify the gallery examples that hand-rolled these primitives — MECHANICAL MIGRATION ONLY**

> **Ownership boundary — read before editing anything under `examples/`.** This step and Plan 4 both touch the same four files, so the split is explicit:
>
> | this step (Plan 1) | Plan 4 |
> |-|-|
> | **Mechanical migration off private internals only** — delete `_func`/`_args` monkeypatches and private `_morph`/`_shared` imports, replace them with the equivalent `title=` / `on_frame=` call | **All narrative, visualization and notebook work** — rewriting what the example *demonstrates*, its prose, its figures, and the paired `docs/tutorials/*.ipynb` |
> | Behaviour must be **unchanged**: the rewritten example renders what it rendered before | Behaviour changes freely — Plan 4 reframes several examples outright (e.g. Task 3 turns 6 city-datasets into 20 features of one trajectory) |
> | Module docstring: touch **only** the sentences that describe the private workaround being removed | Owns the full docstring/narrative rewrite |
> | **Do not** assert Plan 4's line-count or class-mix metrics here | Plan 4 Task 8 owns those metrics and measures them after its own rewrites |
>
> Rationale: doing narrative work here would be done twice and thrown away, and enforcing Plan 4's final metrics against a file Plan 4 has not rewritten yet would fail for the wrong reason. Plan 4's Tasks 2, 3, 5 and 6 are the authority for what these examples ultimately say.

`examples/animate_morph_zoo.py:99-115` re-implements per-segment titling with `_morph.morph_schedule`/`frame_to_segment` and a `_func` monkeypatch; `examples/animate_conversation.py`, `animate_market_forecast.py` and `animate_weather_decades.py` monkeypatch `_func` and read `ani._args`. Replace each with `title=[...]` and/or `on_frame=`, and delete the private imports. Where an example computes a running quantity, keep the existing precompute-then-index shape (`animate_market_forecast.py:255` builds `ACC`; the wrapper reads `ACC[frame]`) — that is already contract-compliant and needs no redesign.

- [ ] **Step 6: Verify every rewritten example still runs**

Run: `for f in examples/animate_conversation.py examples/animate_market_forecast.py examples/animate_morph_zoo.py examples/animate_weather_decades.py; do echo "== $f"; .venv/bin/python "$f" || break; done`
Expected: each exits 0 with no traceback and no `UserWarning` about ignored kwargs.

- [ ] **Step 7: Verify the docs build clean**

> **Use the CI command, not `make html`.** The command below is byte-for-byte what the `docs-clean` CI job runs (`.github/workflows/test.yml:283-291`), including **`-W`** — warnings are errors, which is the actual release gate and which `make html` does **not** apply. *Historical note:* through 2026-07-30 `make html` also failed outright with `ModuleNotFoundError: No module named '_gallery_log_filter'`, because the installed `sphinx-build` console script puts the venv `bin/` on `sys.path[0]` rather than `docs/`, while `python -m sphinx` puts the CWD there. **That root cause is now fixed** — `docs/conf.py` adds its own directory to `sys.path`, so `make html` builds again (verified: 0 warnings). The CI command stays mandatory here anyway, for `-W`.

Run: `cd docs && MPLBACKEND=Agg ../.venv/bin/python -m sphinx -b html -W -E -a . _build/html 2>&1 | tail -20`
Expected: build succeeds with **0 warnings** (the repo holds an RTD-parity zero-warning standard).

- [ ] **Step 8: Run the FULL suite one last time**

Run: `.venv/bin/python -m pytest -q`
Expected: `2704 passed, 13 skipped`.

- [ ] **Step 9: Commit**

```bash
git add CHANGELOG.md docs/animation.rst docs/index.rst \
        tests/test_animation_guide_docs.py examples/
git commit -m "docs(1.1): document order=, per-dataset alpha=, on_frame, per-segment titles; simplify examples"
```

---

## Self-Review

**Every review finding is mapped to the task that closes it.**

| finding | closed by |
|-|-|
| **C1** wrong `hue=` cardinality (30 vs 90) | Task 2 Step 1 — `_hue_for()` computes `sum(d.shape[0] for d in datasets)` |
| **C2** `_widths()` reads artists that pass on unfixed code | Task 2 Step 1 — `_overlay_widths()` selects only the `'_nolegend_'` head/trail collections (`plot.py:5172`); the static control selects the last `n` collections; measured red state `[1.5,1.5,1.5]`, green `[0.5,0.5,5.0]` |
| **C3** patch in the wrong file, undefined `dataset_kwargs`, kwarg collision | Task 2 Step 3 — one line in **`plot.py`** `_apply_multicolor_animation._linewidth`, reading `head_lines[i].get_linewidth()`; `matplotlib_backend.py` untouched. Verified by applying it |
| **C4** undefined `mode`/`datasets`; no hook point | Task 3 Step 3 — uses the real locals `xform` and `morph_tags`, inserted after `plot.py:3663` where both exist |
| **C5** serial gate unreachable for the list form of `animate=` | Task 5 Step 3 — `_raw_animate_style` maps any list/tuple to `'morph'` for the fail-fast check, and the fold in `_resolve_animate_mode` runs on the **resolved** mode. Tested by `test_per_dataset_morph_list_accepts_order_serial` |
| **C6** one-site `backend_mode` insufficient (4 consumers), incl. the plotly silent trail drop and the hue desync | Task 4 (plotly serial trails, so `_trail_ignoring_modes` needs no backend special case) + Task 5 (ordering folded **into** the mode, so `3760`/`4214`/`4299`/`4379` are all correct). Tested by `test_order_serial_matches_animate_serial_for_hue_overlays` and the two plotly parity tests |
| **C7** `HyperAnimation.on_frame()` can never fire (list identity) | Task 7 — `FrameHooks` created in `plot()`, threaded into `_draw`, **adopted** by `__new__`. Tested by `test_hook_can_be_attached_after_construction` (v1 would fail `0 != 2`) and `test_constructor_and_post_construction_callbacks_both_fire`. The `return_model=True` raw-`FuncAnimation` limitation is documented and tested |
| **C8** `current_fraction not in (0.0, 1.0)` inverts the requirement | Task 7 adds `segment_index`/`segment_kind` as explicit fields (parity from `morph.frame_to_segment`) and says so in the `current_fraction` docstring; Task 8 Step 4 uses `segment_kind` |
| **C9** the morph title test cannot detect C8 | Task 8 — `test_morph_titles_follow_the_hold_transition_schedule_exactly` asserts the exact expected title for **all 24 frames**, derived from `segment_frame_counts`/`frame_to_segment`, plus separate interior-transition and every-hold checks |
| **G1** undefined precedence vs the two internal `alpha` writers | Task 6 Step 3 — internal styling still wins, with a warning mirroring `plot.py:3045-3050`. Tested for both MultiIndex and nested-list inputs |
| **G2** `_expand_styles_to_runs` invariant broken; "which count" unspecified | Task 6 — count is `len(xform)` (final, like `surface_list`/`density_list` at `plot.py:3637-3643`); the `plot.py:242-244` docstring is corrected; `test_alpha_survives_contiguous_run_segmentation` covers run expansion |
| **G3** validation placement defeats fail-fast and misses `plot_stream` | Task 1 — validation at `plot.py:2231`, ahead of `resolve_font` (`2428`) and the `plot_stream` return (`2582`). Two tests: the reduce-error ordering test and the stream test |
| **G4** default cap contradicts the `morph.py:17-24` guarantee | Task 3 — resolved by the maintainer (2026-07-29) with an explicit `simplify=` flag rather than by picking a side: below the cap nothing happens; above it `simplify=True` (default) caps **silently** and `simplify=False` **raises** naming both `morph_samples=` and `simplify=True`. The contradiction is closed at the source — Task 3 Step 6 rewrites `morph.py:17-24` itself so the in-source guarantee states the condition it now depends on, and does the same for its restatement at `plot.py:1516-1518`. Every gallery example already passes `morph_samples=`, so no example changes either way. Decision #1, resolved |
| **G5** "one implementation" unachievable; 2-D gets no hook | Task 7 — `serial_reveal_counts`/`serial_current_index` are module-level and replace all three copies (`matplotlib_backend.py:1316-1318`, `:2062-2064`, `plot.py:5265-5269`); all seven updaters record; `_axes_of()` handles 2-D; `test_hook_fires_for_2d_animations` and `test_2d_animated_hue_honors_per_dataset_linewidth` drive 2-D |
| **G6** `order=` loses the `zorder` did-you-mean hint | Task 5 Step 3 — `_resolve_order` appends the hint for numeric values. `test_numeric_order_still_offers_the_zorder_hint` |
| **G7** `order='serial'` with `animate=False` gives a confusing `NotImplementedError` | Task 5 Step 3 — `ValueError: order='serial' requires an animated plot`, matching the `on_frame` error shape. Tested |
| **G8** `stacklevel=2` breaks the repo convention | Task 5 and Task 6 use `stacklevel=external_stacklevel()`; Task 3 raises rather than warns, so the issue does not arise |
| **T1** three `order=` tests vacuous (artist counts) | Task 5 — `_started()` counts head artists with vertices at frame 3: 3 for parallel, 1 for serial (measured `[247,247,247]` vs `[657,0,0]`), which artist counts (9 vs 9) cannot distinguish |
| **T2** "11 passed" wrong for a 12-case file | Recounted again for v3 by counting `def test_` and expanding each `parametrize` to its cases: Task 1 = 9, Task 2 = 5, Task 3 = **11**, Task 4 = **10**, Task 5 = 20, Task 6 = **10**, Task 7 = **24**, Task 8 = **13**; total 102, final `2653`. v2 itself still carried the v1 defect on four of these (it counted a parametrized def as both a def and its cases) — see *Suite arithmetic* for the per-task table and which differences are v3 content vs. v2 miscounting |
| **T3** fabricated red-state output | Every red state below was **run**: Task 1's stream/reduce failures, Task 2's `[1.5,1.5,1.5]`, Task 4's plotly warning + `4 != 7` traces, Task 5's `TypeError ... did you mean 'zorder'?` |
| **T4** reveal-formula test only exercises exact division | Task 7 — `test_revealed_counts_match_the_drawn_artists_with_unequal_lengths` uses lengths `[17, 23, 11]` with a **marker** format (a line format pre-interpolates to `[13,13,13]` — measured) and asserts against the drawn artists, not a second copy of the formula. Verified: simulated `[4,0,0]/[17,4,0]/[17,23,2]` at frames 1/5/10 equal the measured artist vertex counts exactly |
| **T5** no plotly/trail coverage for serial titles | Task 8 — `test_serial_titles_render_on_plotly_frames`, `test_morph_titles_match_across_backends`, `test_serial_titles_compose_with_chemtrails`. The docstring no longer needs a backend caveat because parity is implemented |
| **T6** morph tests assert only warning-absence | Task 3 — every test asserts the plot actually drew (`test_clouds_at_or_below_the_threshold_keep_every_point` asserts all 300 points survive; `test_explicit_morph_samples_is_respected_above_the_threshold`; `test_default_simplify_downsamples_silently_above_the_threshold` asserts `0 < drawn <= 2000` **and** `caught == []`, so a silent no-render cannot pass), and the timing assertion is outside any `pytest.warns` block. The default path is separately pinned as untouched by `test_simplify_is_a_no_op_below_the_threshold`, which compares drawn point counts across all three spellings of the flag |
| **T7** no `hue=` animation driven through `on_frame` | Task 7 — `test_hook_sees_post_multicolor_artists`, plus the design decision that the dispatcher is the outermost wrapper |
| **T8** duplicate module setup appended to an existing file | Task 2 — new file `tests/plot/test_animated_hue_linewidth.py` (the existing `test_matplotlib_backend_bugs.py` imports neither `pytest` nor `hypertools` and never sets `Agg`) |
| **Maintainer directive** plotly/matplotlib parity for `order='serial'` | Task 4 (implementation, via the same per-segment-alpha trail traces the parallel styles already use at `plotly_backend.py:950-966`) + Task 5 (`test_order_serial_is_identical_on_plotly`, `..._with_chemtrails_is_identical_on_plotly`) + Task 8 (title parity) |

**Placeholders.** None. Every step carries runnable code, a concrete command, and its expected output. No step says "similar to Task N".

**Type consistency.** `_resolve_animate_mode` returns a 3-tuple `(mode, morph_tags, order)` from Task 5 onward; it has exactly one caller (`plot.py:3653`) and no test calls it directly. `_validate_title` is introduced in Task 1 with its **final** signature `(title, style, order, n_datasets)` — Task 8 changes only its behaviour — so an implementer working tasks out of order never sees a signature change. `FrameContext` field names used in Task 8 (`current_index`, `segment_kind`) match the dataclass defined in Task 7.

**Task dependencies.** 1 → 8 (`_validate_title`); 4 → 5 (parity before the new spelling, so `order='serial'` never inherits a backend hole); 5 → 7 (`order` in `FrameContext`) → 8 (`FrameHooks` + `segment_kind`). Tasks 2, 3 and 6 are independent and may be done in any order. **4 → 7 as well, one-way:** Task 7 Step 6a adds the plotly hook dispatch to all four `_add_animation` frame-build sites, one of which is the serial branch Task 4 rewrites, so Task 4 must land first — and Task 4 explicitly does *not* anticipate the hook, so the dispatch block exists in exactly one place in this plan.

**Suite arithmetic (recomputed for v3 — v2's total was wrong).** Baseline `2554 passed, 13 skipped` (2567 collected). Counts below are `def test_` in each task, with every `@pytest.mark.parametrize` expanded to its case count — not estimates:

| task | `def test_` | parametrize | collected | v2 said |
|-|-|-|-|-|
| 1 | 6 | 1 def × 4 cases | **9** | 9 ✓ |
| 2 | 5 | — | **5** | 5 ✓ |
| 3 | 11 | — | **11** | 7 (v3 adds 4 `simplify` tests) |
| 4 | 7 | 1 def × 4 cases | **10** | 11 ✗ |
| 5 | 16 | 1 × 4, 1 × 2 | **20** | 20 ✓ |
| 6 | 10 | — | **10** | 11 ✗ |
| 7 | 30 | 1 × 4, 1 × 4, 1 × 6 | **41** | 20 (v3 dropped the plotly-raises test and added 3 defs incl. the ×4 parity case; v4 added two mutation-retention tests + the `FrameContext` export test; v4.1 added three plotly-spin artist tests; v4.2 added `test_matplotlib_artists_are_shared_across_frame_deliveries`; **v4.3** adds `test_plotly_non_spin_frames_are_isolated_per_frame` ×4 and `test_frame_context_containers_are_canonical_tuples` ×6) |
| 8 | 13 | — | **13** | 14 ✗ |
| 9 | 10 | 1 def × 10 cases | **19** | 0 (**new in v4**: `tests/test_animation_guide_docs.py`; **v4.1** adds three more — post-construction qualification, backend-labelled examples, spin's shared artists; **v4.3** adds `test_animation_guide_gives_both_failure_modes_not_just_persistence`, and corrects the def count, which read 8 for a file that has always held 9) |

Added: 9 + 5 + 11 + 10 + 20 + 10 + **41** + 13 + **19** = **138**. Final expected: `2692 passed, 13 skipped` (v4.4 baseline 2554). *(v3 totalled 102 → 2,653; v4 → 120 → 2,671; v4.1 → 126 → 2,677; v4.2 → 127 → 2,678; v4.3 adds the isolation and container-type guards.)* Each task's Step "run the FULL suite" states its own running total, so a drift is caught at the task that caused it.

**A v4.3 correction to this table itself.** Task 9's `def test_` column read **8** while its test file has always contained **9** defs — the *collected* figure (18) was right, so no total was ever wrong, but the middle column contradicted this table's own stated method (*"`def test_` in each task … not estimates"*). Counted by name: `..._exists`, `..._is_in_the_toctree`, `..._covers` (the ×10 parametrize), `..._documents_both_backend_schedules`, `..._states_the_callback_contract_verbatim`, `..._does_not_call_the_contract_purity`, `..._marks_post_construction_registration_matplotlib_only`, `..._labels_its_backend_specific_examples`, `..._documents_artist_lifetime_for_both_backends` = 9, plus v4.3's new one = **10**, giving 9 plain + 10 cases = **19**.

The three ✗ rows are a **v2 counting error, not a v3 change**: v2 counted a parametrized def as both one def *and* its cases (visible in its own Task 4 breakdown, *"4 plain + 4 parametrized + 3 plain"* for a file with 7 defs), and over-counted Tasks 6 and 8 by one each with no parametrization present to explain it. Nothing about those tasks' contents changed in v3.

**Remaining risk.** Task 4 is the largest single diff (a rewritten plotly frame loop) and the likeliest to disturb existing figures; `test_plain_serial_parity_is_unchanged` and `test_parallel_trail_parity_is_unchanged` are the guards, and the branch is only entered for `animate == 'serial'`. Task 7 touches all seven matplotlib updaters, the return path, and now four plotly frame-build sites; if it grows beyond one reviewable diff, split it into "publish the schedule from one helper" (Steps 4–5), "wire the matplotlib hook" (Step 6) and "wire the plotly hook" (Step 6a) as separate commits, running the full suite after each. Step 6a is the lowest-risk of the three — it only *adds* a `record`/`dispatch` pair before existing `frames.append` calls and changes no frame content — but it is the one whose per-branch field values must match Step 5's exactly, which `test_on_frame_context_metadata_parity_across_backends` is there to catch.

---

## Decisions (all resolved)

**Nothing here is open.** All four of v2's flagged decisions were resolved by the maintainer on 2026-07-29 and are implemented in the tasks above; this section is the record, kept so a reader can see what was chosen and how to reverse it. Nothing in this plan is waiting on anyone.

> **These entries are deliberately UNNUMBERED — cite them by name.** Four separate instances of citation drift in this plan set traced to numeric references going stale when a list was reordered or an item removed. The plan-set README's open-decision list was de-numbered for the same reason. Refer to *"the animated-hue linewidth decision"*, not *"#1"*.

- **Animated-hue default linewidth: 1.5 → 1.0.** Task 2's fix reads the width off the hidden head artist, which for a caller who passed no `linewidth=` is `1.0` (the backend's `pop("linewidth", 1)` default at `matplotlib_backend.py:1603`) rather than `1.5` (`rcParams['lines.linewidth']`). Measured today, the animated **no-hue** lines are already `1.0`, so this makes hue and no-hue animations agree — but it is a visible change to existing hue animations.
   - **RESOLVED: ship it.** No implementation change; Task 2 already does this. Pinned by `test_animated_hue_default_width_matches_the_artist_it_replaces`, and recorded under **Changed** in the CHANGELOG (Task 9) as a visible change to existing animated hue figures, with `linewidth=1.5` named as the way to keep the old look.
   - *Not taken:* also changing `matplotlib_backend.py:1603` / `:2198` to `pop("linewidth", plt.rcParams['lines.linewidth'])`, making **all** animated lines `1.5` and matching static plots. Broader blast radius; would need its own full-suite pass.

- **`morph_samples` above 2000: cap or refuse?** `morph.py:17-24` guarantees no real data point is ever dropped, and `tests/test_morph_animation.py:121-131` encodes the uncapped default; v1 capped silently (an error), v2 raised unconditionally.
   - **RESOLVED: a `simplify=` flag decides, and the default is to cap.** Verbatim: *"add a 'simplify' flag to control this behavior; if below cap, simplify does nothing. otherwise either silently drop with no warning if simplify=True (default), or print an informative message with a suggestion to set simplify=True and then raise an exception if simplify=False."* Implemented in Task 3: no-op below the cap; **silent** downsample above it by default (no `warnings.warn`, no `print`); `ValueError` naming `simplify=True` when `simplify=False`. Contract 7 is rewritten to the conditional guarantee, and Task 3 Step 6 rewrites `morph.py:17-24` and `plot.py:1516-1518` so no in-source guarantee outlives it.
   - *Scope:* `simplify=` governs morph tractability **only**, and is documented as such.
   - *To make the message print separately as well:* prepend `print(_msg)` before the `raise` in Task 3 Step 3 and add a `capsys` assertion to `test_simplify_false_over_the_threshold_raises_naming_simplify`. See the *Revision note (v3)* for why one `raise` carries it instead.

- **`order='serial'` with `animate='spin'` or `'window'`: warn-and-ignore vs. hard error.** v1 raised `NotImplementedError`; the review noted this is a *new* hard error where the repo's established behaviour for the same shape of request is warn-and-ignore (`plot.py:3760-3781`, measured: `animate='spin', chemtrails=True` → *"animate='spin' does not support trail styles; ignoring chemtrails for datasets [0, 1, 2]"*).
   - **RESOLVED: unchanged — warn and ignore**, matching the established convention. `test_serial_ordering_warns_and_is_ignored_for_spin_and_window`.
   - *To switch to a hard error:* in Task 5 Step 3's fold, replace the `warnings.warn(...)`/`order = 'parallel'` pair with a `raise NotImplementedError(...)`, and change that test to `pytest.raises`.

- **`FrameContext` exposes backend-native objects (`on_frame=` on the plotly backend).** v2 asserted this was the one place parity was unreachable, on the theory that a plotly animation is precomputed JSON with no Python per-frame loop.
   - **RESOLVED: the theory was wrong; `on_frame=` ships on both backends and the `NotImplementedError` is deleted.** `_add_animation` (`plotly_backend.py:2517`) builds every frame in a Python loop at build time — `frames = []` (`:2601`), `frames.append(go.Frame(**frame_kwargs))` at `:2729` (spin), `:2819` (morph), `:2865` (serial), `:2975` (parallel/window). What plotly lacks is a Python loop during *playback*. Task 7 Step 6a implements the dispatch across all four sites; Task 4 references it rather than duplicating it.
   - *The cost, stated rather than hidden:* the two backends call back on different **schedules** (matplotlib at render time, possibly repeating a frame index across a loop or a save; plotly exactly once per index at build time), so `on_frame` must be **deterministic and idempotent for a given `FrameContext`** — never described as "pure", since mutating artists is the entire point — and `figure`/`axes`/`artists` are backend-native. Both facts are in the docstring, the guide and the CHANGELOG. **Context-metadata** parity over the backend-independent fields is asserted by `test_on_frame_context_metadata_parity_across_backends`; the backend-native fields by `test_plotly_frame_context_carries_backend_native_objects`; and per-backend mutation *retention* by the Step 1b pair. **Output parity is explicitly not claimed** — artists and traces are backend-native, so a mutation callback is not source-compatible across backends and any such assertion would be false. This is v2's *Alternative A*, adopted with its parity hazard named and tested rather than assumed away.
