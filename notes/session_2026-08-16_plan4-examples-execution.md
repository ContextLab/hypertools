# Plan 4 (examples and tutorials) — execution record

Started 2026-08-16, on `dev-1.0`, after the maintainer's review round 7 directed
"begin Plan 4". Plan: `docs/superpowers/plans/2026-07-28-hypertools-1.1-examples-and-tutorials.md` (v4).

**Procedure chosen, explicitly** (the plan offers two and forbids half-doing
both): **Task 8 Steps 0–2 FIRST**, so Tasks 1–7 have somewhere to record their
measurements, and Task 8 Step 3's reds are the to-do list. The alternative —
Task 8 last — was rejected because every task's measure step writes into
`SCRIPT_BUDGETS` / `EXPECTED_VISIBLE_OUTPUTS`, which do not exist until Step 2.

## Order of work

| # | what | state |
|-|-|-|
| T8 S0 | `HyperAnimation.n_frames` / `.n_segments` / `.draw_frame` | **DONE** |
| T8 S0b | loader/builder split contract (text; Tasks 2–6 apply it) | read, not code |
| T8 S0c | budget renegotiation | already done 2026-08-04; verified, not repeated |
| T8 S1 | `scripts/measure_native_ratio.py` | **DONE** |
| T8 S2 | `tests/test_examples_are_native.py` (the gate) | **DONE — deliberately RED** |
| T1 | native palette-from-image | **DONE** |
| T2 | Market — the MultiIndex showcase | **IN PROGRESS** — library work done and committed; the artifact itself is NOT settled (see below) |
| T3–T7 | weather, paintings, conversation, morph, 15 older tutorials | pending |
| T8 S3–S7 | run the gate, re-measure, headless run, notebook figures, full suite | pending |

## Status, 2026-08-18/19 (maintainer review round 8)

Task 2 is partly executed. Everything that is a LIBRARY fix has landed as its
own commit; the Market artifact deliberately has not, because the evidence now
says its forecast framing is wrong and the representation is still under review.

| commit | unit |
|-|-|
| `3fe1c6a5` | tag animated multicolor collections with their trace |
| `6da372b4` | matrix hue through a column hierarchy, contract pinned + documented |
| `9615c382` | forecast projection sampled at two DISTINCT history lengths |
| `9d630c9f` | study: horizon-aware window baseline; rule applied in code |
| `fd46b7db` | correct the stale Market budget claim (gate stays red) |

**Deliberately uncommitted:** `examples/animate_market_forecast.py`,
`docs/tutorials/market_forecast.{ipynb,gif}`, `scripts/execute_tutorial.py`.
The staged GIF/notebook publish an artifact whose own supporting study now
says it should carry no forecast claim; committing them would ship two
contradictory conclusions in one tree.

**The editorial decision, and the evidence for it.** Applied in code, the
preregistered rule passes three specifications, all `drawdown` at h=1 — so
the earlier flat claim that "nothing passes" was too broad and is retracted
in the study notes. But the drawdown audit those notes pre-committed to
kills all three: a parameter-free "predict full recovery" rule beats the
models in 10 of 12 cells. **No specification earns a forecast claim at the
horizon the example draws.** Market should become the hierarchy showcase
with no forecast-skill claim; the prediction story moves to data with real
temporal structure. Awaiting the maintainer's sign-off before the rewrite.

## Review round 9 (2026-08-19)

| commit | unit |
|-|-|
| `d41fc265` | explicit `import numpy as np` in plot.py — ruff 186 → 28, zero net-new keys |
| `dfc8e7db` | `hue_mode=` — mixture vs RGB said outright instead of inferred from width |
| `1d220643` | acceptance rule hardened (positive + exact block set) + study made self-contained |

**`hue_mode=` exists because the previous commit broke its own use case.**
Mixture weights are one palette colour per column, one column per leaf —
but the routing rule sent any matrix wider than 3 columns to the RGB
reducer, so at most three leaves could ever get a primary. Measured on the
Market candidate (6 leaves, 4-colour palette): `palette=` was ignored and
leaves intended blue drew pure red. The width rule is now the DEFAULT, not
the only rule; flipping the default would silently repaint existing figures.

**The Market artifact was discarded**, not committed. Its script, notebook
and GIF were copied to a SESSION-SCOPED scratchpad —
`/private/tmp/claude-501/-Users-jmanning-hypertools/7e6531b3-.../scratchpad/discarded_market_artifact/`
— which is outside the repository and does not survive the session. Treat
the artifact as GONE: it is not recoverable from this note, and nothing
depends on recovering it (its representation was rejected, so what matters
is the measurements, which are in
`notes/market_representation_study_2026-08-17.md`). The earlier version of
this paragraph wrote that path relative, as `scratchpad/…`, which resolves
to nothing from the repo root and read as a promise of recoverability the
tree could not keep.

The prototypes ARE preserved, because a visual decision cannot be reviewed
from prose: `notes/evidence/plan4-market-prototypes/` holds both PNGs and
the script that makes them (3 s, live data with a synthetic fallback, and
it re-asserts the shared-normalization property on the way past).

Discarding the artifact broke the committed study, which imported the
example's universe and fetcher — fixed by lifting both in, because evidence
must not depend on the artifact it judges.

**Two library limits measured while prototyping the replacement**, both
relevant to any 2-D Market composition:

* `hyp.plot` draws in a NORMALIZED unit box (`xlim == [-1.1, 1.1]` whatever
  the data) and removes ticks by design. `xlabel=`/`ylabel=` are native and
  do render, so an axis can say what it IS but never what it equals.
* each call normalizes ITS OWN inputs, so N calls give N private scales.
  Small multiples are only comparable if every panel is passed the same
  data and differentiated by hue — verified: all six panels then share one
  limit tuple exactly.

**Prototype verdict:** the single fixed 2-D panel still tangles and its
labels collide (`PROTO_A2_single_panel.png`). The small multiples are
legible (`PROTO_B2_small_multiples.png`) and are the composition to take
forward, pending the maintainer's approval. No GIF was rendered.

## Review round 10 (2026-08-19) — RESUME HERE

The maintainer's round 10 found no code defects. Both High findings were
about **process integrity**: the plan, the gate, the recorded measurements
and the actual files had drifted apart when the Market rewrite was
discarded after its measurements had already been written down. Both were
verified by measurement and corrected in this session:

| finding | verified as | disposition |
|-|-|-|
| recorded gate state ≠ repository gate state | **confirmed exactly** — 191/145 and 187/150, not the recorded 148 | budget comment corrected; `EXPECTED_VISIBLE_OUTPUTS['market_forecast']` REMOVED (it described a discarded notebook and was failing against `{2, 4, 5, 6}`); reds table re-measured at 42 |
| scratchpad evidence unavailable | **half right, and the half that matters is right** — the assets exist, but at an absolute session-scoped path outside the repo, which no reader can reach | prototypes + generator now TRACKED at `notes/evidence/plan4-market-prototypes/`; the discarded artifact is declared gone rather than implied recoverable |

### What is NOT done — the resume list

The maintainer's sequence, steps 1–3 complete:

1. ~~correct the session record's Market measurements and red-count~~ **done**
2. ~~correct the gate so it describes HEAD~~ **done**
3. ~~recover and track the prototype PNGs + generation script~~ **done**
4. ~~formally revise Plan 4's Market and prediction requirements~~ **done**
   — *Revision note (v5)*, `22413692`. It retires the forecast claim with
   the evidence, lists eight acceptance criteria for the replacement (five
   gate-enforced), voids the 145/150 budget, and records a Task 3 defect
   the weather study surfaced. **The carrier question is NOT closed, and
   the reason is measured:** weather was tested under the same
   preregistered rule (`scripts/weather_forecast_study.py`, `c2a211c7`)
   and does not earn the claim either — `climatology` is the strongest
   baseline in all eight cells, and the one model that beats it does so in
   a single block. It fails for the OPPOSITE reason to Market, which is
   what makes it a decision rather than a dead end: Market's survivors
   were artifacts of anti-correlated baselines, whereas the weather models
   genuinely track what happens next (temperature **r=+0.90**) and lose
   only to a baseline handed the calendar. Recorded as the open decision
   *"Which example carries the prediction story"*, with the implemented
   option being that no example claims skill. **This needs the
   maintainer's call.**

   *Superseded — the original wording of this step:*
   The plan still prescribes a Market forecast story. This note recording
   that the study rejected it is an execution-note deviation, not a revised
   specification; without the revision, Task 2 could be called complete
   while failing the plan that defines completion. The revision must state:
   the forecast claim was rejected by the preregistered study plus the
   post-hoc recovery audit; the replacement's acceptance criteria; WHICH
   example carries the prediction story instead (maintainer's suggestions:
   seasonal weather, motion, sensor data); and any budget change if the new
   composition changes the loader/builder split.
5. ~~review the small multiples at documentation display width~~ **done**
   — and the answer was "reject and iterate", which is why there is now a
   **prototype C** rendered at **736 px** (furo's content column), `e3b0c5cd`
6. approve a static composition BEFORE building any animation — **blocked
   on the maintainer**, and now with a measured question attached (below)
7. implement the new Market script, no scoring claims
8. generate + execute the notebook, then record its REAL output cells in
   `EXPECTED_VISIBLE_OUTPUTS` (it is empty again, deliberately)
9. re-measure budgets and update the reds table ONCE, after the design settles
10. Tasks 3–7, then the publication and full-suite gates

## Review round 11 (2026-08-19)

Round 11 accepted the round-10 corrections ("finding 6: the restored gate
record is now coherent") and raised two Highs, both handled here.

| finding | disposition |
|-|-|
| **2 (High)** weather study pools cities in scale-sensitive raw units | **fixed and rerun.** Scores are now divided by each city's own units before pooling, and reported three ways. `5ac84aed` |
| **1 (High)** B2 not publication-ready: thin band, empty panels, clipped labels, "dark" mean that is not dark | **prototype C**, at 736 px. `e3b0c5cd` |
| **3 (Medium)** v5 says five gates, marks four; "one `hyp.plot` call" contradicts a six-panel figure | criterion 4 now explicitly `(gate)` with its assertion assigned to the examples gate; criterion 3 restated as "no trajectory is ever drawn by hand". `e3b0c5cd` |
| **4 (Medium)** the dark market mean is not produced by hierarchy arithmetic | **confirmed by measurement, and it is worse than "not dark": it cannot be made visible this way at all.** See below. |
| **5 (Low)** retry schedule (15 s) does not match the recorded ~30 s | schedule now spans 62 s, honours `Retry-After` in both spellings, jitters, retries 5xx, refuses to retry a 4xx, and paces cities. `5ac84aed` |

### The scale correction moved numbers, not verdicts

`evaluate` concatenated all six cities in raw units, so precipitation in a
city that swings by 80 mm outvoted one that swings by 2 mm by roughly the
square of the ratio. (`market_representation_study.py` had already banned
exactly this on the MEASURE axis, in its own docstring — the weather study
just failed to apply it on the CITY axis.) Each city's predictions and
realisations are now divided by that city's own month-over-month std, a
constant that never reaches a model and divides both sides of every
competitor identically.

Rerun on the same 6 × 420 real archive: Kalman's block-1 temperature falls
**+0.697 → +0.530** (its per-city correlations span +0.24 to +0.92, which
is what the raw pooling had been hiding), and **no verdict moves**. The
rule was applied under all three aggregations — `pooled_scaled` (headline),
per-city Fisher-z, and the original raw pooling — and all three refuse
everything at t=1. All three sets of numbers are pinned in tests.

**No end-to-end "rescale a city, same verdict" test exists, deliberately.**
The shipped forecaster is not scale-equivariant: at ×100, Kalman's one-step
change moves 41 % on one column and ARIMA's 32 %, and n_iter 1 / 5 / 25 /
100 does not close it (pykalman's EM starts from identity covariances and
settles elsewhere). So the invariance is asserted where it is real — the
scoring layer, plus the exactly-equivariant baselines. **Flagged for the
maintainer:** `hypertools.predict` documents no scaling requirement, and a
forecast that depends on the units you supply is at minimum a docs defect.

### The market mean cannot be a visible reference in this composition

Not a tuning problem. The hierarchy parent is the mean of its children, so
with one focused leaf and five pale ones it always lands *inside* the peer
greys. Measured parent-vs-peer luminance gap at three palette settings:
**0.094 / 0.088 / 0.062**, and the hierarchy's automatic doubled linewidth
(measured 2.0 vs 1.0) does not recover a 0.09 gap at 736 px. C therefore
stops claiming it: the title no longer says "against the market mean", and
the mean is findable by its neutral endpoint marker, not by its path.

So step 6 now carries a real question: **peers as context with no visible
mean (C as it stands), or the mean as a reference with peers dropped or
darkened** — which needs a dark component shared by every leaf and muddies
the leaf colours. Arithmetic cannot settle it.

### What C fixed, measured

`hyp.plot` centres each dimension separately but applies ONE gain to all of
them (probe: x ∈ [0, 100], y ∈ [0, 1] → drawn y ∈ [-0.01, 0.01]). Return
and drawdown have half-ranges 0.456 and 0.129 over the complete frame, so
drawdown was drawn into a third of its due height. Each measure now gets
one display gain computed once over the complete frame; drawn spans are
0.33–1.18 (x) and 0.46–1.45 (y) out of the 2.0 available, all six panels
still share one limit tuple exactly.

## Review round 12 (2026-08-19)

| finding | disposition |
|-|-|
| **1 (High)** C hides the feature the example showcases | **prototype D**, and criterion 3 relaxed on a proof rather than a preference |
| **2 (Medium)** the headline scale used future evaluation data | **fixed**: units are calibrated on the 24 changes before the first anchor and reused for both blocks; `calibration_end` is a REQUIRED argument and `evaluate` raises if it reaches past the first anchor |
| **3 (Medium)** the retry tests need to bind a socket | socket tests marked and skipped where binding is denied (asked by TRYING a bind); 6 socket-free tests added through a new `opener=` seam |
| **4 (Medium)** scale-sensitive forecasting should be a documented limitation | **Notes** section on `hyp.predict`'s docstring + `notes/followup_predict_scale_sensitivity.md`; GitHub issue NOT filed (needs the go-ahead) |
| **5, 6 (Low)** prediction decision and `hue_mode` coverage adequate | no action |

### The mean-colour constraint is a theorem, not a tuning problem

Round 11 asked whether a dark parent could be tuned into existence. It
cannot:

> With `hue_mode='mixture'` the parent's colour is the MEAN of its
> children's colours; a mean lies in the convex hull of what it averages;
> **so the parent can never be darker than its darkest leaf.**

Five pale peers therefore force a pale mean at every palette setting, which
is why three settings all measured a ~0.09 gap. Criterion 3 now requires
the hierarchy to be native in **discovery, geometry and style** but not in
**colour** — and D's mean goes from a 0.088 gap to **0.76**.

D draws the same complete frame twice (matrix hue for the leaves, a dark
single-colour palette for the parent) and hides the second call's leaves.
The two calls agree on the parent path to **0.00e+00**, so the overlay is
exact rather than approximate. It works, and a shipped example needing two
calls plus a visibility toggle is an argument for closing the API gap:
**there is no spelling of `hyp.plot` that gives per-leaf colours AND an
independently coloured parent** — measured five ways, table in
`notes/evidence/plan4-market-prototypes/README.md`. Library work Plan 4
does not own.

### The calibration fix changed a published claim

Numbers moved, no verdict moved — but "climatology is the strongest
baseline in **all eight** cells" is no longer true under the headline
aggregation: `seasonal_naive` takes windspeed/block 2 at +0.296, so it is
**seven of eight**. ARIMA clears that cell by **0.0001** (a rounding
artefact) and loses the same measure in block 1 by 0.57, so the both-blocks
clause refuses it anyway. The study now names the winning baseline PER
AGGREGATION rather than once per cell, and prints the winning margin, so a
near-tie cannot read as a clean pass. Plan, notes and the pinned tests are
all updated to seven-of-eight.

Still open from round 9's sequence: harden `scripts/execute_tutorial.py`
(argparse, create the output directory, reject output collisions) — it is
currently DELETED from the tree and lives only in the session scratchpad,
so it must be rewritten, not restored.

### Checks that were re-run after these edits

`tests/test_examples_are_native.py` (42 red, unchanged and every red
accounted for above), the sdist tracked-files guard (5 passed — the newly
tracked evidence files are the thing that guard exists to catch), and
`ruff check` on both touched files, clean.

## The reds: **42** (re-measured 2026-08-19)

`tests/test_examples_are_native.py` collects **139**: **92 pass, 42 fail,
5 skip**.

**This supersedes a "39 → 31" table that was wrong in both directions.**
Those 31 were counted against the Market rewrite, which was then discarded;
the tree went back to the un-rewritten example and notebook, and the count
went back UP. It landed at 42 rather than the original 39 because two of
the gate's own records had been updated to describe the rewrite and were
not reverted with it — the budget comment (145 "measured", then 148) and
`EXPECTED_VISIBLE_OUTPUTS['market_forecast'] = {3, 5, 6, 7, 8}`. The second
is the one that mattered: the committed notebook emits on `{2, 4, 5, 6}`,
so that test failed by comparing HEAD against a file that no longer
existed. Both are corrected; the market entry is REMOVED rather than
re-measured, since the committed notebook is itself due for replacement and
re-measuring would only pin a second doomed artifact.

| test | n | note |
|-|-|-|
| `test_no_defect_marker_in_the_launch_examples` | 18 | Tasks 2–7; market contributes 6 again now that the un-rewritten file is back |
| `test_file_is_within_its_size_budget` | 7 | market **191/145**, weather 195/75, paintings 146/140, conversation 165/110, market.ipynb **187/150**, weather.ipynb 194/80, conversation_shape.ipynb 176/115 |
| `test_the_right_cells_carry_visible_output` | 5 | `EXPECTED_VISIBLE_OUTPUTS` is empty again — all five say "no measured index set recorded" |
| `test_examples_produce_their_stated_artifact` | 5 | Step 0b's split, Tasks 2–6 |
| `test_older_tutorials_dropped_their_hand_rolled_helpers` | 4 | Task 7 |
| `test_analyze_tutorial_actually_plots`, `test_reduce_tutorial_mentions_describe` | 2 | Task 7 |
| `test_every_allowlisted_reach_is_still_present_and_still_explained` | 1 | conversation's `ani._args`; deliberate, Task 4's to-do — NOT to be silenced by re-adding a dead entry |

Every one of the 42 is an un-rewritten example. **Nothing in this table is
Task 2's own any more**: with the rewrite discarded, Task 2 has produced
library commits and evidence, and no artifact. Regenerate with
`.venv/bin/python -m pytest tests/test_examples_are_native.py -q --tb=no -rf`.

## The original 39 deliberate reds (historical — superseded by the table above)

`tests/test_examples_are_native.py` collects **139**: 93 pass, **39 fail**, 7 skip
(2 `PRIVATE_API_EXCEPTIONS` + 5 opt-in smoke). Every failure is an un-rewritten
example, i.e. the to-do list Tasks 1–7 burn down. Recorded here so a NEW failure
is distinguishable from an expected one — a red suite that hides a regression is
worse than no gate:

| test | n | why |
|-|-|-|
| `test_file_is_within_its_size_budget` | 7 | market 191/145, weather 195/75, paintings 146/140, conversation 165/110, market.ipynb 187/150, weather.ipynb 194/80, conversation_shape.ipynb 176/115 |
| `test_no_defect_marker_in_the_launch_examples` | 16 | `ani._func`×5, `SentenceTransformer`×4, `antialias_line`×2, `ani._args`×2, `hypertools._shared`, `from hypertools.plot import morph`, `morph_schedule` |
| `test_examples_produce_their_stated_artifact` | 5 | no `__main__` guard yet — Step 0b's split is Tasks 2–6's work |
| `test_the_right_cells_carry_visible_output` | 5 | `EXPECTED_VISIBLE_OUTPUTS` ships empty by design |
| `test_older_tutorials_dropped_their_hand_rolled_helpers` | 4 | Task 7 |
| `test_analyze_tutorial_actually_plots`, `test_reduce_tutorial_mentions_describe` | 2 | Task 7 |

The exact node-id list is in the commit that added them; regenerate with
`.venv/bin/python -m pytest tests/test_examples_are_native.py -q | grep ^FAILED`.

## Deviations from the plan's prescribed code, and why

1. **`measure_native_ratio.py` used a bare `open(path).read()`** in both
   `_code_lines_py` and `_code_lines_nb`. The gate measures 30 files per run, so
   under `-W error` that is a `ResourceWarning` per file and the budget checks
   become errors; on a non-refcounting interpreter it is a real descriptor leak.
   Both are now `with` blocks. The measurement is unchanged — market still reads
   `code=191 native=11 ratio=5.8%`, reproducing the plan's BEFORE table exactly.
2. **Two unused top-level imports** in the plan's Step 2 block (`importlib.util`,
   `json`) — both are re-imported inside the function bodies that use them, so
   kept verbatim they are new ruff `F401`s. Removed; no body touched.
3. **Every `plot.py`/`colors.py` line number in Task 1 was stale**, as the plan
   itself predicts (it is sequenced after Plans 1–3). Located by symbol instead.
   The plan's *content* claims all held, including "five raw-seaborn call sites,
   one of them `sns.set_palette`".

## Verified, not assumed

- `HyperAnimation` accessors: 9 tests, and **both** morph tag sites are
  mutation-verified — deleting the 2-D tag alone fails
  `test_n_segments_is_set_for_a_2d_morph_too`, which is exactly the
  silently-`None`-for-half-of-morphs failure the plan warns about.
- Task 1's largest-cluster regression really reproduces: restoring the old
  `score = frac` ordering fails 5 tests, and `pal[0]` comes back as the beige
  background `(0.784, 0.769, 0.737)` instead of the vivid `(0.863, 0.078, 0.078)`.
- Live market data is reachable for Task 2: **2514 sessions, 2016-08-15 →
  2026-08-14**, `adjclose` present — matching the plan's 2026-08-15 measurement
  (the `range=10y` window slides, so this count is run-date dependent by design).

## Full-suite state at the checkpoint

Before the gate landed: **3589 passed, 13 skipped, 2 deselected, 1 failed** — the
one failure was `test_sdist_contains_only_tracked_files_plus_allowlist`, caused by
the new test files being untracked at collection time. It passes once they are
`git add`ed, which is the guard working as designed.


## Review round 13 (2026-08-20)

| finding | disposition |
|-|-|
| **1 (High)** D is static-only under the current architecture | **premise confirmed, conclusion reversed**: panels moved into the DATA, so six panels are one call and one animation (prototype **E**) |
| **2 (High)** the smoke gate still requires the retired forecast | **fixed**: `predicts=True` removed, `min_frames` 100 -> 60; the gate revision is now an explicit Task 2 checklist item |
| **3 (Medium)** D's mean dominates | D2 followed the prescription and the knot survived; **diagnosed** (roughness) and fixed by fewer, wider strides -> **D3** |
| **4 (Medium)** don't teach the two-call workaround | quarantined in `draw_hierarchy_mean` with a docstring saying why it exists; plan says Task 2 does NOT add the parent-colour API |
| **5, 6, 7 (Low)** weather calibration, retry portability, predict note | closed, no action |

### Three measurements that decided the animation question

1. **`hyp.plot` ignores `ax=` when `animate=` is set** -- it builds its own
   figure and leaves the passed axes empty. That, not the schedule count, is
   why six panels could not be one animation; it also makes the two-call
   dark-mean workaround impossible in any animated plot.
2. **`draw_frame(i)` is public** and documented as the supported way to drive
   an animation, so N animations CAN be stepped in lockstep -- but they live
   on N figures, so there is nothing to save them into.
3. **Panels do not have to be axes.** Translating each panel group into its
   own region of one shared box makes six panels six column groups of one
   frame: one call, one animation, an ordinary `.save()`. Measured on real
   data: 90 frames, 0.87 MB, 6 colours, `[1, 1, 1, 1, 2]` linewidths per
   panel. Panels laid out in the data share one normalization BY
   CONSTRUCTION, which is criterion 4 satisfied structurally.

One backend behaviour a multi-panel animated example must handle: it
**draws its frame box as an axes patch**, not as spines, so cropping the
view leaves its left and right edges crossing the figure as two vertical
rules. `frame_kwargs={'visible': False}` is the documented knob.

**CORRECTED in round 13 (2026-08-20):** this note also claimed the backend
"re-applies its axis styling every frame". FALSE -- inferred from those two
rules surviving a `save()`, which the frame box explains on its own.
Measured: limits, spines and patch visibility all survive `draw_frame` AND
`save`. `on_frame` is for decoration that must CHANGE per frame (the head
dots), not for styling that has to persist. The claim had reached the plan,
this note, the evidence README, the prototype's comments, the library
docstring and a commit message; all six are corrected.

### The knot was in the data

`roughness = total turning / total drawn length`: 8.59 for the market mean
against 2.94 for a sector, unchanged by the shorter window, and made WORSE
by smoothing (10.10) because smoothing removes length faster than turning.
The market mean covers 0.46x the ground of one sector while turning as
often -- averaging cancels direction, not noise. Six-month strides over
five years: 4.18, and more history rather than less.

### Open: which composition

**Market stays animated** (maintainer's standing preference; the evidence no
longer argues against it). The open pick is D3 vs E -- they differ in what
the hierarchy MEANS, not in how it looks. Table in the plan's *Animated or
static* section. E needs no workaround at all and animates in one call; D3
keeps the market mean as the shared reference and cannot animate without
new API.


## Review round 13 (2026-08-20) -- E SELECTED

| finding | disposition |
|-|-|
| **1 (High)** Plan v5 contradicts E's architecture | criteria 3 and 4 rewritten as **two alternative contracts** -- multi-axes (a) and tiled-one-call (b); a worker following the old text would have rejected E |
| **2 (High)** E changes what the hierarchy MEANS | recorded formally: Market now shows **sector -> ticker**, and a top-level `Market` parent **must not** be added back (offsets are applied before the means, so a global parent would average layout translations) |
| **3 (Medium)** "comparable" overclaims | criterion 4 now separates same-units/same-gain (yes), same local box (yes), cross-panel distance (**no**); the figure carries the caption saying so |
| **4 (Medium)** layout plumbing dominates | dead colour-keyed cells removed; decoration in one `decorate_panels`; identity by the library's **labels + order**, proven by the mean identity, not by colour; the affine probe kept but re-explained as annotation placement |
| **5 (Medium)** `animate=` silently ignoring `ax=` | **fixed in the library**: `ValueError`, 13 tests, docstring |
| **6 (Medium)** the gate is necessary but insufficient | `_assert_tiled_composition` added with 8 checks, plus `tests/test_tiled_composition_gate.py` -- 7 tests that exercise it against a conforming composition and 5 real mutations |
| **7 (Low)** visual polish | darker/heavier parents (2.6 vs 0.9), head dots via `on_frame`, caption, mid-frame kept as evidence |

### A claim of mine that was WRONG, and is now corrected everywhere

Round 13's evidence commit asserted that the animated backend **re-applies
its axis styling on every frame**. It does not. That was an inference from
two vertical rules surviving a `save()` -- and the library's own frame box
(an axes PATCH, exposed publicly as `frame_kwargs`) explains those by
itself. Measured directly: axis limits, spine visibility and patch
visibility all survive `draw_frame` AND `save`.

The claim had reached six places: the plan, this note, the evidence README,
the prototype's comments, the new `hyp.plot` docstring, and a commit
message. All are corrected. `on_frame` is for decoration that must CHANGE
per frame -- the head dots -- not for styling that merely has to persist.

The gate keeps a view-stability assertion, but now as a REGRESSION check
with an honest comment, and its self-test mutates an `on_frame` callback
rather than pretending the backend resets anything.

### What the tiled gate can and cannot prove

It cannot detect a per-panel RESCALE of the source data: a panel scaled up
by 3 renders identically to a panel whose data legitimately spans 3x as
much, and no property of the figure separates them. Pooled per-measure
scaling is therefore held BY CONSTRUCTION (one call, one frame, one divisor
computed once) and criterion 4 records that. The gate's box-size check is a
LAYOUT claim -- panels get the same room -- and is worded as one.

### The animated hierarchy contract, and the trap in checking it

`hyp.plot` resamples each trace along its OWN arc length, and the sample
count depends on the frame (measured: 980 points at the last frame of a
90-frame reveal over 12 rows; a static draw of the same data gives 1101 =
11 x 100 + 1). So the mean of several drawn paths is NOT the drawn mean
between vertices -- whole-path comparison shows a ~2e-3 residual that means
nothing, and on an animated figure the vertices are not at predictable
indices either.

What IS exact on a fully revealed trace is its ENDPOINTS. The gate asserts
the identity there (`< 1e-9`) and then requires each parent to be at least
20x closer to its own leaves than to any other panel's. On a static draw
the vertex check is exact: **4.44e-16**, against **1.62e+00** for a
mis-attributed control.


## Task 2 LANDED (2026-09-03) -- composition E, and what landing it measured

Resumed after a two-week gap from the "E SELECTED" state above. The gate
went **42 -> 30** reds; every market entry is green, and nothing else moved.

| unit | state |
|-|-|
| `scripts/execute_tutorial.py` | created from the plan's Step 1 text, smoke-tested with `--out-dir` (tree clean), then **amended** -- see the trap below |
| `examples/animate_market_forecast.py` | rewritten as E with the loader / fixture / builder split; payload `Market(stocks, source)` |
| `docs/tutorials/market_forecast.ipynb` | regenerated from the script (code cells = the script's sections), executed for real, `EXPECTED_VISIBLE_OUTPUTS = {7, 8}` |
| `docs/tutorials/market_forecast.gif` | 90 frames, 0.86 MB (was 2.6 MB), live Yahoo data Feb 2022 - Aug 2026 |
| `tests/test_examples_are_native.py` | market budget re-measured **181 -> 185**; visible set recorded |
| `docs/tutorials.rst` | section title no longer says forecast |
| plan | Task 2 checklist ticked; budget note; Task 2 header marks Steps 2-6 as superseded history |

### Three things measured on the way that the plan did not know

1. **The default trail window hides most of the path at the last frame.**
   `tail_duration` defaults to 2 s; at 6 s the last frame held only
   **16-54%** of each path (arc length against a static draw), so the start
   of a drawn trace was mid-path and the gate's exact endpoint identity
   failed by 6e-4. The gate's own self-test uses a 2 s clip, where the
   window happens to cover everything. The example passes
   `tail_duration=DURATION`; the last frame is then fully revealed
   (0.998-1.000) and the identity holds to 2e-16. Recorded in the script's
   comment, since the number is the reason for the argument.
2. **No closed-form affine reproduces the drawn coordinates.** Midpoint
   centring matched x to 1e-16 and missed y by 0.149; mean centring missed
   by 0.07. The static probe the prototype used is exact and stays, and is
   explained as annotation placement (round 13, finding 4).
3. **The Colab install cell rewrites the venv.** Executing a launch notebook
   locally runs `%pip install "hypertools[...] @ git+...@dev-1.0"`, which
   installs the REMOTE branch over the editable checkout mid-run; the
   market notebook then failed in its own kernel with "48 dimensions ...
   static plots support at most 2" (column-MultiIndex support gone), and
   `pip show hypertools` reported the git install afterwards. Every
   committed launch notebook carries 2026-07-30 execution timestamps on
   that cell -- in July local and remote agreed, so nothing noticed. The
   runner now tags `pip install` cells `skip-execution` in memory and
   strips the tag before writing, so the committed cell is byte-identical
   (verified against HEAD). Also found on the way: the venv held a
   NON-editable hypertools 1.0.0, so `python examples/<file>.py` from any
   directory but the repo root ran 1.0.0. Reinstalled with
   `pip install -e ".[dev]"`; **check `pip show hypertools` says Editable
   after any notebook execution.** A project skillnote records this.

### Decisions taken here, flagged for the maintainer

- **File names kept.** `animate_market_forecast.py` / `market_forecast.ipynb`
  are published 1.0 URLs. No displayed prose says "forecast"; a comment at
  the top of the script says why the name stays. Renaming costs the five
  touch points the plan lists.
- **Payload shape.** `Market(stocks, source)` rather than the plan's
  `Market(regimes, closes, source)`: those fields belonged to the discarded
  rewrite, and the tiling is presentation, so only the
  `(Sector, Ticker, Measure)` frame crosses the loader/builder boundary.
- **Strokes counted back from the latest month**, and a month in progress
  is dropped, so the span ends at the most recent complete month rather
  than five months early (the first cut decimated from the window's start).
- **The notebook shows no inline video.** `draw_frame` returns the wrapper,
  whose repr embeds an 89 KB autoplay mp4; assigned away, so the page
  carries the GIF once, via the markdown cell, like the other four.

### Remaining reds (30), which are Tasks 3-7

weather 5 (2 budgets, `ani._func` in the notebook, artifact, visible set),
paintings 5, conversation 8, morph 6, older tutorials 4, analyze 1,
reduce 1 -- exactly the per-task lists at the top of this note, minus
market. `test_every_allowlisted_reach_is_still_present_and_still_explained`
fails on conversation's reach (Task 5), as its comment already says.

**Resume at Task 3 (Weather).** The pattern that worked for market: rewrite
the script with the split written in, generate the notebook from the
script's sections (generator in the session scratchpad, not the repo),
execute with `scripts/execute_tutorial.py`, record the visible set, run
the gate, re-measure the budget once, commit script + notebook + GIF +
gate together.

### Maintainer calls, 2026-09-03 (after the Task 2 commit)

| question | call | done |
|-|-|-|
| untracked AGENTS.md hierarchy, `.omo/`, the 2026-08-21 note | **ignore** the agent files (gitignore + a MANIFEST.in `global-exclude`, because `graft tests` walks the filesystem and would still ship `tests/AGENTS.md`); commit the note | yes |
| Market file names | **rename to `*_sectors`** -- `examples/animate_market_sectors.py`, `docs/tutorials/market_sectors.{ipynb,gif}`; the two 1.0 URLs break (no redirect mechanism in `docs/conf.py`) | yes; notebook re-executed under the new name, visible set still `{7, 8}` |
| Tasks 3-7 | **proceed straight through**, commit per task, report at the end | in progress |

Historical mentions of `animate_market_forecast` / `market_forecast.ipynb` in
the plan, this note and the gate's comments describe the files as they were
and are left as written.

## Task 3 LANDED (2026-09-03) -- Weather, the paper figure

Gate **30 -> 26**; no weather entry red. The plan's prescribed rewrite was
applied nearly verbatim with the split written in (`Weather(temps, cities,
source)`); the fetcher names the exception it swallows (the defect note's
rule), writes its cache atomically, and `load_weather` catches the offline
refusal, which is +6 over the plan's 73 -> budget **80** (77 measured).
Notebook 81 (≤ 85), visible set `{4, 5}`, GIF 160 frames / 5.2 MB (same
size as before). `docs/tutorials.rst` section retitled and its synopsis
rewritten: the old one explained the list-vs-MultiIndex hemisphere design
the rewrite deletes. Data verified live: 1645 complete months, 20 cities.

## Task 4 LANDED (2026-09-03) -- Paintings

Gate **26 -> 21**; no paintings entry red. Script 147 (budget 150), notebook
151 (≤ 155), visible set `{4, 5}`, GIF 180 frames / 5.5 MB at 75 dpi,
fixture `examples/data/painting_palette_fixture.png` (7.3 KB, 64 px).

Measured corrections to the plan, all recorded in its Task 4 banner: the
payload carries `vectorizer` so the fixture embeds with TF-IDF (no model
download in the default suite); a luminance floor (≤ 0.6) on the canvas
colour because The Great Wave's two most salient clusters are cream and
its whole panel vanished on white; 15 fps because five paragraphs of
antialiased text make a 240-frame GIF 7-10 MB.

Two defects fixed on the way:

* **`HyperAnimation.save()` dropped every keyword but `fps=`.** `dpi=75`
  produced a byte-identical 10 MB GIF; the market notebook's `dpi=100` had
  been doing nothing too. Now forwarded to the raster/video writers, and
  any other keyword raises `TypeError`. Three tests in
  `tests/test_animation_export.py` (real GIFs at two dpis, the default,
  the refusal); CHANGELOG entry under 1.1.0 bug fixes.
* **Progress-bar widgets in committed notebooks.** transformers' "Loading
  weights" tqdm arrived as a `widget-view` display_data with no saved
  state (the docs would render a stuck 0% line). The runner sets
  `HF_HUB_DISABLE_PROGRESS_BARS=1`, which the kernel inherits.

Also: a background agent cleared the 111 pre-existing ruff findings in
`tests/` and `scripts/` on a worktree branch (commit `7442ac7e`, based on
`dcd72d29`); cherry-picked after this task's commit.

## Task 5 LANDED (2026-09-03) -- Conversation

Gate **21 -> 12** (every conversation entry green, and the allowlisted-reach
test with them). Script 113 (budget 115), notebook 118 (≤ 120), visible
set `{6, 7}`, GIF 192 frames / 5.9 MB. `tests/plot/test_recency_fade.py`:
20 tests, all driving the real callback.

Two corrections, both in the plan's Task 5 banner: the payload carries
`texts` so the titles do not read a module global; and trails take their
head's alpha rather than 0.3x -- on a serial reveal the trail is the
spoken part of the current turn (821 points vs a 6-point head), so the
plan's convention made the current turn the faintest line on screen. The
first render looked washed out for exactly that reason; measured, fixed,
re-rendered.

The plan's maintainer call on `vectorizer` in the payload: taken as the
plan's table assumed (fixture = TF-IDF, no model download in the suite).

## Task 6 LANDED (2026-09-03) -- Morph

Gate **12 -> 6** (what is left is Task 7: four older tutorials, analyze,
reduce). Script 61 (budget 65: Contract 4's offline fallback for
`hyp.load` costs five parametric clouds), notebook 65 (≤ 70), visible set
`{5, 6}`, GIF 240 frames / 6.2 MB. The plan's Step 3 title check omits
`rotations` and so reports 34 false boundary mismatches; with the weighted
schedule the titles match on all 240 frames. Recorded in the plan.

## Task 7 LANDED (2026-09-03) -- the older tutorials

Gate **6 -> 0**: `tests/test_examples_are_native.py` is fully green for
the first time. Eight notebooks edited by one content-matching script,
executed, ratios all above baseline (table in the plan's Task 7 banner).
Measured surprises, each in the banner: direct GIFs at 900 frames are
13 MB (three animations cut to 300 frames, one to 15 fps); `datasets`
returns a `Column` that `hyp.plot` refuses (`list(...)`); an import-order
slip in projectile; the double-figure repr under the inline backend.

The second ruff agent's commit (`bd5df563`, based on `ebed10c2`; full
suite 3767 passed on its branch) is cherry-picked after this commit. It
noqa'd E402 in six tutorial notebooks' code cells, three of which this
task re-executed, so the cherry-pick may conflict on those; resolution is
to keep the executed notebook and re-add the noqa comments.

## Task 8 LANDED (2026-09-03) -- verification, and the gallery scraper

Gate: 139 collected, 134 passed, 0 failed, 5 skipped (opt-in smoke). Docs
`-W -E -a` from a wiped `_build`: 0 warnings. Five thumbnails generated
and referenced; `tests/test_docs_thumbnails.py` green.

**The plan's Step 6 could never have worked as written.** sphinx-gallery
only scrapes animations whose figure is in `plt.get_fignums()`, and
`show=False` leaves the figure unmanaged -- so the five launch examples
had never rendered anything in the gallery, before or after Plan 4. Fixed
in `docs/conf.py` with `hyperanimation_scraper` (renders any
`HyperAnimation` through sphinx-gallery's own `_anim_rst`); examples
untouched. Remember: a scraper change does not invalidate the gallery's
md5 cache -- delete `docs/auto_examples/<stem>.py.md5` to re-run one.

Also found: Sphinx keeps orphaned output from renamed pages across `-E -a`
(wipe `_build`); sphinx-gallery executes without `__file__` (paintings
guards it); the hierarchy-guide control forbids the phrase "one moving
path" anywhere in `docs/tutorials.rst`, including alt text.

`RELEASE_CHECKLIST.md` rewritten for 1.1.0: push + integration PR FIRST
(no hosted CI has seen this line), local validation list, v1.1.0
everywhere, conda-forge is now a bot bump on the existing feedstock.

**Plan 4 is complete.** What remains is not example work: push `dev-1.0`,
open the PR, wait for matrix CI, then follow the checklist.

## RESUME HERE (paused 2026-09-03 ~11:10, computer suspended)

State: `dev-1.0` at `f8d75655` (+ this note commit), tree clean, 184+
commits ahead of `origin/dev-1.0`, nothing pushed. Plan 4 complete.

The one thing not finished: the FINAL full-suite run on the committed tree
was started and then killed by the suspend. An earlier full run on a
slightly older tree (before the scraper/thumbnail/checklist commit) was
also interrupted. The last COMPLETE full-suite figures are the ruff agent's
on its branch (3767 passed, 13 skipped) and this session's pre-rename run
(3744 passed, 13 skipped, gate deselected). So:

    .venv/bin/python -m pytest -q -p no:cacheprovider 2>&1 | tail -3

must be run once, to completion, with nothing else running. Expected:
0 failed, ~3790 passed (139 gate tests are now included), 5 + 13 skipped.

Then the release line, in `RELEASE_CHECKLIST.md` step 0 order: push
`dev-1.0`, open the integration PR to `master`, wait for matrix CI.
Maintainer calls still open: delete `dev-1.0` after release (checklist
step 8); nothing else.

## Final full-suite run DONE (2026-09-03 ~11:50) -- and the push

The interrupted run had in fact survived the suspend (orphaned at 69%, 0
failures, output unreadable); stopped it and re-ran from scratch:

    3901 passed, 18 skipped, 2 deselected (bigdata), 0 failed, 12m34s

Only warnings: umap-learn's own "n_jobs value 1 overridden to 1 by setting
random_state" (3x, from any seeded UMAP call; emitted by umap regardless of
n_jobs -- third-party, benign, left alone). `ruff check .` clean; venv
editable 1.1.0. Pushed `dev-1.0` and opened the integration PR to `master`
(checklist step 0); matrix CI is the next gate.

**PR #283 opened** (2026-09-03 ~12:05): https://github.com/ContextLab/hypertools/pull/283,
`dev-1.0` -> `master`, head `b028d0b9`. Found while drafting it: the
`ax=`+`animate=` refusal (6c73421c) had a test but no CHANGELOG entry;
added under 1.1.0 *Changed / validation* (the four CHANGELOG-reading test
files pass, 66/0). Next gate: matrix CI on the PR (first hosted run of
this line; expect Linux/Windows findings, fix on `dev-1.0`, PR updates).

## PR #283 first matrix run (2026-09-03): three platform findings, all fixed

Local macOS/py3.12/pandas 3 had never exercised these. Each reproduced
locally where a venv could (scratchpad venvs: py3.13; py3.11 + pandas<3).

| lane(s) | failure | root cause | fix |
|-|-|-|-|
| ubuntu+windows 3.13 | `test_plot_docstring_type_lines_have_no_stray_optional_default_markers` | Python 3.13 dedents docstrings at compile time (gh-81283); the scan picked "exactly 4-space" lines = params on 3.12, DESCRIPTIONS on 3.13, so prose with a colon parsed as a param | `inspect.cleandoc` + column-0 match (`0c0333ca`); 3.13: 1 failed -> 3 passed |
| ubuntu+windows 3.10 (pandas 2.3.3) | `test_colorbar_shows_one_segment_per_top_level_group` (warnings as errors) | pandas 2.x FutureWarning for `groupby(level=[one_level])` at both `hierarchy.py` sites; pandas 3 silent | scalar level when one grouping level, keys unchanged (`c8f3cc07`); pandas 2: 2 failed -> 162 passed; regression test |
| windows 3.10-3.13 | `test_render_script_exits_NO_BROWSER_when_CHROME_CANNOT_BE_FOUND` exit 0 | choreographer's download dir on Windows is `%LOCALAPPDATA%` via platformdirs ctypes/shell API: no env var moves it, so the pre-fetched Chrome is found despite HOME+BROWSER_PATH; the script's own override bypasses the plotly wrapping the test proves | skip on Windows ONLY when that download exists, naming the path (`87c2e3d4`) |

Green lanes: ubuntu 3.11 (pandas 3, no 3.13 dedent), wheel-smoke,
docs-clean, dataset-gate, live-source-gate; release-gate skipping (not
master). Windows lanes take ~28 min; ubuntu ~21.
