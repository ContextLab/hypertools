# Plan 3 Task 4 — session notes, 2026-08-02

Branch `dev-1.0`. Task 4 split into two commits per maintainer review ("schedule/bounding-box
integration, then live drawing — they have different failure modes and can be reviewed
independently").

## State

| step | commit | state |
|-|-|-|
| Task 4 tests (18) committed red | `fcaf13c8` | 17 failed / 10 passed |
| Task 4a — Steps 3–4 (snapshot, schedule, box) | `9afe7831` | still 17F/10P, by design |
| Task 4b — Step 5 (live artists + internal updater) | working tree | **27 passed** |
| Step 8 docstring + CHANGELOG | working tree | pending full suite |

27 passed is exactly the number Task 4 Step 6 predicts (9 from Task 3 + 18 here).

Line numbers in Plan 3's Task 4 were re-derived before use, per the rule added to Plan 4:
Steps 3/4/5 cite `plot.py:3391-3402`, `:4552`, `:4555`, `:4858-4898`; the real sites at
implementation time were `:3404`, `:4565`, `:4581`, `:4912`. Plan 3's own Tasks 0–3 moved them.

---

# TWO DEFECTS FOUND WHILE VERIFYING — neither is Task 4's fault, both are real

## 1. `hyp.predict(model='Kalman')` is numerically unstable (dispatched for root-cause fix)

Found by asking whether Step 4's fold-in is a NO-OP — i.e. whether including
`schedule.stacked_paths()` in the joint stack actually changes the box. It changes it
catastrophically:

```
data      min/max:  -26.086   26.107
schedule  min/max: -3.5e18    2.0e18     (520 rows, 40 fits)
```

Reproduced with **no plot code involved at all**:

```
.venv/bin/python -c "
import numpy as np, hypertools as hyp, warnings
warnings.simplefilter('ignore')
hits=0; trials=0; worst=0
for seed in range(12):
    rng=np.random.default_rng(seed)
    x=np.cumsum(rng.standard_normal((40,3))+0.5,0)
    rng_range=float(x.max()-x.min())
    for k in range(5,41):
        fc=np.asarray(hyp.predict(x[:k],model='Kalman',t=12),dtype=float)
        r=float(np.abs(fc).max())/rng_range
        trials+=1
        if r>100: hits+=1
        worst=max(worst,r)
print(f'{hits}/{trials} fits exceed 100x the data range; worst = {worst:.1f}x')
"
-> 19/432 fits exceed 100x the data range; worst = 10309425.2x
```

**Not monotonic in history length** — for `default_rng(3)`, k=20 and k=22 explode (3.4e3, 4.2e4)
while k=15, k=30 and k=40 are all fine (30.6, 9.7, 27.6). That non-monotonicity points at the
estimation rather than at "short history = wide uncertainty".

Why Task 4 meets it and the static path mostly does not: the static path fits ONCE, on the full
history. `ForecastSchedule` fits at EVERY revealed length, so it samples the unstable ones.

Dispatched to a debugger subagent: root cause in `hypertools/predict/kalman.py`, per-model
comparison table, TDD fix, full suite. Report → `notes/audit/kalman_instability_2026-08-02.md`.

**RESOLVED 2026-08-02, by the root-cause fix.** After the stability constraint landed
(`notes/audit/kalman_instability_2026-08-02.md`), the same measurement gives:

```
data     min/max:  -26.086   26.107
schedule min/max:  -27.059   27.445      (was -3.5e18 / 2.0e18)
box widened by 0.97 below / 1.34 above -- a forecast-sized amount
```

Independently re-run end to end: **0/432 fits exceed 100x the data range, worst 1.17x** (was
19/432, worst 1.03e7x). So Contract 2's "the box contains every forecast, nothing is clamped" is
sound exactly as written, and **`min_history` needs no change**. The fold-in was never the
problem; the forecaster was. Recording the superseded question below, because the reasoning is
still the right reasoning if a future forecaster misbehaves:

~~Open design question (for the maintainer):~~
Contract 2 says the box contains every forecast so nothing is clamped. But the schedule fits from
as little as `DEFAULT_MIN_HISTORY = 2` observations, so even with a perfectly stable forecaster
the frame is sized by the LEAST-informed forecast the animation ever draws. Options are (a) keep
containing everything, (b) clamp — which contradicts the stated contract, or (c) raise
`min_history` so under-determined fits are simply not drawn (`forecast_from_history` already
returns `None`, and the updater already hides the artist). I lean (c) plus the root-cause fix, but
the value of `min_history` should be measured, not guessed, once the Kalman fix lands.

## 2. A user-facing warning names an internal matplotlib sentinel

```
UserWarning: hue category '_nolegend_' has only one observation; a pure line format
cannot render a single point, so it will be invisible -- pass fmt='.' or fmt='o-' ...
```

Emitted at `plot.py:4126`, message built from `hue_group_labels[i]`. At `plot.py:4053`
`hue_group_labels` substitutes the literal `'_nolegend_'` for a `None` category — a matplotlib
sentinel meaning "keep this artist out of the legend". Leaking it into prose addressed to a user
is meaningless: there is no category called `_nolegend_`.

Verified **pre-existing**: reproduced with all of Task 4 reverted out of `plot.py`. The warning
itself is CORRECT (`hue=['a','b']*30` with `fmt='-'` really does produce 60 singleton runs) — only
the name in it is wrong. Fix separately, in its own commit; the message should say something like
"an unnamed hue category" for the `None` case.

---

## Note to self

I used `git stash push <path>` to test whether the warning predated Task 4 — the exact recipe
Plan 4 documents as a data-loss hazard and that I removed from Plan 4 two commits earlier.
Nothing was lost (verified: stash list empty, both hunks present), but the correct command was
`git show <ref>:<path>`. Writing a rule down is not the same as following it.


---

# Tasks 5 and 6 (same day)

## Task 5 — `forecast_trail=` (`90a63a1a`)

11 passed, exactly the plan's predicted count. Full suite **2846 passed, 13 skipped, 0 failed**
(2835 + 11). Docs build clean at `-W -E -a`. Fan visually verified: at frame 20 a cap of 8 is
saturated, alphas 0.6 → 0.138 monotonically, all inside the cube.

**Three of the plan's own prescribed tests could not fail without the feature.** Same class as
Task 4's two vacuous tests, found by the same check (does this assertion discriminate?):

| test | why it passed without the feature | fix |
|-|-|-|
| `test_trail_is_capped_by_an_integer` | asserted only `<= 5` / `<= 4`; a plot with NO trail satisfies both | assert the cap is SATURATED (`== 4`, `== 5`) |
| `test_the_fan_is_a_pure_function_of_the_frame_index` | compared two lists through `zip()`, which iterates zero times when both are empty | require `len(sequential) > 1` first |
| `test_trail_alpha_decreases_with_age` | `[] == sorted([])` is True; then `min([])` raises a confusing error rather than naming the real problem | require at least two trail artists |

Red state went from 4 failures to 7. **That difference is the measure of what those three tests
were worth.**

Also fixed a stale docstring block Task 4b missed: the `animate=` entry still said every non-spin
mode raises `NotImplementedError`. I had updated the `predict :` entry and not that one — the same
documented-but-not-done shape I have been auditing Plan 4 for. Grep for the claim, not just the fix.

## Task 6 — Plotly parity (delegated)

**Found by asking what `forecast_trail=` does on plotly** (answer: silently nothing) and then
asking the better question — what does animated `predict=` do on plotly at all:

```
plotly ANIMATED predict= : traces = 3 | frames = 8
dashed traces in f.data  : 1
  frame 0: no dashed trace   frame 4: no dashed trace   frame 7: no dashed trace
```

One dashed trace exists at build time and **no frame contains one**, so the full-history forecast
is on screen from frame 0 and never updates. This is the exact defect Task 3 fixed for matplotlib
("drawing both would put a frozen full-history forecast on screen from frame 0"), still live on
the other backend. It is worse than drawing nothing: it shows the viewer a prediction made from
data they have not been shown.

The plan predicted this ("a plotly animation would show a **frozen** full-history overlay"), which
is a good sign for the plan — but it was written as a parity checklist item, not as a live defect.
It is a live defect.

Test file extracted and strengthened the same way (two more that could not fail: an in-range check
whose loop `continue`d past every empty trace, and a parity check satisfied by both backends
drawing nothing). Red at **17 failed, 1 passed**; delegated with that as the contract.


---

# Plan 3 COMPLETE — Tasks 0-8

| task | commit | verification |
|-|-|-|
| 0 `FrameHooks.add_internal` | (earlier) | internal updaters run before user callbacks |
| 1-2 forecast helper + `ForecastSchedule` | `e1aa1144` | |
| 3 narrow the refusal | `e1aa1144` | morph-only refusal |
| 4a schedule + bounding box | `9afe7831` | fold-in proven not a no-op |
| 4b live per-frame artists | `4a6eb389` | 27 passed; frames rendered and inspected |
| 5 `forecast_trail=` | `90a63a1a` | 11 passed; fan inspected at 3 frames |
| 6 plotly parity | `bb6fcb18` | 18 passed; **bit-identical** to matplotlib |
| 7 `return_model=` contract | `e45e5371` | 31 passed in the task's own file |
| 8 CHANGELOG + docs | (this) | every claim verified against the code |

Suite went 2806 -> **2864+** across the plan, 0 failures at every checkpoint. Docs build clean at
`-W -E -a` after each doc-touching task.

## Verification that went beyond what the tests check

- **Task 6 parity.** The prescribed test compares the two backends only at the FINAL frame, where
  both have revealed the whole history — precisely where a wrong reveal-count mapping would be
  invisible. Comparing at four MID frames across all four animate styles gives
  `max |plotly - matplotlib| = 0.00e+00`, 16 comparisons.
- **Task 7 premise.** Rather than trust "at the final frame the revealed history is the full
  history", measured `anim_window_bounds(total-1, total, n_grid, 0)` for four grid/frame
  combinations: `end == n_grid` every time, and `revealed_raw_counts` returns 45 of 45 rows.
- **Docs claims.** Every factual assertion in the new `docs/animation.rst` section was checked
  against the code: all three artist roles exist, the plotly `meta` keys are really
  `hyp_forecast_role`/`_age`/`_alpha`, and `ctx.artists` really does exclude forecast artists
  (measured: 2 entries for 2 datasets, all with role `None`).

## The pattern worth carrying forward

**Prescribed tests that pass without the feature, in every single task.**

| task | how many | the shape |
|-|-|-|
| 4 | 2 | looped over `_forecasts(ax, role='live')`, empty pre-task, so the body never ran |
| 5 | 3 | upper bound with no lower bound; `zip` of two empty lists; `min([])` |
| 6 | 2 | `continue` past every empty trace; a parity check satisfied by both sides drawing nothing |
| 7 | 1 | asserted two arrays had the same SHAPE to prove no MUTATION |

Eight in total. Every one reports green whether or not the work was done. The check that finds
them is one question — *what state would make this assertion fail?* — and if the answer is "none",
the test is decoration. Strengthening Task 5's three moved its red state from 4 failures to 7;
that gap is the measure.

## Process slips (recorded because the rule was already written down)

1. Used `git stash push <path>` to check whether a warning predated Task 4 — the exact recipe I had
   removed from Plan 4 two commits earlier as a data-loss hazard. Nothing lost; the right command
   is `git show <ref>:<path>`, which I used for the rest of the session.
2. Extracting Task 7's tests, matched a marker string that appears TWICE in the plan and appended
   Task 4's 16 tests a second time. Caught on the test count (22 -> 38, expected 26), reverted with
   `git checkout --`, re-extracted by taking the second occurrence explicitly. **A locator that is
   not unique is not a locator** — the same lesson the citation sweep kept producing.
3. `tests/plot/test_forecast_animation_plotly.py` was staged early and got swept into the notes
   commit `3452355b` rather than the Task 6 commit. Tracked and correct; not worth rewriting
   history over, but the tidy move is to stage at commit time.

## STILL OPEN — Jeremy's calls, from Plan 3's "Decisions still needed"

None of these is blocked on anything; all four are taste/product calls the plan deliberately did
not invent.

1. **Silent forecast drop with `hue=`/`cluster=`.** Regrouping nulls the forecasts with no warning,
   and the animated path inherits that verbatim, so a user asking for `predict=` silently gets
   none. Implemented as-is and pinned by a test. Options: keep silent / warn / raise for the
   animated path only.
2. **Throttling beyond memoization.** Memoizing by revealed-count caps a 900-frame 3-dataset
   60-row animation at <=177 fits (~10 s instead of ~146 s), but a 500-row history is ~440 ms per
   fit, so a long real series is still minutes. Options: memoization only / a `forecast_every=`
   kwarg / automatic striding past a ceiling. The last two need a default value that is a product
   decision.
3. **`min_history`.** A 2-row history gives a degenerate flat stub, drawn for the opening frames.
   Kept at 2, matching what `hyp.predict` itself accepts. Whether a flat stub beats nothing is a
   taste call. (NOTE: this is SEPARATE from the bounding-box question, which the Kalman root-cause
   fix closed.)
4. **Frozen forecasts under `order='serial'`.** A fully-revealed dataset's forecast stops changing
   and sits frozen while later datasets animate. Implemented and tested. Options: freeze / fade /
   hide.


## Task 8 caught a real regression — from the docs I had just written

The final full suite came back **1 failed, 2867 passed**:

```
FAILED tests/test_animation_guide_docs.py::test_animation_guide_does_not_call_the_contract_purity
E   AssertionError: assert 'pure function' not in ...
E     'pure function' is contained here:
E        *f* is a pure function of *f*, so a saved animation
```

The guard's docstring explains itself: *"Callbacks mutate artists by design -- calling the contract
'purity' is the misstatement this plan's v4 removed, and the guide's own example sets a title every
frame."* It is a blanket substring ban on `docs/animation.rst`, and I walked straight into it
writing about the forecast FAN, where the phrase is actually accurate.

Two ways to make it pass, and only one is legitimate:

- narrow the guard to the callback section — **weakening a test to fit my prose**, which the repo
  rules forbid and which would let the real misstatement back in later
- reword the prose — which is also just better writing: "recomputed from the frame index rather
  than accumulated in a buffer, so it depends only on which frame is being drawn" says what
  "pure function of *f*" means, without jargon and without colliding with a term this document
  deliberately reserves

Took the second, and applied the same wording to the two places outside the guard's reach
(`plot.py`'s `forecast_trail` docstring and CHANGELOG) — the phrase was accurate there, but
consistency removes the hazard of a future reader conflating the two uses. Zero occurrences remain
in `hypertools/`, `CHANGELOG.md` or `docs/*.rst`.

**Worth noting what this validates:** a documentation guard that bans a specific phrase, written by
an earlier revision to stop a specific misstatement, caught a new instance of that phrase over a
month later, in prose written by someone (me) who did not know the rule existed. That is a test
earning its keep — and the exact opposite of the eight assertions in this plan that could not fail.
