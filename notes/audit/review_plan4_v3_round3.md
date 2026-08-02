# Adversarial review, round 3 — Plan 4 v3 (`2026-07-28-hypertools-1.1-examples-and-tutorials.md`)

Reviewed at `8a582834` on `dev-1.0`. Every finding below was reproduced by execution;
anything I could not reproduce is labelled `unverified` and is not counted as a finding.

**Gate reproduction confirmed.** Extracted the `# scripts/measure_native_ratio.py` block (plan
L2883–3051) and the `# tests/test_examples_are_native.py` block (plan L3067–4043) verbatim, placed
the gate at `<repo>/tests/_r3_gate.py`, made `scripts.measure_native_ratio` importable via
`PYTHONPATH`, ran, then deleted the file and the scratch worktree (`git status --porcelain` shows
only another session's concurrent Plan-3 edits to `CHANGELOG.md` / `plot.py`, which are not mine).

```
$ PYTHONPATH=$SCR MPLBACKEND=Agg .venv/bin/python -m pytest tests/_r3_gate.py -q -p no:randomly
39 failed, 93 passed, 7 skipped in 0.70s
$ ... --collect-only -q
139 tests collected in 0.05s
```

**Your measurement is confirmed exactly: 139 collected — 39 failed, 93 passed, 7 skipped.**

Counts by severity: **2 FATAL, 5 HIGH, 5 MEDIUM, 10 LOW.**

---

## FATAL

### F1. Every per-task "confirm it renders" verification block is dead code the moment its own Step 0 runs

Tasks 2 (Step 4, L1247–1261), 3 (Step 3, L1500–1512), 4 (Step 3, L1781–1791), 5 (Step 2a L2040 /
Step 3 L2206–2225) and 6 (Step 2, L2379–2398) all do `runpy.run_path('examples/<file>.py')` and then
read **module-level** names: `ns['fig']`, `ns['ani']`, `ns['titles']`, `ns['clouds']`, `ns['names']`,
`ns['colors']`.

Step 0 of each of those same tasks moves every one of those bindings inside `construct_artifact()`
and behind `if __name__ == '__main__':`. `runpy.run_path` does not set `__name__` to `'__main__'`:

```
$ .venv/bin/python -c "
import runpy, tempfile, os
p = os.path.join(tempfile.mkdtemp(), 'x.py')
open(p,'w').write(\"print('__name__ is', __name__)\nif __name__ == '__main__':\n    fig = 1\n\")
ns = runpy.run_path(p); print('keys has fig?', 'fig' in ns)"
__name__ is <run_path>
keys has fig? False
```

So after Step 0 every one of those blocks raises `KeyError`. Step 0's own prescribed `__main__`
body binds only `data` / `anim` / `fig` — never `ani`, `titles`, `clouds`, `names`, `colors` — so
they fail even if `run_name='__main__'` were passed.

**Two of them are already broken today, before any Step 0:**

```
$ .venv/bin/python -c "<ast walk of module-level Assign targets>"
weather_decades        module-level of interest: ['anim', 'fig']
conversation           module-level of interest: ['anim', 'colors', 'fig']
market_forecast        module-level of interest: ['ani', 'anim', 'fig']
morph_zoo              module-level of interest: ['ani', 'clouds', 'fig', 'titles']
painting_embeddings    module-level of interest: ['ani', 'clouds', 'colors', 'fig']
```

`ns['ani']` does not exist in weather or conversation. **The plan already diagnosed this exact
defect** — Task 8, plan L3833: *"`ns['ani']` does not exist in weather or conversation, which bind
`anim` (Contract 8), so 2 of 5 parametrisations failed on day one"* — and fixed it only inside
Task 8's own test, leaving the identical bug in five per-task verification steps. This is the
documented-but-not-done pattern, fourth occurrence.

**Minimum fix:** replace each `runpy` block with `spec_from_file_location` + `exec_module` +
`m.construct_artifact(m.fixture_data())`, reading `anim.figure` / `anim.draw_frame(i)` instead of
`ns['fig']` / `ani._func(...)`. Task 8's `_import_example_without_fetching` + `_drive` is already
the correct shape; reuse it.

### F2. Step 0's loader/builder split is deleted by the very next step, in all four tasks that write it

| task | Step 0 says | the next step says |
|-|-|-|
| 2 market | produce `load_market` / `fixture_data` / `construct_artifact` + `__main__` guard | L1031 "Replace `examples/animate_market_forecast.py` **entirely**:" |
| 3 weather | same | L1371 "Replace `examples/animate_weather_decades.py` **entirely**:" |
| 4 paintings | same | L1620 "Keep the `PAINTINGS` dict verbatim … and **replace everything else**" |
| 5 conversation | same | L1904 "Keep `SPEAKER_COLOR` and the `TURNS` list verbatim … **Replace everything below**" |

Measured against the four prescribed blocks (extracted verbatim from the plan):

```
market         code= 109 native= 16 ratio=14.7%  markers=[]
               has __main__: False; construct_artifact: False; fixture_data: False
weather        code=  56 native= 14 ratio=25.0%  markers=[]
               has __main__: False; construct_artifact: False; fixture_data: False
paintings      code=  60 native= 14 ratio=23.3%  markers=[]
               has __main__: False; construct_artifact: False; fixture_data: False
conversation   code=  52 native= 13 ratio=25.0%  markers=[]
               has __main__: False; construct_artifact: False; fixture_data: False
```

Not one of them defines `construct_artifact`, `fixture_data`, or a `__main__` guard. An executor
who follows the steps in order does Step 0, then obliterates it. The result is precisely the failure
Step 0b warns about (L2777): *"skipping this step leaves `test_examples_produce_their_stated_artifact`
calling names that do not exist"* — and `_import_example_without_fetching` will `assert` on the
missing guard.

Task 2 and Task 5 carry v3 banners saying "reconcile rather than overwrite", which *contradicts*
their own Step 2/Step 1 text rather than resolving it. Tasks 3 and 4 have no such banner on this
point.

**Minimum fix:** rewrite the four prescribed blocks so they already contain the split (payload
`NamedTuple`, `load_*`, `fixture_data`, `construct_artifact`, `__main__`), and delete the
"replace entirely" wording, or move Step 0 to *after* the rewrite in every task.

---

## HIGH

### H1. `test_every_launch_notebook_ran_every_cell_it_should` is a control, and is described twice as the enforcement mechanism

```
$ .venv/bin/python -c "<dump execution_count / outputs per code cell>"
market_forecast     ncode=7 exec_counts=[None,2,3,4,5,6,7] n_outputs=[0,0,1,0,1,1,1] install_idx=[0]
weather_decades     ncode=7 exec_counts=[None,2,3,4,5,6,7] n_outputs=[0,0,1,0,0,0,1] install_idx=[0]
painting_embeddings ncode=6 exec_counts=[None,2,3,4,5,6]   n_outputs=[0,0,0,2,0,1]   install_idx=[0]
conversation_shape  ncode=6 exec_counts=[None,2,3,4,5,6]   n_outputs=[0,0,2,0,0,1]   install_idx=[0]
morph_shapes_zoo    ncode=6 exec_counts=[None,2,3,4,5,6]   n_outputs=[0,0,0,0,0,1]   install_idx=[0]
```

Every non-install code cell already carries an `execution_count`; only the install cell is `None`,
and `_is_install_cell` exempts it. The test therefore **passes on all five today** (confirmed:
`5 PASSED test_every_launch_notebook_ran_every_cell_it_should`). The defect is missing *outputs*,
not missing execution.

But the plan asserts twice that this test pins the defect:

- L4213–4214: *"They currently ship only **1–4 executed cells each** … **(implemented)** both halves:
  execute them (Tasks 2–6 …, **pinned by `test_every_launch_notebook_ran_every_cell_it_should`**)"*
- L4262 (Remaining risk 3): *"`test_every_launch_notebook_ran_every_cell_it_should` **makes skipping
  it a test failure** rather than an oversight."*

Neither is true of the state it is claimed to detect. The plan labels its other controls honestly
(`test_each_notebook_ships_its_rendered_artifact`: *"It PASSES today … a CONTROL, not coverage"*;
`stock_forecasting`/`projectile_kalman`). This one is not labelled, and is load-bearing in the
"Decisions still needed" and "Remaining risk" sections.

Answering your attack directly — **a notebook that passes all four parts while being broken**: take
today's `morph_shapes_zoo.ipynb`. It (a) passes the execution gate, (b) has no error outputs,
(c) embeds `morph_zoo.gif`, which exists, and (d) would pass the index-set test the moment its
current one-output set `{5}` is recorded. Five of its six code cells emit nothing; the docs page
renders code and one line of text. Nothing in the four-part contract objects.

**Minimum fix:** label the test a control in its own docstring, and delete the two claims above —
or add the property that actually matters (e.g. every cell that calls `hyp.plot`/`hyp.describe`
must carry a non-`stderr`, non-widget output).

### H2. The +15 split overhead was applied to the budgets and to the gate, but to none of the five "Execute and measure" steps — weather's is now unsatisfiable

| task | its "Execute and measure" step says | its AFTER line + enforced `SCRIPT_BUDGETS` say |
|-|-|-|
| 2 market (L1296) | ≤ 115 / ≤ 120 | ≤ 130 / ≤ 135 |
| 3 weather (L1541) | **≤ 62** / ≤ 67 | ≤ 77 / ≤ 82 |
| 4 paintings (L1820) | ≤ 118 / ≤ 123 | ≤ 133 / ≤ 138 |
| 5 conversation (L2254) | ≤ 90 / ≤ 95 | ≤ 105 / ≤ 110 |
| 6 morph (L2427) | **≤ 30** / ≤ 35 | ≤ 45 / ≤ 50 |

Weather's own AFTER line (L1323) states the arithmetic that makes L1541 impossible: *"The rewrite
alone measures 56, and 56 + 15 = 71 ≤ 77."* A file that measures 71 cannot satisfy "≤ 62". The same
line then names this exact instance as a fixed defect: *"v3 briefly had 62 here with the split
mandated on top, i.e. 71 against 62 — unsatisfiable, the exact class this plan claims to have made
impossible."* **It was fixed at L1323 and left at L1541.** Fifth documented-but-not-done.

Morph is the same shape: 26 code lines today (measured) plus a split overhead the plan itself puts
at ~15, against a Step 4 instruction of "≤ 30".

Contract 6 forbids the escape hatch: *"the assertion is never weakened to fit the code"*, so an
executor who hits 71-against-62 has no sanctioned move except renegotiating the plan mid-flight.

**Minimum fix:** propagate `+15` into all five "Execute and measure" steps, and into Contract 6b's
verification line (L185, still 120/67/123/95/35).

### H3. Two Task 8 gates are contradicted by Tasks 2/3/5's prescribed content

Measured on the four prescribed rewrite blocks:

```
weather        contains 'ani._func' anywhere: False | 'ani._args': False | 'hypertools._shared': False
conversation   contains 'ani._func' anywhere: False | 'ani._args': False | 'hypertools._shared': False
market         contains 'ani._func' anywhere: False | 'ani._args': False | 'hypertools._shared': False
paintings      contains 'ani._func' anywhere: False | 'ani._args': False | 'hypertools._shared': False
```

1. `test_every_allowlisted_reach_is_still_present_and_still_explained` asserts `ani._args` **and**
   `hypertools._shared` are *still present* in `examples/animate_market_forecast.py`
   (`assert hits, f'{path} no longer contains {marker!r}; drop the PRIVATE_API_EXCEPTIONS entry …'`).
   Task 2's "What goes, and to what" table deletes both (L911 and L914), and the prescribed block
   contains neither. → **FAILS after Task 2.**
2. `test_a_docstring_naming_a_removed_pattern_is_not_a_defect` asserts `'ani._func' in _read(path)`
   for weather and conversation. Neither prescribed block contains the string. → **FAILS after
   Tasks 3 and 5.**

Both tests pass today (confirmed in my run), so this is a regression the plan writes into itself.

Consequence for Step 3's arithmetic: applying the prescribed blocks makes the two allowlisted
`test_no_defect_marker_in_the_launch_examples` IDs *pass* rather than skip (the markers are gone),
so the outcome becomes **127 passed / 7 failed / 5 skipped**, not the stated "127 passed, 5 failed,
7 skipped". The `127` coincides; the other two numbers do not.

**Minimum fix:** decide, per file, whether Contract 3's allowlist survives Task 2 — and if it does,
say so inside the prescribed block rather than only in a banner above it.

### H4. `STATED_ARTIFACT['animate_painting_embeddings']['palette'] = True` is never read

```
$ ... inspect.getsource(test_examples_produce_their_stated_artifact)
occurrences of the word palette in the test body: 0
['want = STATED_ARTIFACT[stem]', "assert anim.n_frames >= want['min_frames'], (", ...,
 "if want.get('axes'):", "if want.get('predicts'):", "if want.get('on_frame'):", "if want.get('morph'):"]
```

The branches are `axes` / `predicts` / `on_frame` / `morph`. There is no `palette` branch. So
paintings — the only example whose whole purpose is Task 1's native palette, and the only one that
costs a committed fixture — is gated by nothing but `anim.n_frames >= 60`, against a prescribed
`duration, fps = 12, 20` (240 frames, L1729). The module docstring's claim that these gates each
*"fail loudly if the rewrite drops the thing the example is for"* is false for one of the five.

**Minimum fix:** add the branch, e.g. assert the drawn cloud colours match
`image_palette(<fixture thumbnail>)[0]` to within tolerance — the same assertion
`test_a_vivid_minority_colour_beats_the_muted_background` already makes at the library level.

### H5. Step 0b's worked example — the pattern the plan tells you to copy — calls two names that do not exist

```python
def fixture_data():
    hemis = [hemi for _n, _lat, _lon, hemi in cities_spec()]     # plan L2844
    return Weather([synthetic_city_months(h) for h in hemis],
                   [synthetic_city_daily(h, N_DAYS) for h in hemis],  # plan L2846
                   hemis, 'synthetic (fixture)')
```

```
$ grep -rn "N_DAYS\|cities_spec" examples/ hypertools/ scripts/
(no output)
$ grep -n "cities_spec\|N_DAYS" docs/superpowers/plans/2026-07-28-...md
2844:    hemis = [hemi for _n, _lat, _lon, hemi in cities_spec()]
2846:                   [synthetic_city_daily(h, N_DAYS) for h in hemis],
```

Neither exists in the repo, and neither is defined anywhere else in the plan. Worse, the shape does
not match: `CITIES` (`examples/animate_weather_decades.py:62`) is a **dict** of
`name -> (lat, lon, hemi)`, so a 4-tuple unpack has nothing to iterate. The plan asserts (L2813)
*"Every one of these `fixture_data()` bodies calls synthetic functions the example already has"* —
two of the three calls here do (`synthetic_city_months(hemi, n_months=420, seed=0)`,
`synthetic_city_daily(hemi, n_days, seed=0)` — both verified present); the other two names are
invented.

**Minimum fix:** `hemis = [h for _lat, _lon, h in CITIES.values()]`, and give `N_DAYS` a value
(the current example passes `n_days` positionally from its fetch loop).

---

## MEDIUM

### M1. `GUARD_KNOWN_UNCAUGHT` omits the simplest evasion, so the guard's "honest move is to name the holes" docstring is not honest

Constructed evasions **not** in `GUARD_KNOWN_UNCAUGHT`, run through the extracted
`_unpacked_wrapper_uses`:

```
A alias OF an unpacked name    'fig, ani = hyp.plot(d)\nb = ani\nb.on_frame(cb)\n'          -> []
B alias of subscript result    'anim=hyp.plot(d)\nani=anim[1]\nb=ani\nb.on_frame(cb)\n'      -> []
C alias of .animation          'ani=hyp.plot(d).animation\nb=ani\nb.on_frame(cb)\n'          -> []
D passed as a function arg     'def use(a): a.on_frame(cb)\nfig,ani=hyp.plot(d)\nuse(ani)\n' -> []
E tuple-of-tuple target        '(fig, ani), z = hyp.plot(d), 1\nani.on_frame(cb)\n'          -> []
F for over a Name iterable     'R=[hyp.plot(d)]\nfor fig,ani in R:\n    ani.on_frame(cb)\n'  -> []
H reversed alias chain (>3)    5-hop chain written bottom-up                                 -> []
```

Case **A** is the one that matters: a bare `b = ani`. The guard *does* implement alias propagation
in the other direction — `GUARD_MUST_FLAG['alias chain']` (`a = hyp.plot(d)` / `b = a` /
`fig, ani = b`) is flagged, and there is a `for _ in range(3)` fixed point specifically to make
wrapper aliases propagate. A reader will assume the symmetry holds. It does not, and the docstring's
list of "what this deliberately does NOT catch" names five exotic spellings while omitting the
plainest one.

`test_the_contract_8_guard_actually_detects` itself is **not vacuous** — every `GUARD_MUST_FLAG`
entry is genuinely flagged, every `GUARD_MUST_IGNORE`/`_FOREIGN` entry genuinely ignored, and the
`GUARD_KNOWN_UNCAUGHT` loop is a real tripwire that fails if the guard is strengthened. Confirmed
`1 PASSED`. The defect is the completeness claim, not the test.

**Minimum fix:** add case A to `GUARD_MUST_FLAG` and propagate `unpacked` through plain aliases in
the same fixed-point loop (three added lines); add B/C/E to the docstring's known-holes list.

### M2. The guard false-positives on the FIGURE half of a legitimate unpack, with a factually wrong message

```
I 'fig, ani = hyp.plot(d)\nprint(fig.figure)\n'   -> [('fig', 'figure')]
$ .venv/bin/python -c "from matplotlib.figure import Figure; print(type(Figure().figure).__name__)"
Figure
```

The reported message would read *"`fig` comes from unpacking a hyp.plot() result, so it is a raw
FuncAnimation and has no `.figure`"* — wrong twice: `fig` is a `Figure`, and `Figure.figure` is real
public matplotlib API. `note_unpack` adds **every** `Name` in the tuple target, including the figure,
and `WRAPPER_ONLY` contains `'figure'`.

This is the same class `_hypertools_names`' docstring says it fixed: *"`Line2D.figure`/`Axes.figure`
are real public attributes, so `WRAPPER_ONLY` containing `figure` turned them into false positives
with a factually wrong message."* Fixed for `ax.plot`/`df.plot`; left for the figure element of a
hypertools unpack — and the plan prescribes `fig, ani = anim` in the market example and
`fig, ani = hyp.plot(...)` in morph and paintings, so the trigger is in gated files.

**Minimum fix:** in `note_unpack`, add only elements at index ≥ 1 (or exclude `'figure'` from the
check for element 0).

### M3. The paintings fixture thumbnail has no path, no File Structure row, and no `git add`

Contract 4 (L169), Step 0b (L2797) and Task 4 Step 0 (L1592) all require "one committed 1.7 KB
64-px thumbnail". Its filename and directory are never given. It is absent from the File Structure
table, and from every `git add` in the plan:

```
--- L1825
    git add examples/animate_painting_embeddings.py docs/tutorials/painting_embeddings.ipynb
```

Global Constraints (L212) warn that exactly this trips
`test_sdist_contains_only_tracked_files_plus_allowlist`. The same table row also contradicts
itself: *"the one committed 1.7 KB 64-px thumbnail — no network, **no committed bytes unless
stated**"*.

### M4. `EXPECTED_VISIBLE_OUTPUTS` is written into a file that does not exist yet

Tasks 2–6's measure steps (L1296, 1541, 1820, 2254, 2427) each say "Record the measured
visible-output index set into `EXPECTED_VISIBLE_OUTPUTS`". That dict is created in Task 8 Step 2,
which runs after all of them; Task 8 Step 3 then says the five reds are the instruction and you fill
them in there. Both cannot be the procedure. (Same ordering wrinkle the plan already flags for
`scripts/measure_native_ratio.py` at L4254 — apply the same remedy.)

### M5. `test_the_right_cells_carry_visible_output` says "visible output" but tests `bool(cell['outputs'])`

`got = {i for i, c in enumerate(cells) if c.get('outputs')} - installs`. The plan itself records
(L3967–3971) that the only `display_data` in these notebooks are tqdm progress widgets from
`sentence_transformers`, so a cell whose sole output is a progress bar counts as "visible". The test
does discriminate (an index set fails and names the cell), but it is a *regression pin recorded from
the artifact it gates* — record it once from a figure-less notebook and it passes forever. Combined
with H1 this is why the four-part notebook contract does not force a figure onto the docs page.

---

## LOW

| # | finding | evidence |
|-|-|-|
| L1 | Task 8 Step 0's first run is stated as "**9 failed**"; the block defines **8** `def test_`. | `len(re.findall(r'^def test_', block, re.M))` → `8` |
| L2 | Task 8 Step 0's "whole suite → baseline + **8**" contradicts the finished file's 9 tests and the Self-Review's `148 = 139 + 9`. | same measurement |
| L3 | Self-Review L4244 still calls it a "**109-test** gate"; it is **139** (correctly itemised in Step 3). | `139 tests collected` |
| L4 | Self-Review L4247's citation list still contains `plot.py:2750-2751`, which the plan's own Verification note (L66) documents as wrong. Also `plot.py:1066` is cited for the `palette` docstring entry (File Structure L270, Task 1 Step 6 L824) — line 1066 is inside the `linewidth` entry; `palette :` is at **1074**. | `sed -n '1066p;1074p' hypertools/plot/plot.py` |
| L5 | Contract 6b (L185) still carries pre-split notebook figures (120/67/123/95/35) vs the derived 135/82/138/110/50. | `BUDGETS` in the extracted gate |
| L6 | `_import_example_without_fetching` asserts the literal `"__name__ == '__main__'"`; a double-quoted guard fails with a message blaming the split. | source read |
| L7 | `HYPERTOOLS_OFFLINE` is popped in the helper's `finally`, so `construct_artifact(fixture_data())` runs with the guard off. Belt without braces. | source read |
| L8 | Task 6 Step 2's verification uses `ani._func(frame, *ani._args)` — a `DEFECT_MARKERS` pattern — although `HyperAnimation.draw_frame()` exists as of Task 8 Step 0. Task 8's `_drive()` was migrated for exactly this reason; Tasks 2–6 were not. | plan L2391, L3795–3802 |
| L9 | Task 4's "rewrite alone measures **111**" is right only because `PAINTINGS = {` / `...  # UNCHANGED` / `}` is a 3-line stub: 60 − 3 + 54 = 111. A naive splice measures 114. | `measure()` on block + on `src[42:96]` |
| L10 | `examples/animate_morph_zoo.py`'s shipped comment still says "for the 5 clouds = 9 segments" while the file passes 6 clouds / 11 segments. The plan documents the discrepancy (L3740–3744) but never instructs fixing the comment. | file read |

---

## Checked and CONFIRMED CORRECT

So you can tell coverage from silence. Everything here I ran; none of it is a finding.

| claim | how checked | result |
|-|-|-|
| Gate red state `139 collected / 39 failed / 93 passed / 7 skipped` | extracted + ran | **exact match** |
| Step 3's 16-row ID table sums to 139 | hand sum + real collection | 139 = 139 |
| **N9** baseline table (all 10 rows) | `measure_native_ratio.py` on the ten files | conversation 165/9/5.5, market 191/11/5.8, morph 26/6/23.1, paintings 146/11/7.5, weather 195/11/5.6; nbs 176/11/6.2, 187/11/5.9, 46/9/19.6, 121/11/9.1, 194/11/5.7 — **every row exact** |
| five-script total `48 / 723 = 6.6%` | summed the above | 48/723 = 6.639% ✓ |
| Notebook output counts 4/7, 2/7, 2/6, 2/6, 1/6 | JSON walk | **all five exact** |
| **N12** Contract 8 guard extended to `AnnAssign` / `NamedExpr` / `For` | ran all 12 `GUARD_MUST_FLAG` cases | all flagged |
| **N12** `test_the_contract_8_guard_actually_detects` exists and is not vacuous | ran it; ran all IGNORE / FOREIGN / KNOWN_UNCAUGHT arms | `1 PASSED`; every arm discriminates |
| **N12b** no line numbers ship into example/notebook source | regex `(plot\|colors\|animate\|matplotlib_backend\|helpers)\.py:\d` over all 5 prescribed source blocks | **0 hits in all five** |
| **N10** `git add` lists | extracted all 12 | complete except M3 (paintings fixture) |
| Task 1: 19 `def test_` and "19 passed" | applied all 3 patches to a scratch worktree, ran the prescribed test file | `19 passed in 2.28s` |
| Task 1: six consumer paths score 6/6 | ran real `hyp.plot` calls on the patched tree | **7/7** (their six + the colorbar path) |
| Task 1: no colour regressions | `tests/test_colors.py`, `tests/plot/test_colors_module.py`, `tests/test_colorbar.py`, `tests/test_docstring_examples_audit.py` on the patched tree | `45 passed` |
| `colors.py:305-306` (string branch), `:287`, `:227`, `:250`, `:106`, `:118`, `:158`, `:246`, `:259-260` | `sed`-printed each | **every citation exact** |
| `_seaborn_palette_arg` at `plot.py:113`, function spans `113-124` | grep + read | exact |
| `_widths = [ri.shape[1] for ri in raw]` | grep | now at **3165** (plan says re-derive by symbol — correct guidance) |
| Task 8 Step 0: `_save_count == 40` at duration=4/rate=10 | ran `hyp.plot` | 40 |
| Task 8 Step 0: sub-frame request gives 1 | duration=0.01/rate=1 | 1 |
| Task 8 Step 0: 3 clouds → 5 morph segments | `segment_frame_counts(3, 30)` | 5 |
| Task 8 Step 0: two morph `FuncAnimation` sites, 3-D `:2036`(`sum` `:2039`) and 2-D `:2448`(`:2451`) | grep `matplotlib_backend.py` | **both exact** |
| Raw `FuncAnimation` has no `.figure` (Contract 8's premise) | `hasattr(a[1], 'figure')` | `False` |
| `_hyp_forecast_role == 'live'` exists for the `predicts` gate | grep | `plot.py:5011` |
| all five `min_frames` floors are clearable | duration×fps from the prescribed blocks: 160/160/240/192/240 vs floors 100/100/60/100/200 | all clear |
| morph 6 clouds → 11 segments, 11-entry `rotations` | `[0.75] + [0.5,1.0]*4 + [0.5,0.75]` | 11 = 11 |
| Task 6 Step 1 "already landed" | `grep -nE "_morph\|morph_schedule\|ani\._func\|shape_title"` → empty; `title=titles` present | **verified done** |
| Task 6 does NOT overwrite the newer morph script with stale code | Step 1 is `- [x]` and marked "do not apply"; the fallback block is gated behind "if the greps do NOT come back clean" | **correctly rebased** |
| conversation rewrite measures 88 | kept lines 1–85 + prescribed block, `measure()` | 88 |
| market rewrite 109, weather 56 | `measure()` on the blocks | 109, 56 |
| Suite delta `+179 = 19 + 12 + 148` | 19 (counted), 12 (8 defs, 2 parametrized ×3), 148 (139 + 9) | arithmetic sound given L1/L2 |
| `docs/conf.py:131` = `nbsphinx_execute = 'never'` | `sed -n '131p'` | exact |
| `scripts/generate_gallery_thumbs.py:26` = `MPL_ANIMS` | `sed -n '26p'` | exact |
| `docs/_static/thumbnails/` holds 12 files | `ls \| wc -l` | 12 |
| 10 `examples/animate_*.py`, all with 0 `__main__` | `grep -c` | 10 files, all 0 |
| sphinx-gallery runs examples under a fake `__main__` at `gen_rst.py:1271-1280` | read installed sphinx_gallery 0.21.0 | *"Examples may contain if `__name__ == '__main__'` guards"* at 1271 |
| `from scripts.measure_native_ratio import measure` works without `scripts/__init__.py` | `tests/__init__.py` exists; prepend mode inserts rootdir | confirmed by my own run |
| adding `scripts/__init__.py` cannot leak into the wheel | `[tool.setuptools.packages.find] include = ["hypertools", "hypertools.*"]` | **safe** |
| `test_notebook_budgets_are_derived_not_written_down` is a real equality check | ran | passes, and would fail on a hand-written number |
| **Round-1/2 fixes still in place** — `strip_docstrings` is ast-based and shared; `_code_lines_nb` strips docstrings; `_code_text` returns a string; `_save_count` gone from the gate; the `'morph' in 'morph'` tautology gone; no `git stash` recipe anywhere | read + ran | **all confirmed fixed, none regressed** |

---

## VERDICT

**NOT IMPLEMENTABLE as written.**

Tasks 1, 7 and 8-Step-0 are implementable today and I would sign those off unchanged: the palette
work is fully verified (19/19 tests, 7/7 consumer paths, no colour regressions, every `colors.py`
citation exact), and the accessor design is exact against the real backend at both morph sites.

The blocker is that the v3 loader/builder split was added to Tasks 2–6 without being threaded
through the v2 material that surrounds it. Step 0 creates the split; the next step deletes it (F2);
and every verification step in those tasks still reads module-level names the split removes (F1).
An executor following the steps in order cannot get past Task 2 Step 4.

**Minimum work to make it implementable:**

1. **F2** — rewrite the four prescribed example blocks so they already contain
   `NamedTuple` / `load_*` / `fixture_data` / `construct_artifact` / `__main__`, and drop the
   "replace entirely" / "replace everything else" wording. (Largest item; ~4 blocks.)
2. **F1** — replace the five `runpy.run_path` verification blocks with
   `spec_from_file_location` + `exec_module` + `construct_artifact(fixture_data())`, driving frames
   with `anim.draw_frame(i)`. Reuse Task 8's `_import_example_without_fetching` / `_drive`. Fixes L8
   at the same time.
3. **H2** — propagate `+15` into all five "Execute and measure" steps and into Contract 6b's L185.
4. **H3** — decide per file whether the two `PRIVATE_API_EXCEPTIONS` reaches and the two
   `ani._func` docstring mentions survive, and put that decision *inside* the prescribed blocks;
   then re-derive Step 3's expected pass/fail/skip split.
5. **H5** — fix `cities_spec()` / `N_DAYS` in the Step 0b worked example.
6. **H1** — relabel `test_every_launch_notebook_ran_every_cell_it_should` as a control and delete
   the two claims that it pins the unexecuted-notebook defect (L4214, L4262).
7. **H4** — give `palette=True` a branch, or delete the key.
8. **M3** — name the paintings fixture path, add it to File Structure and to Task 4's `git add`.
9. **M1/M2** — three lines in `_unpacked_wrapper_uses` (propagate `unpacked` through aliases;
   skip element 0 in `note_unpack`), plus the docstring update.
10. **M4, L1–L5, L9, L10** — editorial.

Items 1–2 are the only ones that require new prose of any length. Everything else is a number, a
label, or a few lines.
