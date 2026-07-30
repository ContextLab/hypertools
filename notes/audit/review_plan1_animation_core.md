# Adversarial review — `docs/superpowers/plans/2026-07-26-hypertools-1.1-animation-core.md`

Repo: `/Users/jmanning/hypertools`, branch `dev-1.0`. All code run with `.venv/bin/python`
(Python 3.12.10, matplotlib 3.10.8, numpy 2.3.5, scipy 1.17.0, pytest 9.0.2).

**Baseline claim check — PASSES.** Plan says "2551 passed, 13 skipped" (= 2564).

```
$ .venv/bin/python -m pytest -q --collect-only 2>&1 | tail -3
tests/test_window_animation.py::test_focused_invalid_value_raises

2564/2566 tests collected (2 deselected) in 7.46s
```

**Task soundness one-liners.**

| task | verdict |
|-|-|
| 1 — reject non-string `title=` | premise correct, nothing in repo depends on stringification; **placement is wrong** (G3) |
| 2 — `linewidth=` in animated hue | **broken**: test can't run, then can't fail, and the patch is in the wrong file (C1–C3) |
| 3 — default `morph_samples` cap | **broken**: snippet references two undefined names (C4); contradicts a maintainer guarantee (G4) |
| 4 — `order=` axis | **broken**: gate unreachable for list-form morph (C5); one-site `backend_mode` insufficient (C6); 3/8 tests vacuous (T1) |
| 5 — per-dataset `alpha=` | mechanism is sound and plotly needs no change, but precedence vs. two internal alpha writers is undefined (G1, G2) |
| 6 — `on_frame` hook | reveal formula is **correct**; `HyperAnimation` attrs **do** work; but `HyperAnimation.on_frame()` as written cannot be wired (C7) and 2-D gets no hook (G5) |
| 7 — per-segment titles | **broken**: `not in (0.0, 1.0)` inverts the requirement (C8) and its own test can't detect that (C9) |
| 8 — CHANGELOG/docs | no findings |

---

## CRITICAL FINDINGS

### C1 — Task 2 Step 1: wrong `hue=` cardinality; both tests error before any assertion

`_hue_datasets()` returns 3 datasets × 30 rows = **90 observations**, but the test builds
`hue = np.linspace(0.0, 1.0, ds[0].shape[0])` = **30 values**. The validator at
`/Users/jmanning/hypertools/hypertools/plot/plot.py:3372` demands one entry per observation.

Ran the plan's test file verbatim:

```
E   ValueError: hue has 30 entries but the data has 90 observations; hue must
    have exactly one value (or one row, for a matrix hue) per observation.
hypertools/plot/plot.py:3372: ValueError
FAILED test_animated_continuous_hue_honors_per_dataset_linewidth
FAILED test_static_continuous_hue_linewidth_still_correct
============================== 2 failed in 1.65s ===============================
```

This is the same defect class as the sibling plan's "120 hue values where 480 demanded".
Fix: `hue = np.linspace(0.0, 1.0, sum(d.shape[0] for d in ds))`.

### C2 — Task 2 Step 1: with C1 fixed the test PASSES on unfixed code — `_widths()` reads the wrong artists

After correcting the hue length, the plan's test passes against current `dev-1.0`:

```
task2_test.py::test_animated_continuous_hue_honors_per_dataset_linewidth
ANIM n_collections= 9 n_lines= 3
ANIM coll widths= [1.5, 1.5, 1.5, 1, 1, 1, 1, 1, 1]
ANIM line widths= [0.5, 0.5, 5.0]
PASSED
task2_test.py::test_static_continuous_hue_linewidth_still_correct
STATIC widths= [1, 1, 1, 1, 1, 1, 0.5, 0.5, 5.0]
PASSED
============================== 2 passed in 1.83s ===============================
```

Detailed artist dump:

```
STATIC collections: 6 × Line3DCollection lw=1 (cube planes), then lw=[0.5, 0.5, 5.0]
STATIC lines:       []                          <- lines are swapped out entirely
ANIM  collections:  lw=[1.5, 1.5, 1.5] (head colls = THE BUG), then 6 × lw=1
ANIM  lines:        [(0.5, visible=False), (0.5, False), (5.0, False)]
rcParams['lines.linewidth'] = 1.5
```

`_widths(ax)` unions `ax.collections` **and** `ax.lines`. In the animated case the
head `Line2D`s are hidden but still carry the correct 0.5/0.5/5.0 (they are created with
`linewidth=linewidths[idx]`, `matplotlib_backend.py:1627`), so `max(widths)==5.0` and
`min(widths)==0.5` hold regardless of the bug. The test therefore cannot fail.

Also note the assertions are wrong for the obvious repair: restricting `_widths()` to
`ax.collections` makes `min(widths) == 1` (the six cube-plane collections), so
`assert min(widths) == pytest.approx(0.5)` would fail *after* a correct fix. The test must
target only the three head collections.

### C3 — Task 2 Step 3: the patch is in the wrong file and cannot be applied as written

The plan lists only `matplotlib_backend.py:1602-1604` / `:2197-2199` as modified. Verified
line numbers are right, but:

* The **reader** is `_apply_multicolor_animation`, which lives in
  `/Users/jmanning/hypertools/hypertools/plot/plot.py:5104`, with
  `plot.py:5150-5153`:
  ```python
  def _linewidth(i):
      tkwargs = kwargs_list[i] if i < len(kwargs_list) else {}
      return (tkwargs.get('linewidth')
              or plt.rcParams['lines.linewidth'])
  ```
  The collections are built in **plot.py** (`_make_collection`, `plot.py:5161-5176`), so
  "pass `requested_linewidth` through to the collection construction" has no meaning in
  `matplotlib_backend.py`. `plot.py` is absent from Task 2's file list and its `git add`.
* The snippet names `dataset_kwargs`, which does not exist. The real code is a *list
  comprehension over all datasets* (`matplotlib_backend.py:1602-1606`), so the replacement
  must produce per-dataset **lists**, not the scalar `line_kwargs` / `requested_linewidth`
  the plan writes.
* If `linewidth` merely stops being popped, `kwargs_list[idx]` still contains it and the
  four surviving `**kwargs_list[idx]` expansions collide with the explicit
  `linewidth=linewidths[idx]` argument → `TypeError: ax.plot() got multiple values for
  keyword argument 'linewidth'`. Sites: `matplotlib_backend.py:1621-1631`, `1633-1643`,
  `1645+`, and the 2-D twins at `2214-2223`, `2225+`.

The minimal correct fix is one line in plot.py — read the width off the already-correct
hidden head artist (`head_lines[i].get_linewidth()`, `plot.py:5138`) — which the plan
never considers.

### C4 — Task 3 Step 3: the snippet references two names that do not exist

```python
if mode == 'morph' and morph_samples is None:
    largest = max(int(np.asarray(d).shape[0]) for d in datasets)
```

* `mode` — there is no such local. The mode is the rebound `animate`
  (`plot.py:3653`: `animate, morph_tags = _resolve_animate_mode(animate, len(xform))`).
* `datasets` — no such local. `grep -n "datasets" plot.py` returns only docstrings,
  `n_datasets`, and `n_morph_datasets`. The post-pipeline arrays are `xform`.

There is also **no "where `morph_samples` is resolved"** to insert into: `morph_samples`
is *validated* early (`plot.py:2261-2274`) and then passed verbatim to the backends at
`plot.py:4239` (plotly) and `plot.py:4324` (matplotlib). Choosing a hook point is a real
design decision the plan skips: the row count is only knowable from `xform` (line 3600+)
or from raw `x`, and morph-tag filtering (`morph_tags`) is only known after line 3653.

Timing check (the premise is otherwise sound):

```
n=1000  0.41 s   cost-matrix 0.008 GB
n=2000  3.02 s   cost-matrix 0.032 GB
n=4000  25.82 s  cost-matrix 0.128 GB
```

⇒ ~O(n³); 12 000 points ≈ 11–12 min, 30 000 points ≈ >3 h and a 7.2 GB cost matrix.
Note `pytest-timeout` **is** installed (2.4.0) and `pyproject.toml` sets `timeout = 1200`,
so the "hang" is actually a 20-minute kill — Step 2's hedge is unnecessary but harmless.

### C5 — Task 4 Step 3: the serial-capability gate is unreachable for the list form of `animate=`

The plan orders the new checks *before* "existing list/morph handling":

```python
if order == 'serial' and animate not in _SERIAL_CAPABLE_STYLES:
    raise NotImplementedError(...)
# ... existing list/morph handling, unchanged ...
```

But `animate` may legitimately be a **per-dataset list** (`plot.py:480-505`), e.g.
`animate=['morph', None, 'morph']`, which `_resolve_animate_mode` resolves to `'morph'` —
a style the plan itself declares serial-capable. As written,
`['morph', None, 'morph'] not in (True, 'parallel', 'serial', 'morph')` is `True`, so a
valid combination raises `NotImplementedError`. The gate must run *after* resolution, on
the resolved `mode`, not on the raw argument.

(`animate=` dicts are fine — `plot.py:2119-2181` unwraps `{'style': ...}` to a scalar long
before line 3653.)

### C6 — Task 4 Step 3: a single `backend_mode` substitution is not enough

`animate` is consumed at four *semantically distinct* sites after `plot.py:3653`, all of
which need the resolved backend mode, not just "where the backend receives its mode":

1. `plot.py:3760` `if animate in _trail_ignoring_modes:` — and `plot.py:3755-3758`:
   ```python
   _trail_ignoring_modes = ("spin", "morph", "window")
   if resolve_backend(backend) == "plotly":
       _trail_ignoring_modes = _trail_ignoring_modes + ("serial",)
   ```
   So `animate=True, order='serial', chemtrails=True` on **plotly** would silently drop the
   trails with **no warning** (the check sees `animate is True`). That is exactly the
   silent-drop the repo added this warning to prevent, reintroduced by the new spelling.
2. `plot.py:4214` `animate=animate` → `plotly_draw`.
3. `plot.py:4299` `animate=animate` → matplotlib draw.
4. `plot.py:4379` `style=animate` → `_apply_multicolor_animation`, which branches on
   `if style == 'serial':` at `plot.py:5258` to recover the serial reveal position:
   ```python
   _lengths = [_points(j).shape[0] for j in range(n)]
   _start_i = int(sum(_lengths[:i]))
   revealed = (sum(_lengths) * num / max(1, int(total_frames) - 1))
   shown = int(np.clip(revealed - _start_i, 0, n_pts))
   ```
   With `hue=` + `order='serial'` the multicolor overlay would use the *parallel* window
   while the backend animates serially — a visible desync, untested by the plan.

Verified current behavior (measured artist counts, `duration=3, frame_rate=4`, frame 7):

```
animate=True          (3 lines, 6 colls)
animate='serial'      (3 lines, 6 colls)
serial+chemtrails     (6 lines, 6 colls)
parallel+chemtrails   (6 lines, 6 colls)
spin+chemtrails       (3 lines, 6 colls)  + UserWarning:
    "animate='spin' does not support trail styles; ignoring chemtrails for datasets [0, 1, 2]"
```

Note this also shows the plan's `NotImplementedError` for `animate='spin', order='serial'`
is a *new hard error* where the established repo behavior for the same intent is
warn-and-ignore.

### C7 — Task 6 Step 5: `HyperAnimation.on_frame()` as specified can never fire

The plan's stated worry ("`HyperAnimation` subclasses `tuple`") is **unfounded** — verified:

```
tuple subclass attrs OK: [1]   has __dict__: True   slots? None
```

`hypertools/plot/hyper_animation.py:45,53-55` defines no `__slots__`, so instance
attributes set in `__new__` work and survive.

The real defect is list **identity**. The per-frame updater closure is created inside
`matplotlib_backend.animate_plot3D` (around `matplotlib_backend.py:1316`), long before
`plot()` wraps the result in a `HyperAnimation`. `self._frame_callbacks = []` in
`__new__` therefore creates a *fresh, unreferenced* list; anything appended by
`HyperAnimation.on_frame(cb)` is invisible to the updater, so
`test_hook_can_be_attached_after_construction` (asserting `len(seen) == 2`) fails with
`0 != 2`. The plan must specify a single shared mutable registry threaded from `plot()`
into the backend and adopted (not re-created) by `HyperAnimation.__new__`.

Related: with `return_model=True`, `plot()` hands back the RAW `FuncAnimation` and never
constructs a `HyperAnimation` (`hyper_animation.py:31-35`), so `.on_frame()` is
unavailable on that path — undocumented.

### C8 — Task 7 Step 4: `ctx.current_fraction not in (0.0, 1.0)` inverts the requirement

Computed against the real schedule for the plan's own test parameters (3 clouds,
`duration=6, frame_rate=4` ⇒ `total_frames=24`), using
`hypertools.plot.morph.segment_frame_counts` / `frame_to_segment`
(`morph.py:263-328`), `frame_counts = [5, 5, 5, 5, 4]`:

```
frame  seg  kind    step/n   fraction   plan blanks?
0      0    hold    0/5      0.000      no   <- correct
1      0    hold    1/5      0.250      YES  <- WRONG: hold blanked
2      0    hold    2/5      0.500      YES  <- WRONG
3      0    hold    3/5      0.750      YES  <- WRONG
4      0    hold    4/5      1.000      no
5      1    MORPH   0/5      0.000      no   <- WRONG: transition named
6      1    MORPH   1/5      0.250      YES
7      1    MORPH   2/5      0.500      YES
8      1    MORPH   3/5      0.750      YES
9      1    MORPH   4/5      1.000      no   <- WRONG: transition named
10     2    hold    0/5      0.000      no
11-13  2    hold    …        0.25-0.75  YES  <- WRONG
14     2    hold    4/5      1.000      no
15     3    MORPH   0/5      0.000      no   <- WRONG
16-18  3    MORPH   …        0.25-0.75  YES
19     3    MORPH   4/5      1.000      no   <- WRONG
20     4    hold    0/4      0.000      no
21-22  4    hold    …        0.33-0.67  YES  <- WRONG
23     4    hold    3/4      1.000      no
```

Exact-float equality is not the problem (`morph_positions`/`smoothstep` do produce exactly
0.0 and 1.0 at segment endpoints, `morph.py:342-346`); the problem is that **holds and
transitions both sweep 0→1**, so the fraction cannot distinguish them. This also
contradicts the `FrameContext.current_fraction` docstring the plan writes in Task 6
("How far through `current_index` the reveal has progressed, in [0, 1]"). The only correct
discriminator is `seg_idx % 2` from `frame_to_segment` — which the plan mentions in prose
but never puts in the contract or a test.

### C9 — Task 7 Step 1: the test that allegedly proves C8's feature passes under the inverted behavior

`test_morph_titles_blank_during_transitions` asserts only:
`'' in seen` (true: 12 blank frames), `{'alpha','beta','gamma'} & set(seen)` (true), and
`0.1 < blank_fraction < 0.9` — blank_fraction = **12/24 = 0.5**, comfortably inside the
window. The test cannot tell "named only during holds" from "named only at segment
endpoints, blanked mid-hold". A discriminating test would assert the title is non-empty
for *every* frame of a hold segment and empty for every *interior* frame of a transition,
derived from `frame_to_segment`.

---

## IMPORTANT GAPS

### G1 — Task 5: undefined precedence against two existing internal per-dataset `alpha` writers

`mpl_kwargs["alpha"]` is already set to a per-dataset **list** in two internal paths:

* `plot.py:3056` `mpl_kwargs["alpha"] = _mi_style["alphas"]` (row-MultiIndex level fading)
* `plot.py:3629` `mpl_kwargs["alpha"] = [max(0.3, 0.9 ** (d - min_depth)) ...]` (nested-list depth fading)

Today a user `alpha=` arrives through `**kwargs` and loses to these, per the documented
rule at `plot.py:71-75`:

> "A key already present in a given dataset's dict (set by a named parameter, e.g.
> `color=`, or by internal styling logic, e.g. **MultiIndex/mixture-cluster `alpha`**,
> `legend=`'s `label`, `explore=`'s `picker`) is left untouched — named/internal styling
> always wins over a same-named extra kwarg."

Promoting `alpha` to a named parameter that writes into `mpl_kwargs` silently reverses that
precedence. Note the MultiIndex branch already warns and *ignores* `linewidth=`
(`plot.py:3044-3050`); the plan adds no equivalent for `alpha=`, and no test covers a
MultiIndex or nested-list input with `alpha=`.

### G2 — Task 5: promotion breaks a documented invariant in `_expand_styles_to_runs`

`plot.py:243-245`, verbatim:

> "Generic ``**kwargs`` passthrough values (e.g. ``alpha=``) never reach `mpl_kwargs` --
> they are applied verbatim per trace, never broadcast -- so they are unaffected here."

Once `alpha` is named it *does* reach `mpl_kwargs`, so contiguous-run segmentation (hue /
cluster grouping turns N datasets into ≥ N runs) must expand it or `parse_kwargs` raises a
length `ValueError`. Related: the plan's `_validate_alpha(alpha, n_datasets)` never says
*which* count — `len(x)` (input datasets) or `len(xform)` (final, post cluster/hue-reshape,
which the plan's own Task 4 interface uses). They differ exactly in the cases that matter.

**Positive:** the "no backend change" claim holds for plotly —
`plotly_backend.py:776` reads per-dataset alpha straight off `kwargs_list`:
`color = _to_plotly_color(tkwargs.get('color'), tkwargs.get('alpha'))`. And measured
today: `hyp.plot(ds, '-', alpha=0.25)` → `[0.25, 0.25, 0.25]` on `ax.lines`;
`alpha=[0.1,0.5,1.0]` → `TypeError: alpha must be numeric or None, not <class 'list'>`
from `matplotlib/artist.py:1023`, exactly as the plan states.

Watch item: the existing `tests/test_gh206_extra_kwargs.py:71-78
::test_alpha_kwarg_reaches_line_artists` lives in the `**kwargs`-passthrough suite and
asserts `hyp.plot(data, alpha=0.42)` reaches every line — it must keep passing after alpha
leaves that mechanism, and its *placement* becomes misleading.

### G3 — Task 1: validation placement defeats the repo's own fail-fast rule and misses `plot_stream`

Step 3 says to call `_validate_title` "immediately after `_resolve_animate_mode(...)`" —
that is `plot.py:3653+`, **after** the whole analyze/reduce/align/cluster pipeline. This
contradicts the stated principle at `plot.py:423-430`:

> "Fail fast, BEFORE the analyze/reduce pipeline runs, on extra kwargs that no backend can
> use ... previously a renamed 0.x kwarg ran the whole pipeline and then died with a
> cryptic AttributeError"

Two concrete consequences:
* `plot_stream` returns at `plot.py:2583`, before line 3653, and `title` is in its
  `_stream_forwarded` set (`plot.py:2551-2556`) — streaming plots keep silently
  stringifying a list title.
* `resolve_font(font, _font_texts)` already consumes `title` at `plot.py:2424-2429`,
  before validation.

**Positive:** nothing in the repo depends on the current stringification. Confirmed
measured behavior:

```
TITLE list  -> "['a', 'b', 'c']"
TITLE tuple -> "('a', 'b', 'c')"
TITLE int   -> '3'
TITLE dict  -> "{'a': 1}"
```

and the only non-literal `title=` in tests is `tests/test_multibyte.py:711` (`title=text`,
where `text` is a `str` from a parametrize list); examples all use f-strings or
`.capitalize()` (`examples/plot_shapes_zoo.py:36`, `plot_datasaurus.py:34`,
`plot_impute.py:55`, `plot_datasets_tour.py:66`). Step 6's "fix the test's call"
contingency is unnecessary.

### G4 — Task 3: the default cap contradicts an explicit maintainer-requested guarantee

`hypertools/plot/morph.py:17-24`:

> "Full-sample morphs (maintainer request, 2026-07-06 follow-up): earlier versions of this
> module sampled every dataset down to the SMALLEST morphing dataset's point count. Every
> dataset now keeps its FULL point count instead ... **No real data point is ever dropped.**"

and `plot.py:1516-1518`:

> "Default `None`: no cap -- every dataset keeps its full point count, and the target count
> is simply the largest dataset's own size."

`tests/test_morph_animation.py:121-131 ::test_default_is_uncapped_target_is_largest_cloud`
encodes this. It calls `morph.sample_and_match_clouds` directly, so it will not break — but
after Task 3 the `plot()`-level default and the `morph` module default diverge silently.
This should be framed as a maintainer decision (or an opt-in / hard error naming
`morph_samples=`), not as a bug fix.

### G5 — Task 6: the "exactly one implementation" claim is unachievable at the stated scope, and 2-D gets no hook

The reveal formula exists in **at least three** places today, not one:

* `matplotlib_backend.py:1316-1318` (3-D `update_lines_serial`) — the one the plan names
* `matplotlib_backend.py:2062-2064` (identical 2-D twin in `animate_plot2D`)
* `plot.py:5265-5269` (re-derived inside `_apply_multicolor_animation`)

**Positive:** the plan's `serial_reveal_counts` is numerically identical to the real
formula (`shown = int(np.clip(revealed - start, 0, n_pts))`, `matplotlib_backend.py:1325`)
— `max(0, min(length, remaining))` ≡ `np.clip`, and `remaining` after i steps
≡ `revealed - sum(lengths[:i])`. Verified by simulation.

No proposed test drives a 2-D animation. Every `_datasets()` helper produces 4-column
data reduced to 3-D, and both `_drive` / `_titles_over` / `_artist_count` helpers select
`[a for a in fig.axes if hasattr(a, 'zaxis')][0]`, which raises `IndexError` on a 2-D
figure (verified: `hyp.plot(ds, ndims=2, animate='serial')` produces axes with no `zaxis`).

### G6 — Task 4: `order=` collides with an existing did-you-mean hint and with a predict model kwarg

Measured today:

```
hyp.plot(ds, '-', order=3) ->
TypeError: plot() got an unexpected keyword argument 'order'; did you mean 'zorder'?
```

After the change this becomes `ValueError: order must be 'parallel' or 'serial'` — the
`zorder` hint is lost for a plausible typo. `order=` is also ARIMA's `(p, d, q)` spelling
(`hypertools/predict/arima.py:182`), so the name is already overloaded inside the package.

### G7 — Task 4: no specified behavior for `order='serial'` with `animate=False`

`False not in (True, 'parallel', 'serial', 'morph')` ⇒ the most likely user mistake
(`hyp.plot(x, order='serial')` with no `animate=`) raises
`NotImplementedError: order='serial' is not implemented for animate=False`, which is a
confusing category and message. No test covers it. Contrast Task 6, which explicitly
raises `ValueError("on_frame requires an animated plot...")` for the same shape of error.

### G8 — Task 3: `stacklevel=2` breaks the repo's warning convention

Every other `warnings.warn` in `plot.py` uses `stacklevel=external_stacklevel()`
(e.g. `plot.py:3776-3781`, `3049`, `3372`-adjacent). `stacklevel=2` points inside
hypertools rather than at the user's call site.

---

## TEST WEAKNESSES

### T1 — Task 4: three of the eight `order=` tests are vacuous

`_artist_count` = `len(ax.lines) + len(ax.collections)`. Measured:

```
animate=True         3 + 6 =  9
animate='serial'     3 + 6 =  9
serial+chemtrails    6 + 6 = 12
parallel+chemtrails  6 + 6 = 12
```

* `test_order_defaults_to_parallel` — 9 == 9
* `test_order_serial_equals_the_legacy_animate_serial_alias` — 9 == 9
* `test_animate_serial_implies_order_serial` — 9 == 9

All three pass if `order=` is accepted and then **completely ignored**. They prove only
that the kwarg exists. `test_order_serial_composes_with_chemtrails` (12 > 9) discriminates
trails-vs-no-trails but not serial-vs-parallel (parallel+chemtrails is also 12). A real
test must inspect *which* points are drawn per frame (e.g. that only one dataset is
non-empty early in a serial reveal), not artist counts.

### T2 — Task 4 Step 4: "Expected: 11 passed" is wrong — the file defines 12 cases

5 plain + 4 (`test_invalid_order_raises` × `['Serial','sequential',True,1]`) +
2 (`test_unimplemented_serial_styles_raise_clearly` × `['spin','window']`) + 1 morph = 12.
(Other counts check out: Task 1 = 7 ✓, Task 2 = 2 ✓, Task 3 = 4 ✓, Task 5 = 5 ✓,
Task 6 = 10 ✓, Task 7 = 6 ✓.)

### T3 — Task 2 Step 2: the expected failure output is fabricated

The plan predicts `assert 1.5 == 5.0 ± 5.0e-06`. Actual: `ValueError: hue has 30 entries
but the data has 90 observations` (C1); and once that is fixed, `2 passed` (C2). Neither
the red nor the green state the plan describes is reachable.

### T4 — Task 6: `test_revealed_counts_match_the_library_reveal_formula` only tests the exact-division case

lengths `[20, 20, 20]`, `n_frames = 16` ⇒ `revealed = 60·f/15 = 4f`, always an integer, so
the `<= 1` tolerance is never exercised and per-dataset `int()` truncation never
accumulates. Simulated frame-by-frame output (plan's own helpers):

```
f=0  counts=[0,0,0]     idx=0 frac=0.000
f=5  counts=[20,0,0]    idx=0 frac=1.000
f=6  counts=[20,4,0]    idx=1 frac=0.158
f=10 counts=[20,20,0]   idx=1 frac=1.000
f=15 counts=[20,20,20]  idx=2 frac=1.000
indices monotonic: True   indices[0]==0: True   max==2: True
```

So `test_serial_schedule_is_exposed_so_callers_need_not_re_derive_it` and
`test_title_list_tracks_the_revealed_dataset` (`seen[0]=='first'`,
`index('second') < index('third')`) both hold — those two are sound. Use unequal dataset
lengths (e.g. 17/23/11) and a frame count that is not a divisor to make T4 meaningful.

### T5 — Task 7: nothing covers `order='serial'` titles on plotly, or serial titles with trails

Plotly ignores serial trails entirely (`plot.py:3755-3758`) and has no per-frame
`ax.set_title` path; the Step 7 docstring text advertises per-segment titles with no
backend caveat. Defect class "advertised support a downstream check rejects".

### T6 — Task 3: two tests assert only the absence of a warning, never that the plot succeeded

`test_no_warning_when_clouds_are_already_small` and
`test_explicit_morph_samples_is_respected_without_warning` discard the return value and
assert only on `caught`. Also, in `test_default_morph_caps_sampling_and_completes_quickly`
the `pytest.warns(...)` block wraps the *timed* region, so a warning-text mismatch masks
the timing signal the test exists to produce.

### T7 — Task 6: no test drives a `hue=` animation through `on_frame`

`_apply_multicolor_animation` **wraps** the animation's frame callback
(`plot.py:5104-5122`: "wrap the animation's frame callback so that, after each original
update runs, every collection is re-sliced"). So for hue plots `ani._func` is the wrapper,
and whether the `on_frame` hook sees pre- or post-multicolor artists in `ctx.artists` is
unspecified and untested.

### T8 — Task 2 Step 1 appends duplicate module-level setup

`tests/plot/test_matplotlib_backend_bugs.py` already exists; the plan's block re-adds
`import matplotlib / matplotlib.use("Agg") / import numpy / import pytest / import
hypertools as hyp` at the *bottom* of the file. Harmless but should be trimmed on append.

---

## Verified-correct claims (do not re-litigate)

* Baseline suite size: 2564 collected ⇔ plan's 2551 passed + 13 skipped.
* `matplotlib_backend.py:1602-1604` and `:2197-2199` are exactly where `linewidth` is
  popped; `plot.py:5150-5153` is exactly where it is read back. The pop→rcParams fallback
  bug is real (measured 1.5 vs the requested 0.5/0.5/5.0).
* `_resolve_animate_mode` has exactly **one** caller (`plot.py:3653`) and no test calls it
  directly, so the 2-tuple → 3-tuple change is a one-line unpack update.
* `plot.py:453-513` line range for `_resolve_animate_mode` is correct.
* `matplotlib_backend.py:1316-1318` is the real reveal formula, and the plan's
  `serial_reveal_counts` reproduces it exactly.
* `HyperAnimation` (`hyper_animation.py:45`) is a `tuple` subclass with **no** `__slots__`;
  instance attributes set in `__new__` work and survive. The plan's premise here is fine.
* `ani._func(f, *ani._args)` is a valid way to drive frames for `animate=True`,
  `'serial'` and `'morph'` (verified: morph `_save_count=24`, 5 fargs, all frames render).
* 4 of the 5 new gallery examples do monkeypatch `_func`
  (`examples/animate_conversation.py`, `animate_market_forecast.py`,
  `animate_morph_zoo.py`, `animate_weather_decades.py`; `animate_painting_embeddings.py`
  does not) — Task 6's motivation is accurate.
* `alpha=[...]` really does raise `TypeError: alpha must be numeric or None` today, and
  scalar `alpha=` really does reach every `Line2D` (static and animated).
