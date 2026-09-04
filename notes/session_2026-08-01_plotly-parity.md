# Session 2026-08-01 (part 2) — plotly/matplotlib animation parity

Branch `dev-1.0`, commit `4877287c`, on top of `791ab508`. **Nothing pushed.**

Jeremy ruled the open conflict from part 1: *"plotly issue is not acceptable as is. we do need
parity with matplotlib, and both backends need to produce nice looking and viable
plots/animations."* So "additive only" governs modes, spellings and signatures — not "no rendered
pixel may ever move." Recorded in the plan's Global Constraints.

**Suite 2754 → 2769.** Sphinx `-W -E -a` 0 warnings. Packaging 49+6s. Live-source 41 both modes.

## What shipped

`matplotlib_backend._anim_window_bounds` **moved** to `trails.anim_window_bounds` — a module both
backends already imported — and both now call the same object, per dataset, per frame. The
diagnosis had recommended reimplementing the formula inside plotly; that was declined, because a
transcription is precisely what drifted. A shared callee cannot drift from itself.

Six divergences closed, all shipped in 1.0. Three were the diagnosed A/B/C; three were found
while fixing those:

| | defect | measured |
|-|-|-|
| A | missing `-1` in `start` | 90 of 120 frames off by one; ALSO a one-segment gap between a chemtrails trail and its head |
| B | one shared window from `max_len` | 5-row dataset drew NOTHING for 9 of 15 frames (60% of its own animation) |
| C | `end` floored at 2 vs 1 | precog frame 0: 12 vs 11 |
| D | frame count floored at 2 vs 1 | 1 vs 2 frames on a sub-frame request |
| E | matplotlib `'serial'`/`'spin'` frame count | **ZERO frames** on a sub-frame request; `.save()` raised `CalledProcessError` |
| F | matplotlib spin azimuth | 289.71° vs plotly 280.0° at fr=7/dur=2.5; overshot a `rotations=1` turn |
| G | plotly Play button speed | 5% slow at fr=3/dur=1.4; **2× fast** on a sub-frame request |

## The lesson: fixing a defect can create one

D was mine to introduce. `_add_animation` resolves `n_frames` ONCE, before branching by style, so
flooring it at 1 to match matplotlib landed on all four styles — while matplotlib floored only its
parallel/`'window'` path. I closed three divergences and opened two (`serial` 1 vs 0, `spin` 1 vs
0), then wrote a comment asserting an exactness that no longer held. **The review caught it, not
me.**

The repair was NOT to narrow my change. matplotlib asking `FuncAnimation` for zero frames is its
own bug — an animation that draws nothing, and the re-review found `.save()` crashes outright. So
every matplotlib frame-count site was floored (7 newly wrapped, 12 total). Flooring the
`FuncAnimation` count alone would have been a `ZeroDivisionError`: several updaters divide by their
own `total_frames`. Both had to move together.

## Two divergences in code nobody touched

F and G were found by reviewers sweeping ADJACENT code, not the diff. G is the sharper one: the
GIF/APNG export path in the same file already carried a comment saying the delay must be
`1000/frame_rate` and explicitly **"NOT 1000*duration/n_frames"** — and the Play button, 1500 lines
later, was doing exactly the forbidden thing. The rule was written down and violated in its own
file.

## A false claim in my own release note

The re-review's only blocker was prose, not code: I transcribed the two spin azimuths **backwards**
into both the CHANGELOG and the source comment, and inverted the direction — wrote "stopped short"
when the raw product had made it travel FURTHER (349.71° vs 340.0°). The prior review had them the
right way round; I inverted them in transcription. Verified by direct measurement before
correcting, rather than taking either reviewer's word.

Worth remembering: the code was verified six ways and the sentence describing it was still wrong.
Prose in a commit needs the same verification as the code it describes.

## Test quality

`tests/test_backend_window_parity.py`: 29 tests, **24 red at `791ab508`**. The 5 that pass there
are non-diverging controls (integer products, morph's own schedule, a numerically-coincident
window); each is labelled as a control beside a diverging sibling, so no future reader mistakes it
for regression coverage.

Two existing tests had to change. Neither was weakened:

- `test_plotly_window_exact_bounds_mid_animation` had transcribed the renderer's own arithmetic
  (`max(2, ...)` floors included) as its "expected" value, so it pinned whatever the code did — it
  passed while plotly ran a point short. Now derives the expectation from the public knobs:
  `focused` seconds × `frame_rate`, + 1.
- `test_plotly_chemtrails_past_trail` asserted `len(trail) > len(head)` — an inequality that held
  only by the coincidental one-point margin the bug created. Now pins the shared vertex, which an
  off-by-one cannot satisfy. (It failed pre-fix on a *different vertex entirely* — that gap was
  visible output, not just a count.)

## Verification performed

- Real renders throughout; no mocks. Both reviewers independently confirmed the red/green split.
- Move proven behavior-preserving for matplotlib over **143,472** argument tuples (AST-identical
  too).
- **408** head/trail configs, **96** serial configs, **90** frame-count combos, **48** window
  configs — 0 divergences.
- Spin azimuth: max gap **0.000000000°** over 10 rate/duration pairs × 7 `rotations` values × 4
  azimuths × 5 frames.
- Visual: real PNGs both backends, pre vs post. The 5-row dataset is **absent** pre-fix and drawn
  post — the numbers understated it; the plot was simply missing data.

## Still open — needs Jeremy

1. **Plans 3 and 4 carry unaddressed FATAL findings** (`notes/audit/review_plan3_v2_recheck.md`,
   `notes/audit/review_plan4_examples_and_tutorials.md`). Neither is implementation-ready. Plan 3's
   prescribed code was ALSO retargeted here — it named `matplotlib_backend._anim_window_bounds` as
   literal executable text, which this commit deletes.
2. **Nothing pushed.** 44 commits ahead of `origin/dev-1.0`.
3. **Restart for OMC 4.15.7** — the loaded 4.2.15 hook emitted false "Edit operation failed"
   notices throughout, every one on a call that succeeded.
