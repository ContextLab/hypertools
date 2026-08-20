# Market composition prototypes (Plan 4, Task 2)

Static evidence for a design decision, tracked because a visual choice
cannot be reviewed from a sentence saying one image is legible.

| file | what it shows |
|-|-|
| `PROTO_A2_single_panel.png` | one fixed 2-D panel, six sector leaves + the market mean |
| `PROTO_B2_small_multiples.png` | one small panel per sector, market mean repeated as a reference |
| `PROTO_C_small_multiples.png` | B2 with the vertical compression fixed, rendered at **736 px** — the actual documentation width |
| `PROTO_D_small_multiples.png` | C, plus a **dark hierarchy mean** |
| `PROTO_D2_small_multiples.png` | D at 21 months, mean restrained — round 12's prescription, followed |
| `PROTO_D3_small_multiples.png` | the knot diagnosed: 5 years at 6-month strides — **static candidate** |
| `PROTO_E_animated.gif`, `PROTO_E_midframe.png`, `PROTO_E_lastframe.png` | six panels, **one call, one animation** — **SELECTED** (round 13) |
| `make_prototypes.py` | regenerates all of them, ~90 s (E renders 90 frames) |

### The knot was in the data, not the styling

Round 12 asked for a shorter window and a restrained mean. Done (D2) — and the mean was still a
knot. Measured with `roughness = total turning / total drawn length`:

| | mean | one sector |
|-|-|-|
| 30 months, monthly | 8.59 | 2.94 |
| 21 months, monthly | 8.59 | 2.94 |
| + `Smooth(kernel_width=5)` | **10.10** | 3.23 |
| 60 months, 6-monthly | **4.18** | 2.74 |

Smoothing makes it *worse*: it removes length faster than it removes turning. The cause is that
over the same span the market mean covers **0.46×** the ground of a single sector while turning as
often — averaging six sectors cancels *direction*, not noise. Only fewer, wider strides fix it, and
they show more history rather than less. That is D3.

Run it from anywhere:

```
.venv/bin/python notes/evidence/plan4-market-prototypes/make_prototypes.py
```

It fetches live closes and degrades to a synthetic series offline, so the
data moves with the run date; the composition does not.

## What they were made to test

The maintainer's direction after the 3-D forecast artifact was discarded:
"a hierarchy-native 2-D or small-multiple composition: six sector leaf
trajectories; one cross-sector mean; monthly observations; cumulative
return versus drawdown ...; fixed sector colors; a short trailing window
rather than full-history chemtrails; either no forecast or a forecast shown
purely as API mechanics."

**A2 tangles and its labels collide.** B2 is legible at full PNG size, and
review round 11 rejected it for reasons that only show up smaller: the
traces sat in a thin horizontal band, most of each panel was empty, the
repeated y-labels overlapped, and the title called the market mean "dark"
when the hierarchy blends it pale.

**C answers those, at 736 px, and it is not approvable yet either.** What C
changes, and what it measured:

* **The thin band was a display-gain problem, and it is now fixed.** `hyp.
  plot` centres each dimension separately but applies ONE gain to all of
  them (probe: x in [0, 100] and y in [0, 1] are drawn as x in [-1, 1] and
  y in [-0.01, 0.01]). Cumulative return and drawdown have half-ranges of
  0.456 and 0.129 over the complete frame — a factor of 3.5 — so drawdown
  was drawn into a third of the height it deserved. C gives each MEASURE
  one display gain, computed once over the complete frame and applied to
  every panel, so it cannot make panels incomparable. Drawn spans are now
  0.33–1.18 in x and 0.46–1.45 in y, out of the 2.0 the box allows.
* **One figure-level x and y label**, one short title, smaller endpoint
  markers and arrowheads (what reads on a 1500 px prototype is a blob at
  736 px), and peers paled to `#ededed` so the focused sector dominates.
* **The market mean cannot be made a visible reference this way, and that
  is a composition decision rather than a tuning problem.** The parent is
  the mean of its children by construction, so with one focused leaf and
  five pale ones it always lands *inside* the peer greys. Measured at three
  palette settings, the parent-vs-peer luminance gap is 0.094 / 0.088 /
  0.062 — and twice the leaf linewidth (measured: 2.0 vs 1.0) does not
  recover a 0.09 gap at this size. C therefore claims only what it
  delivers: its title no longer says "against the market mean", and the
  mean is findable by its neutral endpoint marker, not by its path.

### D — and why C's constraint was never going to yield

Review round 11 asked whether a dark mean could be tuned into existence.
It cannot, and the reason is arithmetic rather than aesthetic:

> With `hue_mode='mixture'`, the parent's colour is the **mean** of its
> children's colours. A mean lies in the convex hull of what it averages.
> **So the parent can never be darker than its darkest leaf.** Five pale
> peers force a pale mean, at every palette setting.

Round 12 therefore relaxed Plan v5 criterion 3: the hierarchy must stay
native in **discovery** (leaves from the column MultiIndex), **geometry**
(the parent is the mean of its children) and **style** (its heavier line),
but the parent's **colour** may be assigned directly. D does that, and the
mean goes from a 0.088 luminance gap to **0.76**.

### The API gap D documents

There is no spelling of `hyp.plot` today that gives per-leaf colours AND an
independently coloured parent. Measured:

| call | result |
|-|-|
| 3-level frame + `palette=[...]` | 6 leaves + 1 parent, **all one colour** (leaves α 0.7 lw 1.0, parent α 1.0 lw 2.0) |
| 2-level frame + `palette=[...]` | 6 leaves, each its own colour, **and no parent at all** |
| matrix `hue` + `hue_mode='mixture'` | arbitrary per-leaf colours, parent forced to their mean |
| `color=`/`colors=` with a column MultiIndex | ignored, with a warning: colour comes from the top-level index |
| `hue=` one label per trace | every trace takes `palette[0]` — a per-trace value has no range to map |

So D draws the **same complete frame twice**: once with matrix hue for the
leaves, once with a dark single-colour palette for the parent, hiding the
second call's leaves. Both calls receive identical input, so they normalize
identically — asserted point-for-point, and the two parent paths agree to
**0.00e+00**. Nothing is drawn by hand and the dark line is the library's
own parent trace in the library's own hierarchy style.

That works, but a shipped example needing two calls and a visibility toggle
is an argument for closing the gap rather than a design to copy. **The
minimal fix**: let a hierarchy assign one palette colour per child group,
with the parent's colour separately settable. That is a smaller and better
defined change than any of the alternatives — and it is library work Plan 4
does not own, so it needs the maintainer's call.

## Two measured library limits the prototypes work around

1. `hyp.plot` draws in a NORMALIZED unit box (`xlim == [-1.1, 1.1]`
   whatever the data) and removes ticks by design. `xlabel=`/`ylabel=` are
   native and DO render, so an axis can say what it IS but never what it
   equals — "cumulative return versus drawdown" reads as a direction, not
   a number.
2. Each call normalizes ITS OWN inputs, so N calls give N private scales.
   A first attempt at small multiples was invalid for this reason: the
   shared market curve came out a different shape in every panel. Every
   panel here is passed the WHOLE frame and differentiated only by hue,
   using `hue_mode='mixture'` with a 7th near-white palette entry so the
   focused sector takes its primary and the peers go pale. The script
   asserts the result — all six panels share one limit tuple exactly, in
   B2 and in C.

That highlight IS the hierarchy's arithmetic (the mean of the weight rows),
which is what makes it a demonstration rather than a workaround.

### E — panels in the data, not in matplotlib

Measured, in order:

1. `hyp.plot` **ignores `ax=` when `animate=` is set** and builds its own figure — so animated
   panels cannot be composed through `ax=`, and the two-call dark-mean workaround is impossible
   in an animated plot.
2. `HyperAnimation.draw_frame(i)` **is public** and documented as the supported way to drive an
   animation, so N animations can be stepped in lockstep — but they live on N figures, and
   there is nothing to save them into.
3. So the panels move into the **data**: each panel group is translated into its own region of one
   shared coordinate box. Six panels become six column groups of one frame — **one call, one
   animation, an ordinary `.save()`** — and they share one normalization by construction.

With the panels in the data, colour-by-top-level-group stops being an obstacle and becomes the
right behaviour: each panel is one sector, its leaves are that sector's stocks, and its parent is
that sector's mean, automatically computed and automatically heavier. **No hue matrix, no second
call, no hidden artists.** Measured: 6 colours, `[1.0, 1.0, 1.0, 1.0, 2.0]` linewidths per panel,
90 frames, 0.87 MB.

One thing the animated backend does that a multi-panel example must handle: it **draws a frame box
as a patch on the axes** (not as the spines), whose left and right edges cross the figure as two
vertical rules once the view is cropped to the panel grid. `frame_kwargs={'visible': False}` is the
documented knob for it.

**Correction.** This file previously also claimed the backend "re-applies its axis styling every
frame". That was **wrong** — an inference from those two rules surviving a `save()`, which the
frame box explains by itself. Measured: limits, spines and patches all survive `draw_frame` *and*
`save`. `on_frame` is used here for the one thing that genuinely must change per frame, the dot
riding the head of each sector mean.

The grid geometry is recovered from the affine transform `hyp.plot` applied, read off a **static**
draw of the same frame: an animated last frame stops a hair short of the full path, which biased
the two dimensions differently and made one gain look 7% larger than the other. Off the static
draw, the two gains agree exactly (0.2851 / 0.2851) — one gain, per-dimension centring, confirmed.


### E, as selected (round 13)

Refinements from finding 7, all display-side: parent means darkened to 62% of the panel colour and
drawn at **2.6** against the leaves' **0.9**; a dot rides the head of each mean through the public
`on_frame` hook, so the direction of the reveal reads in a still as well as in motion; leaves at
alpha 0.55; and a caption the tiling owes the reader — *"same cumulative-return (x) and drawdown
(y) scale in every panel; panel positions are layout only"*. The **mid-frame** is kept as evidence
alongside the last frame: with the reveal in progress the means are the most legible thing in the
figure, which the finished frame understates.

What the plumbing does now, after finding 4:

- panel identity comes from **the labels the library assigns** (each group's parent carries its
  group name) plus draw order — never from colour, which two panels could share;
- that attribution is **proven, not trusted**: each parent must equal the mean of the leaves
  attributed to it (`4.44e-16`, against `1.62e+00` for a deliberately mis-attributed control);
- the affine transform is recovered only to **place annotations** in data terms — it takes no part
  in building the visualization, which is entirely the one `hyp.plot` call;
- the library's frame box is hidden through the documented `frame_kwargs`, not hunted down as an
  unexplained patch.
