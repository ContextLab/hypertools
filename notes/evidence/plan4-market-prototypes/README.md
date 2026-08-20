# Market composition prototypes (Plan 4, Task 2)

Static evidence for a design decision, tracked because a visual choice
cannot be reviewed from a sentence saying one image is legible.

| file | what it shows |
|-|-|
| `PROTO_A2_single_panel.png` | one fixed 2-D panel, six sector leaves + the market mean |
| `PROTO_B2_small_multiples.png` | one small panel per sector, market mean repeated as a reference |
| `PROTO_C_small_multiples.png` | B2 with the vertical compression fixed, rendered at **736 px** — the actual documentation width |
| `make_prototypes.py` | regenerates all three, ~15 s |

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

So the open question for review is not "is C tuned right" but **what the
panels are for**: peers as context with no visible mean (C as it stands),
or the mean as a reference with the peers dropped or darkened — which, per
review round 11, would need a dark component shared by every leaf and would
muddy the leaf colours. That trade cannot be settled by the arithmetic; it
is a call about what the example is showing.

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
