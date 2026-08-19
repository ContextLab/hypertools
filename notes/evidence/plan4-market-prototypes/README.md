# Market composition prototypes (Plan 4, Task 2)

Static evidence for a design decision, tracked because a visual choice
cannot be reviewed from a sentence saying one image is legible.

| file | what it shows |
|-|-|
| `PROTO_A2_single_panel.png` | one fixed 2-D panel, six sector leaves + the market mean |
| `PROTO_B2_small_multiples.png` | one small panel per sector, market mean repeated as a reference |
| `make_prototypes.py` | regenerates both, ~3 s |

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

**A2 tangles and its labels collide.** B2 is legible. Not yet reviewed at
documentation display width, which is the size that decides it.

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
   asserts the result — all six panels share one limit tuple exactly.

That highlight IS the hierarchy's arithmetic (the mean of the weight rows),
which is what makes it a demonstration rather than a workaround.
