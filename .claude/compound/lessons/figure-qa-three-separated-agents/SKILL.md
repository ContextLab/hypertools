---
name: figure-qa-three-separated-agents
description: Use when reviewing the figures of a hypertools notebook or the feature tour for correctness.
created: 2026-10-03
origin: notes kept in CLAUDE.md
---
Use three SEPARATED agents: one writes each figure's EXPECTED appearance from the notebook
code, prose and docstrings without seeing images; one describes the OBSERVED renders
without seeing code; one adjudicates with probes. This finds gaps solo eyeballing misses
(ax= palette ignored, z-label outside the tight bbox, a 2-D frame on the data, faded
recoloured forecasts). Render plotly outputs from their notebook JSON with kaleido (the
scratchpad review/manifest and tour_out/extract.py pattern), and re-run only the changed
figures each round.
