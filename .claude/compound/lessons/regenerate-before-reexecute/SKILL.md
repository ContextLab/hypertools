---
name: regenerate-before-reexecute
description: Use when changing SPECS (dpi, prose) for the generated tutorial notebooks in hypertools.
created: 2026-10-03
origin: notes kept in CLAUDE.md
---
Tutorial notebooks are GENERATED (scripts/generate_tutorial_notebook.py) and then executed
(scripts/execute_tutorial.py). After changing SPECS, ALWAYS regenerate before
re-executing; otherwise a 10-minute re-execution writes the stale value. Video size is
CRF-bound (not a fixed bitrate), so dpi does not trade against size.
