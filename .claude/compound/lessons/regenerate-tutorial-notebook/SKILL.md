---
name: regenerate-tutorial-notebook
description: Use when regenerating a docs/tutorials notebook from its example script in hypertools.
created: 2026-10-03
origin: notes kept in CLAUDE.md
---
1. Generate cells from the script's section markers with a scratchpad script; carry the
   install cell over byte-identical.
2. Never leave a HyperAnimation as a cell's last expression (its repr embeds an 89 KB video).
3. Execute with scripts/execute_tutorial.py (skips pip-install cells, disables HF progress bars).
4. Save the GIF at <=15 fps and about 300 frames (a 900-frame GIF is 13 MB).
5. Record the measured visible-output set in tests/test_examples_are_native.py and
   re-measure the budget once.
6. Commit script, notebook and GIF and the gate together.
Gallery pages for show=False examples need the HyperAnimation scraper in docs/conf.py.
