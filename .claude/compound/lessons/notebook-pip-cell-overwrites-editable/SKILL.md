---
name: notebook-pip-cell-overwrites-editable
description: Use when executing a docs/tutorials notebook locally in hypertools.
created: 2026-10-03
origin: notes kept in CLAUDE.md
---
A launch notebook's Colab cell `%pip install ... git+...@dev-1.0` overwrites the venv's
editable hypertools with the stale REMOTE branch mid-run (symptom: "48 dimensions ...
static plots support at most 2"). Execute with scripts/execute_tutorial.py, which tags
'pip install' cells skip-execution in memory. After any other notebook run, check that
`pip show hypertools` says Editable, and run `pip install -e .[dev]` if not.
