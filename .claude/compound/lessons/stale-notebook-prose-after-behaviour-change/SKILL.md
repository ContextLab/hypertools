---
name: stale-notebook-prose-after-behaviour-change
description: Use when a hypertools library behaviour changes (new error text, a warning removed, a new return type).
created: 2026-10-03
origin: notes kept in CLAUDE.md
---
Grep docs/tutorials/*.ipynb AND the gitignored notes/colab/*.ipynb (the Colab feature
tour) for prose or stored outputs that demonstrate the OLD behaviour, and re-execute those
notebooks. Examples of what goes stale: a tutorial teaching hyp.load(df) as a TypeError
after load gained passthrough, a stored glyph warning, a stored liblsl ERR line, a tour
note saying "pip install predict-hf first". Before editing an install claim, verify it in
a FRESH venv (python -m venv, pip install ., then the call).
