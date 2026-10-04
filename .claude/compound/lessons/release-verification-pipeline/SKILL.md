---
name: release-verification-pipeline
description: Use when running the full hypertools release verification (notebook re-execution, full pytest, sphinx -W gallery, example smoke gate).
created: 2026-10-03
origin: notes kept in CLAUDE.md
---
The pipeline runs about 50 minutes, past the Bash tool's 10-minute cap. Write it as a
scratchpad zsh script with step markers, launch it with nohup, and poll the log for the
DONE marker from run_in_background watchers (each at most 9.5 minutes). After any sphinx
build, `git checkout` the two autosummary stubs FrameContext and LSLStream, which are
regenerated with whitespace-only changes.
