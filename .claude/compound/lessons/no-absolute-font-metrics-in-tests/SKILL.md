---
name: no-absolute-font-metrics-in-tests
description: Use when writing a hypertools test that asserts text or figure measurements.
created: 2026-10-03
origin: notes kept in CLAUDE.md
---
Never assert absolute font-metric numbers (probe heights, figure inches, pixel rows). CI's
matplotlib (3.11.1) hints text differently from the local 3.10.8, so such tests pass
locally and fail on every CI job. Assert against the library's own probe on the same axes,
or use relative tolerances.
