---
name: lsl-test-unique-stream-type
description: Use when writing or debugging a hypertools LSL test that resolves a stream by type.
created: 2026-10-03
origin: notes kept in CLAUDE.md
---
A test that resolves by type='EEG' fails whenever ANOTHER process advertises an idle EEG
outlet (a notebook kernel running the tutorial's synthetic outlet), because lsl_stream()
takes the first match. Give a test outlet a unique stream TYPE, as with the unique names,
and check pylsl.resolve_streams() for foreign outlets before blaming the library.
