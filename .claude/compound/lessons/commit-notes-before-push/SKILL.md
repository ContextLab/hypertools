---
name: commit-notes-before-push
description: Use when about to push a branch that triggers hosted CI and a session-note or memory update is still uncommitted.
created: 2026-10-03
origin: notes kept in CLAUDE.md
---
Commit the session-note or memory update BEFORE the push. A note committed right after the
push moves the head to a note-only commit, and every CI cycle has to be re-run or cancelled.
