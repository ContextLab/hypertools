---
name: global-setting-as-context-manager
description: Use when changing hyp.set_autoinstall or adding another process-global setting that is both a direct call and a context manager.
created: 2026-10-03
origin: notes kept in CLAUDE.md
---
Model it as LIVE handle records (weakrefs, in construction order) plus a BASELINE folded
in by the weakref callback of a handle that dies unentered; take the newer of the top
record and the baseline by call order; mark exited records finished. The failure modes are
a restore-order race, retained handles, and a construct-then-enter race across threads.
Test overlapping blocks across threads, construct-then-enter interleaving, 100k discarded
direct calls (tracemalloc), and a block handle dying after exit.
