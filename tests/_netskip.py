# -*- coding: utf-8 -*-
"""Shared TRANSIENT-network classifier for tests that make real network calls.

Lives here rather than in a single test module because more than one test file
needs it. Extracted 2026-07-30 from `tests/test_load_sources.py`, which had the
only copy, after `tests/test_load_sklearn_seaborn.py` was found fetching
seaborn-data over the network with no guard at all.

**This is not a way to make failing tests pass.** `is_transient_network` is a
deliberately narrow predicate: timeouts, dropped connections, 5xx, and DNS
resolution failures -- the conditions where the host, not our code, is at
fault. It does NOT match assertion errors, key errors, or anything else that
indicates a real defect, and `skip_on_transient_network` re-raises everything
it does not classify. A genuine regression still fails the suite.

The suite deliberately uses three different strategies for network flakiness;
pick the one that fits rather than adding a fourth:

* **Retry in the library** -- best when we own the request. See
  `hypertools.io.sources._github_get_with_retry`, which retries transient
  gateway errors for the 538 loader (`tests/test_load_538_kaggle.py`).
* **Release-gated skip** -- best when a silent skip would hide a real
  regression at release time. See `tests/test_dataset_compat.py`, where
  `HYPERTOOLS_REQUIRE_*` turns every skip into a hard failure.
* **This predicate** -- best for a one-off live fetch through a third party's
  endpoint that we neither own nor can retry meaningfully.
"""

import contextlib

import pytest

# Hosted-dataset tests must not fail unrelated CI on a TRANSIENT network error
# (a Hugging Face ReadTimeout, a 5xx, a dropped connection) while still
# exercising the real load path when the host is reachable.
# (2026-07: a HF ReadTimeout on test_load_huggingface_dataset flaked one
# ubuntu-3.13 matrix cell. 2026-07-30: Google Sheets read-timed-out and Google
# Drive answered 500 in the same run, failing test_load_google_sheet_live,
# which passed again on re-run 2 minutes later.)
TRANSIENT_NETWORK = (
    'readtimeout', 'read timed out', 'timed out', 'timeout',
    'connectionerror', 'connection error', 'connection reset',
    'connection aborted', 'remotedisconnected', 'incompleteread',
    'max retries', 'service unavailable', 'temporarily unavailable',
    ' 503', ' 502', ' 504', 'temporary failure',
    # A 500 from the host is an upstream fault, exactly like the 502/503/504
    # above -- this predicate's own docstring already claimed "5xx". Matched by
    # the full requests phrase, NOT a bare ' 500', which would false-positive
    # on a real assertion like "shape 500 != 499".
    '500 server error', 'internal server error',
    # DNS-resolution failures. urllib3 raises NameResolutionError whose message
    # is "Failed to resolve '<host>'"; hyp.load() wraps it into its diagnostic.
    # Keep the CamelCase-lowered class name AND the message text (neither
    # contains the spaced "name resolution"/"failed to establish" above).
    'name resolution', 'nameresolutionerror', 'failed to resolve',
    'getaddrinfo',
    'network is unreachable', 'failed to establish',
)


def is_transient_network(text):
    """True if `text` reads like a TRANSIENT network error (timeout, dropped
    connection, 5xx, DNS-resolution failure) rather than a real defect. Pure
    predicate so it can be unit-tested without the skip machinery."""
    text = text.lower()
    return any(marker in text for marker in TRANSIENT_NETWORK)


@contextlib.contextmanager
def skip_on_transient_network(what):
    """Skip -- never pass -- when `what` fails for a transient network reason.

    Anything not classified as transient is re-raised unchanged, so a genuine
    defect still fails.
    """
    try:
        yield
    except Exception as e:            # re-raised below unless it is transient
        if is_transient_network(str(e)):
            pytest.skip(f'transient network error {what}: {e}')
        raise
