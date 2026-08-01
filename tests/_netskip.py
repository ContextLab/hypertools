# -*- coding: utf-8 -*-
"""Shared TRANSIENT-network classifier for tests that make real network calls.

Lives here rather than in a single test module because more than one test file
needs it. Extracted 2026-07-30 from `tests/test_load_sources.py`, which had the
only copy, after `tests/test_load_sklearn_seaborn.py` was found fetching
seaborn-data over the network with no guard at all.

**This is not a way to make failing tests pass.** A genuine regression must
still fail the suite, and the first version of this module did not guarantee
that: it lowercased the whole exception string and matched substrings, one of
which was a bare ``'timeout'``. Review round 13 (2026-07-31) reproduced three
false "transient" verdicts against it --

* ``ValueError: timeout must be positive``
* ``AssertionError: timeout metadata missing``
* an aggregate holding BOTH ``KeyError: parser regression`` (the resolver the
  test actually wanted) and ``ReadTimeout: timed out`` (a later fallback)

-- the third being the dangerous one, because :func:`hypertools.io.sources.
load_source` aggregates every attempted resolver into ONE ``HypertoolsIOError``
(``sources.py:664-670``), so a real defect in the intended resolver routinely
shares a message with an unrelated transient failure further down the chain.

So classification is now STRUCTURAL, in this order:

1. **Exception type.** The raised exception and its whole ``__cause__`` /
   ``__context__`` chain are matched against :data:`TRANSIENT_TYPES` by class
   name across each MRO. Prose can never reach this path.
2. **HTTP status.** ``HTTPError``-family exceptions are transient only for
   **5xx**. A 404 on a dataset URL is a REAL regression -- the URL moved -- and
   must fail, which the old "5xx" substring list could not express.
3. **The wrapped aggregate, and only then, by text.** ``load_source`` raises its
   digest OUTSIDE any ``except`` block, so there is no ``__cause__`` to walk and
   the per-resolver detail exists only as text. Each attempt line is classified
   independently and **any defect-shaped line vetoes the whole message**: one
   transient fallback can no longer excuse a real failure beside it.

A word like "timeout" appearing in prose is now inert; only a ``Name:``-shaped
exception token, a 5xx status, or an unambiguous multi-word network phrase
carries a verdict.

Set ``HYPERTOOLS_REQUIRE_LIVE_SOURCES=1`` to turn every skip here into a hard
failure -- see :func:`live_sources_required`. The ``live-source-gate`` CI job
sets it, so these integrations cannot skip indefinitely without a red gate.

The suite deliberately uses three different strategies for network flakiness;
pick the one that fits rather than adding a fourth:

* **Retry in the library** -- best when we own the request. See
  `hypertools.io.sources._github_get_with_retry`, which retries transient
  gateway errors for the 538 loader (`tests/test_load_538_kaggle.py`).
* **Release-gated skip** -- best when a silent skip would hide a real
  regression at release time. See `tests/test_dataset_compat.py`, where
  `HYPERTOOLS_REQUIRE_*` turns every skip into a hard failure.
* **This predicate** -- best for a one-off live fetch through a third party's
  endpoint that we neither own nor can retry meaningfully. It carries a strict
  mode of its own (above), so it is a superset of the second strategy.
"""

import contextlib
import os
import re

import pytest

# Exception TYPE NAMES that mean the HOST, not our code, is at fault: timeouts,
# dropped/refused connections, and DNS-resolution failures. Matched against the
# full MRO of a live exception, and against `Name:`-shaped tokens inside a
# wrapped diagnostic -- never against free prose.
# (2026-07: a HF ReadTimeout on test_load_huggingface_dataset flaked one
# ubuntu-3.13 matrix cell. 2026-07-30: Google Sheets read-timed-out and Google
# Drive answered 500 in the same run, failing test_load_google_sheet_live,
# which passed again on re-run 2 minutes later.)
TRANSIENT_TYPES = frozenset({
    # requests
    'Timeout', 'ConnectTimeout', 'ReadTimeout', 'ConnectionError',
    'ChunkedEncodingError', 'ContentDecodingError', 'ProxyError',
    'RetryError',
    # urllib3
    'TimeoutError', 'ConnectTimeoutError', 'ReadTimeoutError',
    'NewConnectionError', 'NameResolutionError', 'MaxRetryError',
    'ProtocolError',
    # stdlib: socket / http.client / urllib
    'ConnectionResetError', 'ConnectionAbortedError', 'ConnectionRefusedError',
    'RemoteDisconnected', 'IncompleteRead', 'URLError', 'gaierror',
})

# HTTPError says nothing on its own -- 5xx is the host's fault, 4xx is ours
# (a moved dataset URL is a REAL regression and must fail, not skip).
STATUS_DETERMINED_TYPES = frozenset({'HTTPError', 'HTTPStatusError'})

# Carry no verdict either way: the aggregate's own wrapper class, and bases too
# generic to mean anything. Named so they are not mistaken for defect evidence.
NEUTRAL_TYPES = frozenset({
    'HypertoolsIOError', 'Exception', 'BaseException', 'OSError', 'IOError',
})

# Classic programming-error types. Wherever one of these is raised it means OUR
# code broke, so it vetoes the chain walk even when a transient error sits
# further down `__context__` -- an exception raised INSIDE an `except Timeout:`
# block would otherwise inherit that timeout's verdict and skip.
DEFECT_TYPES = frozenset({
    'AssertionError', 'KeyError', 'IndexError', 'AttributeError', 'TypeError',
    'ValueError', 'NameError', 'UnboundLocalError', 'ZeroDivisionError',
    'ImportError', 'ModuleNotFoundError', 'NotImplementedError',
})

# Unambiguous MULTI-WORD network phrases, for hosts that report an outage in
# prose without a Python type name. Deliberately excludes the bare word
# 'timeout', which is also a keyword argument and ordinary English.
TRANSIENT_PHRASES = (
    'read timed out', 'connection timed out', 'operation timed out',
    'timed out', 'connection reset', 'connection aborted',
    'connection refused', 'max retries', 'service unavailable',
    'temporarily unavailable', 'temporary failure', 'internal server error',
    'bad gateway', 'gateway time-out', 'gateway timeout',
    'name resolution', 'failed to resolve', 'getaddrinfo',
    'nodename nor servname', 'network is unreachable',
    'failed to establish',
)

# "500 Server Error" / "503 Server Error" as requests spells it. Matched with a
# digit pattern plus the words, never a bare ' 500', which would false-positive
# on a real assertion like "shape 500 != 499".
_SERVER_5XX_RE = re.compile(r'\b5\d\d server error\b')

# An identifier immediately followed by ':' -- the shape `load_source` uses for
# every attempt line (`sources.py:592-662`: f'<resolver>: {type(e).__name__}: ...').
_TOKEN_RE = re.compile(r'\b([A-Za-z_][A-Za-z0-9_]*)(?=:)')


def live_sources_required():
    """True when ``HYPERTOOLS_REQUIRE_LIVE_SOURCES=1``, which makes
    :func:`skip_on_transient_network` re-raise instead of skipping.

    Read from the environment on every call rather than captured at import, so
    a test can set it with ``monkeypatch.setenv`` and exercise the real branch.
    """
    return os.environ.get('HYPERTOOLS_REQUIRE_LIVE_SOURCES') == '1'


def _chain(exc):
    """The exception and everything it was raised from/during, once each."""
    seen = set()
    while exc is not None and id(exc) not in seen:
        seen.add(id(exc))
        yield exc
        exc = exc.__cause__ or exc.__context__


def _http_status(exc):
    """The HTTP status carried by an HTTPError-family exception, or None.
    Covers requests (``.response.status_code``) and urllib (``.code``)."""
    code = getattr(getattr(exc, 'response', None), 'status_code', None)
    if code is None:
        code = getattr(exc, 'code', None)
    return code if isinstance(code, int) else None


def _type_names(exc):
    """Every class name in the exception's MRO."""
    return {cls.__name__ for cls in type(exc).__mro__}


def _exception_is_transient(exc):
    """Structural verdict on a LIVE exception: type names across the whole
    ``__cause__``/``__context__`` chain, plus HTTP status. Text is not consulted.
    """
    if _type_names(exc) & DEFECT_TYPES:
        return False
    for e in _chain(exc):
        names = _type_names(e)
        if names & TRANSIENT_TYPES:
            return True
        status = _http_status(e)
        if names & STATUS_DETERMINED_TYPES and status is not None:
            if 500 <= status < 600:
                return True
    return False


def _is_exception_token(token):
    """True if `token` names an exception class rather than part of a resolver
    label ('Google Sheets: ...' must not read 'Sheets' as evidence)."""
    if token in TRANSIENT_TYPES or token in STATUS_DETERMINED_TYPES \
            or token in NEUTRAL_TYPES:
        return True
    # 'Error'/'Exception' alone are words, not type names -- require a prefix.
    return (len(token) > len('Error') and token.endswith('Error')) or \
        (len(token) > len('Exception') and token.endswith('Exception'))


def _classify_line(line):
    """One attempt line -> 'transient', 'defect', or 'neutral'.

    A DEFECT verdict wins over any phrase on the same line: that is what stops
    ``AssertionError: request timed out`` reading as an outage. Only a transient
    exception TYPE or a 5xx status can outrank a named exception.
    """
    lowered = line.lower()
    has_5xx = bool(_SERVER_5XX_RE.search(lowered))
    tokens = [t for t in _TOKEN_RE.findall(line) if _is_exception_token(t)]

    if any(t in TRANSIENT_TYPES for t in tokens):
        return 'transient'
    if any(t in STATUS_DETERMINED_TYPES for t in tokens):
        # 5xx is the host; anything else (404, 403, ...) is a real regression.
        return 'transient' if has_5xx else 'defect'
    if any(t not in NEUTRAL_TYPES for t in tokens):
        return 'defect'
    if has_5xx or any(p in lowered for p in TRANSIENT_PHRASES):
        return 'transient'
    return 'neutral'


def _message_is_transient(text):
    """Verdict on a wrapped diagnostic, line by line.

    ANY defect-shaped line vetoes the whole message. `load_source` lists every
    resolver it tried in one error (`sources.py:664-670`), so a real failure in
    the resolver the test wanted routinely sits beside a transient failure in an
    unrelated fallback -- and must not be excused by it.
    """
    verdicts = [_classify_line(line) for line in text.splitlines() if line.strip()]
    if 'defect' in verdicts:
        return False
    return 'transient' in verdicts


def is_transient_network(exc_or_text):
    """True if this reads like a TRANSIENT network error (timeout, dropped
    connection, 5xx, DNS-resolution failure) rather than a real defect.

    Accepts an EXCEPTION (preferred -- its type chain and HTTP status are
    checked structurally, so prose cannot mislead it) or a string (the wrapped
    aggregate case, where the per-resolver detail exists only as text). Pure
    predicate so it can be unit-tested without the skip machinery.
    """
    if isinstance(exc_or_text, BaseException):
        if _exception_is_transient(exc_or_text):
            return True
        # No structural verdict. Fall back to the text of the whole chain,
        # each entry prefixed with its own type name so the message analyzer
        # sees the same 'Name: message' shape `load_source` writes -- without
        # that prefix, ValueError('request timed out') would read as an outage.
        text = '\n'.join(f'{type(e).__name__}: {e}' for e in _chain(exc_or_text))
    else:
        text = str(exc_or_text)
    return _message_is_transient(text)


@contextlib.contextmanager
def skip_on_transient_network(what):
    """Skip -- never pass -- when `what` fails for a transient network reason.

    Anything not classified as transient is re-raised unchanged, so a genuine
    defect still fails. Under ``HYPERTOOLS_REQUIRE_LIVE_SOURCES=1`` nothing is
    skipped at all: the outage itself becomes the failure, so a live source
    that has been unreachable for a week cannot hide behind a green suite.
    """
    try:
        yield
    except Exception as e:            # re-raised below unless it is transient
        if not live_sources_required() and is_transient_network(e):
            pytest.skip(f'transient network error {what}: {e}')
        raise
