#!/usr/bin/env python
"""Lab Streaming Layer (LSL) input support (GH #130).

`lsl_stream()` resolves a live LSL stream -- e.g. an EEG amplifier, eye
tracker, or any other application publishing via a `pylsl.StreamOutlet`
somewhere on the local network -- and wraps it as a plain Python iterator
of per-sample numeric vectors. That is exactly the shape
`hypertools.io.streaming.is_stream`/`row_to_vector` already expect from any
generator (issue #101), so an LSL stream plugs straight into the existing
streaming machinery with no extra glue::

    hyp.plot(hyp.io.lsl_stream(type='EEG'), stream_init=200, stream_chunk=20)

`pylsl` (https://github.com/labstreaminglayer/pylsl) is an optional
dependency -- it wraps the native `liblsl` library used by essentially
every LSL-speaking acquisition device/app
(https://labstreaminglayer.org) -- the `[lsl]` extra, installed on
first use.

Teardown
--------
The iterator `lsl_stream()` returns is an :class:`LSLStream`. It owns the
`pylsl.StreamInlet` it pulls from, and releases it deterministically the
moment ANY of these happens:

* ``stream.close()`` is called (idempotent; a closed stream raises
  ``StopIteration``, like a closed generator);
* the ``with hyp.io.lsl_stream(...) as stream:`` block exits;
* the stream aborts because the source went silent
  (:class:`~hypertools.core.exceptions.HypertoolsIOError`);
* the last reference to the stream is dropped (garbage collection), or
  the interpreter/kernel exits with the stream still open -- via
  :func:`weakref.finalize`, whose at-exit hook runs BEFORE module globals
  (and with them any `pylsl.StreamOutlet` the script created) are torn
  down.

Releasing means ``close_stream()`` followed by DESTROYING the inlet
(dropping the last reference, so pylsl's ``StreamInlet.__del__`` runs
``lsl_destroy_inlet``). Both steps matter: measured against liblsl 1.17.7,
``close_stream()`` only sets a flag and cancels the inlet's sockets, and its
data thread keeps logging ``data_receiver.cpp:344 ERR| Stream transmission
broke off (Input stream error.); re-connecting...`` if the outlet
disappears first -- liblsl suppresses that message only once the inlet is
shut down, which is what destroying it does. Before 1.1 the inlet lived in
a generator's closure and was ``close_stream()``ed only when the generator
was closed, so a notebook that simply moved on (or a script that exited)
left the inlet alive for liblsl to complain about at kernel/interpreter
teardown.
"""

import threading
import warnings
import weakref


def _import_pylsl():
    from .._shared.lazy_import import lazy_import
    return lazy_import('pylsl', purpose='lsl_stream() (Lab Streaming Layer)')   # installs [lsl] on demand


def _ambiguity_caveat(any_stream):
    """The "...and here is what to do about it" tail of the multi-match
    warning, which differs by resolution path because the two pylsl calls
    behave differently (both measured against pylsl 1.18.2 / liblsl 1.17.7):

    * `resolve_byprop(prop, value, minimum=1, timeout=...)` returns as soon
      as `minimum` streams match -- 0.00s for a live outlet at the default
      `minimum=1`, 1.07s for `minimum=2` -- so its count really is
      best-effort, and `minimum=` is the fix.
    * `resolve_streams(wait_time=...)` takes NO `minimum` argument at all;
      it always waits the full `wait_time` and returns everything it heard
      (0.5s -> 2 outlets, 1.0s and 3.0s -> all 5, on a machine with five).
      So its count is not best-effort in that sense -- but a short
      `timeout=` under-reports, which is why the advice there is to raise
      `timeout=`. Advising `minimum=` on this path would be advising a
      `TypeError`.
    """
    if any_stream:
        return ('(This enumerates every outlet heard within timeout= '
                'seconds; liblsl warns that a short wait returns only a '
                'subset of those present, so raise timeout= if a stream you '
                'expect is missing. minimum= is not accepted on this path -- '
                'pylsl.resolve_streams() takes no such argument.)')
    return ('(This check is best-effort: resolution returns as soon as at '
            'least one stream matches, so a matching outlet that announces '
            'itself later is not detected -- pass minimum=2 to force '
            'ambiguity detection, at the cost of waiting the full timeout '
            'when only one stream exists.)')


def lsl_stream(name=None, type=None, timeout=10.0, **resolve_kwargs):
    """Resolve a live Lab Streaming Layer (LSL) stream and return it as a
    plain Python iterator of per-sample numeric vectors, compatible with
    `hypertools.io.streaming.is_stream`/`row_to_vector` -- so the result
    can be passed directly to `hyp.plot(..., stream_init=..., stream_chunk=
    ...)`.

    Parameters
    ----------
    name : str, optional
        Resolve the stream by its LSL ``name`` property (via
        ``pylsl.resolve_byprop('name', name, ...)``). Takes precedence
        over `type` when both are given.
    type : str, optional
        Resolve the stream by its LSL ``type`` property (e.g. ``'EEG'``
        or ``'Gaze'``), via ``pylsl.resolve_byprop('type', type, ...)``.
        Used only when `name` is not given. Only numeric channel formats
        are supported: string-typed streams (e.g. marker streams with
        ``channel_format='string'``) are rejected with a clear error,
        since hypertools' streaming machinery consumes numeric vectors.
    timeout : float
        Seconds to wait for a matching stream to appear on the network
        before giving up (default: 10.0). When neither `name` nor `type`
        is given, this is also the wait time used to resolve ANY
        available stream (``pylsl.resolve_streams``). The same value also
        bounds mid-stream silence: once samples are flowing, the returned
        generator raises
        :class:`~hypertools.core.exceptions.HypertoolsIOError` if
        nothing arrives for ~`timeout` consecutive seconds (e.g. the
        source device disconnected).
    **resolve_kwargs
        Extra keyword arguments forwarded to the underlying pylsl resolve
        call -- which differs by criterion, and so therefore do the
        arguments it accepts: `name=`/`type=` go to
        `pylsl.resolve_byprop`, which takes ``minimum=``; the "any stream"
        fallback goes to `pylsl.resolve_streams`, which takes only its
        wait time, so passing ``minimum=`` there is a `TypeError` from
        pylsl.

    Returns
    -------
    stream : LSLStream
        An infinite iterator yielding one sample (a list of channel
        values) per ``next()``, pulled from a `pylsl.StreamInlet` opened on
        the first resolved stream (a ``RuntimeWarning`` names the chosen
        stream when several match). Call ``stream.close()`` -- or use it
        as a context manager, ``with hyp.io.lsl_stream(...) as stream:``
        -- to release the inlet; it is also released when the stream is
        garbage-collected, when it aborts on a silent source, and at
        interpreter exit (see the module docstring, *Teardown*).
        ``stream.closed`` reports the state; a closed stream raises
        ``StopIteration``. `hyp.plot` does NOT close the stream when it
        stops at ``stream_max=`` -- the stream is yours to reuse or close.
        NOTE that the multi-match check is
        best-effort: LSL resolution returns as soon as at least one
        stream matches, so a second matching outlet that announces
        itself a moment later goes undetected and the first stream is
        used silently. Pass ``name=`` to pin a specific stream, or force
        ambiguity detection with ``minimum=2`` (forwarded to the pylsl
        resolve call) at the cost of always waiting the full `timeout`
        when only one matching stream exists. Passes
        `hypertools.io.streaming.is_stream`.

    Raises
    ------
    ImportError
        If `pylsl` is not installed and could not be installed on demand.
    TypeError
        If `name` or `type` is not a string (or None).
    ValueError
        If `timeout` is not a positive number of seconds.
    hypertools.core.exceptions.HypertoolsIOError
        If no matching stream is found within `timeout` seconds, if the
        matched stream has a string (non-numeric) channel format, or --
        raised from the returned generator's ``next()`` during iteration
        -- if a stream that was delivering samples goes silent for
        ~`timeout` consecutive seconds.

    Examples
    --------
    >>> import hypertools as hyp
    >>> stream = hyp.io.lsl_stream(type='EEG', timeout=5.0)  # doctest: +SKIP
    >>> hyp.plot(stream, stream_init=200, stream_chunk=20)  # doctest: +SKIP
    >>> stream.close()  # release the LSL inlet  # doctest: +SKIP

    or, equivalently, scoped to a block:

    >>> with hyp.io.lsl_stream(type='EEG') as stream:  # doctest: +SKIP
    ...     hyp.plot(stream, stream_init=200, stream_chunk=20)
    """
    from ..core.exceptions import HypertoolsIOError

    # validate lsl_stream's OWN parameters before they reach pylsl, whose
    # internal failures never name the offending argument (release-1.0
    # audit, D10-tutorials-embeddings-lsl-013: lsl_stream(name=123) raised
    # "descriptor 'encode' for 'str' objects doesn't apply to a 'int'
    # object" from deep inside pylsl).
    if name is not None and not isinstance(name, str):
        raise TypeError(
            f"name= must be a string (the LSL stream's 'name' property) or "
            f"None; got {name.__class__.__name__}: {name!r}. If your stream "
            "ids are numeric, pass the name as a string (e.g. "
            f"name={str(name)!r}).")
    if type is not None and not isinstance(type, str):
        raise TypeError(
            f"type= must be a string (the LSL stream's 'type' property, "
            f"e.g. 'EEG') or None; got {type.__class__.__name__}: {type!r}.")
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) \
            or timeout <= 0:
        raise ValueError(
            f'timeout= must be a positive number of seconds; got '
            f'{timeout!r}.')

    pylsl = _import_pylsl()

    if name is not None:
        criterion = f'name={name!r}'
        infos = pylsl.resolve_byprop('name', name, timeout=timeout,
                                      **resolve_kwargs)
    elif type is not None:
        criterion = f'type={type!r}'
        infos = pylsl.resolve_byprop('type', type, timeout=timeout,
                                      **resolve_kwargs)
    else:
        criterion = 'any stream'
        infos = pylsl.resolve_streams(wait_time=timeout, **resolve_kwargs)

    if not infos:
        raise HypertoolsIOError(
            f'no LSL stream found ({criterion}) within timeout={timeout}s. '
            f'Make sure the source application/device is running and '
            f'publishing an LSL outlet on the local network, and that '
            f'name=/type= (if given) match its StreamInfo.'
        )

    if len(infos) > 1:
        matches = ', '.join(repr(i.name()) for i in infos[:5])
        warnings.warn(
            f'{len(infos)} LSL streams match ({criterion}); using the '
            f'first one: {infos[0].name()!r}. Matching streams: '
            f'{matches}. Pass name= to select a specific stream. '
            f'{_ambiguity_caveat(criterion == "any stream")}',
            RuntimeWarning, stacklevel=2)

    if infos[0].channel_format() == pylsl.cf_string:
        raise HypertoolsIOError(
            f'the resolved LSL stream ({criterion}, '
            f'name={infos[0].name()!r}) has string-typed channels '
            "(channel_format='string', e.g. a Markers stream), but "
            'hypertools can only stream numeric channel formats -- '
            "resolve a numeric stream instead (e.g. type='EEG').")

    inlet = pylsl.StreamInlet(infos[0])

    # a stalled/disconnected device must not hang the consumer forever:
    # pull with a bounded per-sample timeout and give up (with a clear
    # error) after `timeout` seconds of consecutive silence.
    pull_timeout = min(1.0, timeout) if timeout else 1.0
    max_silent_pulls = max(1, int(round(timeout / pull_timeout))) if timeout else 10

    return LSLStream(inlet, criterion, pull_timeout, max_silent_pulls)


def _release_inlet(box):
    """Shut a `pylsl.StreamInlet` down for good: `close_stream()`, then drop
    the last reference so pylsl's `StreamInlet.__del__` destroys it
    (`lsl_destroy_inlet`). `box` is the one-element list that owned the
    inlet, so emptying it IS dropping the last reference (CPython frees it
    on the spot). Called at most once per inlet -- `weakref.finalize`
    detaches itself after its first call."""
    if not box:
        return
    inlet = box.pop()
    try:
        inlet.close_stream()
    finally:
        del inlet


class LSLStream:
    """Iterator of per-sample vectors from a `pylsl.StreamInlet`, with
    deterministic teardown. Returned by :func:`lsl_stream`; not meant to be
    constructed directly.

    Iterating pulls one sample per ``next()`` (an infinite stream that
    raises :class:`~hypertools.core.exceptions.HypertoolsIOError` when the
    source goes silent for ~`timeout` seconds, closing itself first).
    ``close()`` releases the inlet -- ``close_stream()`` and destroy --
    and so does leaving a ``with`` block, garbage collection, and
    interpreter exit (:func:`weakref.finalize`); see the module docstring,
    *Teardown*. A closed stream raises ``StopIteration``, like a closed
    generator, and ``close()`` is idempotent. Passes
    `hypertools.io.streaming.is_stream`.
    """

    def __init__(self, inlet, criterion, pull_timeout, max_silent_pulls):
        from ..core.exceptions import HypertoolsIOError
        self._error_type = HypertoolsIOError
        self._criterion = criterion
        self._pull_timeout = pull_timeout
        self._max_silent_pulls = max_silent_pulls
        self._silent_pulls = 0
        # the pull and the release are serialized so close() from another
        # thread (or from the at-exit hook) can never destroy the inlet
        # mid-pull; RLock because the silence abort closes from inside
        # __next__.
        self._lock = threading.RLock()
        # the inlet lives in a box owned by BOTH this object and its
        # finalizer, so the finalizer can release it whether it fires from
        # close(), from garbage collection, or at interpreter exit.
        self._box = [inlet]
        self._finalizer = weakref.finalize(self, _release_inlet, self._box)

    @property
    def _inlet(self):
        """The live `pylsl.StreamInlet`, or None once closed."""
        return self._box[0] if self._box else None

    @property
    def closed(self):
        """True once the inlet has been released (see :meth:`close`)."""
        return not self._box

    def close(self):
        """Release the LSL inlet: ``close_stream()`` it, then destroy it.
        Idempotent; afterwards ``next()`` raises ``StopIteration``."""
        with self._lock:
            self._finalizer()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False

    def __iter__(self):
        return self

    def __next__(self):
        with self._lock:
            inlet = self._inlet
            if inlet is None:
                raise StopIteration
            while True:
                sample, _timestamp = inlet.pull_sample(
                    timeout=self._pull_timeout)
                if sample is not None:
                    self._silent_pulls = 0
                    return sample
                self._silent_pulls += 1
                if self._silent_pulls >= self._max_silent_pulls:
                    silent = self._silent_pulls * self._pull_timeout
                    self.close()
                    raise self._error_type(
                        f'LSL stream ({self._criterion}) stopped delivering '
                        f'samples: nothing received for ~{silent:.1f}s. The '
                        f'source may have disconnected.'
                    )

    def __repr__(self):
        state = 'closed' if self.closed else 'open'
        return f'<LSLStream {self._criterion} ({state})>'
