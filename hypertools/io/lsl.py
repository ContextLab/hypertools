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
(https://labstreaminglayer.org) -- installed via `pip install
"hypertools[lsl]"`.
"""

import warnings


def _import_pylsl():
    try:
        import pylsl
    except ImportError as exc:
        raise ImportError(
            'lsl_stream() requires pylsl (which wraps the native liblsl '
            'library used by Lab Streaming Layer). Install it with '
            'pip install "hypertools[lsl]"'
        ) from exc
    return pylsl


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
    stream : generator
        An infinite generator yielding one sample (a list of channel
        values) per call, pulled from a `pylsl.StreamInlet` opened on the
        first resolved stream (a ``RuntimeWarning`` names the chosen
        stream when several match). NOTE that the multi-match check is
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
        If `pylsl` is not installed.
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

    def _samples():
        try:
            silent_pulls = 0
            while True:
                sample, _timestamp = inlet.pull_sample(timeout=pull_timeout)
                if sample is None:
                    silent_pulls += 1
                    if silent_pulls >= max_silent_pulls:
                        raise HypertoolsIOError(
                            f'LSL stream ({criterion}) stopped delivering '
                            f'samples: nothing received for ~'
                            f'{silent_pulls * pull_timeout:.1f}s. The source '
                            f'may have disconnected.'
                        )
                    continue
                silent_pulls = 0
                yield sample
        finally:
            inlet.close_stream()

    return _samples()
