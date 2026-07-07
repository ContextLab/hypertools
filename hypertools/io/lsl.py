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
        Resolve the stream by its LSL ``type`` property (e.g. ``'EEG'``,
        ``'Gaze'``, ``'Markers'``), via
        ``pylsl.resolve_byprop('type', type, ...)``. Used only when `name`
        is not given.
    timeout : float
        Seconds to wait for a matching stream to appear on the network
        before giving up (default: 10.0). When neither `name` nor `type`
        is given, this is also the wait time used to resolve ANY
        available stream (``pylsl.resolve_streams``).
    **resolve_kwargs
        Extra keyword arguments forwarded to the underlying pylsl resolve
        call (e.g. ``minimum=`` for `pylsl.resolve_byprop`).

    Returns
    -------
    stream : generator
        An infinite generator yielding one sample (a list of channel
        values) per call, pulled from a `pylsl.StreamInlet` opened on the
        first resolved stream. Passes `hypertools.io.streaming.is_stream`.

    Raises
    ------
    ImportError
        If `pylsl` is not installed.
    hypertools.core.exceptions.HypertoolsIOError
        If no matching stream is found within `timeout` seconds.

    Examples
    --------
    >>> import hypertools as hyp
    >>> stream = hyp.io.lsl_stream(type='EEG', timeout=5.0)  # doctest: +SKIP
    >>> hyp.plot(stream, stream_init=200, stream_chunk=20)  # doctest: +SKIP
    """
    from .._shared.exceptions import HypertoolsIOError

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

    inlet = pylsl.StreamInlet(infos[0])

    def _samples():
        while True:
            sample, _timestamp = inlet.pull_sample()
            if sample is None:
                continue
            yield sample

    return _samples()
