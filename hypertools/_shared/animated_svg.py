#!/usr/bin/env python
"""Animated SVG assembly for hypertools animation export.

Each animation frame is rendered as a standalone (vector) SVG; this module
stitches them into a single self-contained animated SVG using SMIL: every
frame is wrapped in a group whose `display` attribute is switched on for its
time slot (discrete calcMode), looping indefinitely. The result plays in any
modern browser with no JavaScript.
"""

import re


_SVG_TAG = re.compile(r'<svg\b[^>]*>', re.DOTALL)
# keep the unit suffix (pt/px) so the outer document matches its frames
_DIMS = re.compile(r'\b(width|height)="([\d.]+(?:pt|px)?)"')


def combine_frames_svg(frame_svgs, duration):
    """Combine per-frame SVG documents into one SMIL-animated SVG.

    Parameters
    ----------
    frame_svgs : list of str
        Complete SVG documents (one per animation frame), all the same size.
    duration : float
        Total loop duration in seconds.

    Returns
    -------
    str : a single animated SVG document.
    """
    if not frame_svgs:
        raise ValueError('no frames to combine')

    width, height = _frame_dims(frame_svgs[0])
    n = len(frame_svgs)

    parts = [
        '<?xml version="1.0" encoding="utf-8"?>\n'
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'xmlns:xlink="http://www.w3.org/1999/xlink" '
        f'width="{width}" height="{height}">'
    ]
    for i, doc in enumerate(frame_svgs):
        body = _strip_prolog(_namespace_ids(doc, f'f{i}'))
        start = i / n
        end = (i + 1) / n
        if i == 0:
            values, key_times = 'inline;none', f'0;{end:.6f}'
        else:
            values = 'none;inline;none'
            key_times = f'0;{start:.6f};{end:.6f}'
        parts.append(
            f'<g display="none">'
            f'<animate attributeName="display" values="{values}" '
            f'keyTimes="{key_times}" dur="{duration}s" '
            f'calcMode="discrete" repeatCount="indefinite"/>'
            f'{body}</g>'
        )
    parts.append('</svg>')
    return ''.join(parts)


def _frame_dims(doc):
    match = _SVG_TAG.search(doc)
    if match is None:
        raise ValueError('input is not an SVG document')
    dims = dict(_DIMS.findall(match.group(0)))
    return dims.get('width', '640'), dims.get('height', '480')


def _strip_prolog(doc):
    """Drop the XML prolog / doctype so the document can be nested."""
    start = doc.find('<svg')
    return doc[start:]


def _namespace_ids(doc, prefix):
    """Prefix all element ids (and references to them) so frames don't
    collide inside the combined document."""
    doc = re.sub(r'\bid="', f'id="{prefix}-', doc)
    doc = re.sub(r'url\(#', f'url(#{prefix}-', doc)
    doc = re.sub(r'\b(xlink:)?href="#', rf'\1href="#{prefix}-', doc)
    return doc
