#!/usr/bin/env python

"""Font resolution for multibyte (e.g. CJK) text support (GH #205).

matplotlib's default font (DejaVu Sans) has no glyphs for most non-Latin
scripts -- Japanese/Chinese/Korean text rendered with it shows up as "tofu"
(empty boxes), silently, with only a `UserWarning` per missing glyph at
draw time. This module:

- auto-detects an installed font that covers every non-ASCII character
  hypertools is about to draw (`find_covering_font`), so `labels=` (and
  other text hypertools draws) "just work" for CJK/Cyrillic/etc. text
  without the caller having to know or care which font supports it, and
- resolves the user-facing `font=` kwarg (`resolve_font`), which accepts
  `None` (auto-detect), a font family name, a path to a font file, or a
  `matplotlib.font_manager.FontProperties` instance.

No new dependencies: `FT2Font` (used to read a candidate font's character
map) ships with matplotlib itself.
"""

import os
import warnings

import numpy as np
import matplotlib.font_manager as font_manager
from matplotlib.font_manager import FontProperties
from matplotlib.ft2font import FT2Font

# Preference order when multiple installed fonts could cover the needed
# characters: well-known pan-CJK/pan-Unicode families first (checked in
# this exact order), then everything else `fontManager` knows about (in a
# deterministic, alphabetical-by-family-then-file order) -- so the font
# picked for a given set of codepoints is stable across runs/machines that
# happen to have more than one candidate installed.
_PREFERRED_FAMILIES = [
    'Noto Sans CJK JP',
    'Noto Sans CJK SC',
    'Hiragino Sans',
    'Hiragino Kaku Gothic ProN',
    'Yu Gothic',
    'MS Gothic',
    'Meiryo',
    'Arial Unicode MS',
    'Noto Sans',
]

_FONT_FILE_EXTS = ('.ttf', '.otf', '.ttc')

# module-level caches (GH #205): scanning every installed font's cmap is
# not free (each candidate needs an FT2Font open + a glyph-index lookup per
# codepoint), so both the final per-codepoint-set result and any font that
# failed to load are cached for the life of the process.
_covering_font_cache = {}
_font_load_failures = set()


def _iter_texts(obj):
    """Recursively yield every string found in `obj`, which may be a bare
    string, `None`, a numpy array (e.g. a categorical `hue=`), or an
    arbitrarily nested list/tuple of the above (the shapes `labels=`/
    `legend=`/colorbar tick lists/`hue=` can take). Non-string,
    non-container leaves (numbers, bools, ...) are ASCII-only by
    definition and are silently skipped."""
    if obj is None:
        return
    if isinstance(obj, str):
        yield obj
    elif isinstance(obj, np.ndarray):
        for item in obj.tolist():
            yield from _iter_texts(item)
    elif isinstance(obj, (list, tuple)):
        for item in obj:
            yield from _iter_texts(item)


def _non_ascii_codepoints(texts):
    """The set of codepoints > 127 across every string in (possibly
    nested) `texts`."""
    codepoints = set()
    for text in _iter_texts(texts):
        for ch in text:
            cp = ord(ch)
            if cp > 127:
                codepoints.add(cp)
    return codepoints


def _font_covers(fname, codepoints):
    """Whether the font file at `fname` has a glyph for every codepoint in
    `codepoints`, per its cmap (a nonzero `get_char_index` return). Fonts
    that fail to load (corrupt/unsupported files) are cached as failures
    and treated as non-covering rather than raising."""
    if fname in _font_load_failures:
        return False
    try:
        ft = FT2Font(fname)
        return all(ft.get_char_index(cp) != 0 for cp in codepoints)
    except Exception:
        _font_load_failures.add(fname)
        return False


def _ordered_font_entries():
    """Every `ttflist` entry, preferred CJK/pan-Unicode families first (in
    `_PREFERRED_FAMILIES` order, each family's own multiple weight/style
    files sorted by filename for determinism), then everything else
    (sorted by family name, then filename)."""
    ttflist = font_manager.fontManager.ttflist
    by_name = {}
    for entry in ttflist:
        by_name.setdefault(entry.name, []).append(entry)

    ordered = []
    seen_names = set()
    for name in _PREFERRED_FAMILIES:
        entries = by_name.get(name)
        if entries:
            ordered.extend(sorted(entries, key=lambda e: e.fname))
            seen_names.add(name)

    remaining = [e for e in ttflist if e.name not in seen_names]
    ordered.extend(sorted(remaining, key=lambda e: (e.name, e.fname)))
    return ordered


def find_covering_font(texts):
    """Find an installed font whose character map covers every non-ASCII
    codepoint across `texts` (a string, or an arbitrarily nested list/
    tuple of strings/None).

    Returns `None` if `texts` has no non-ASCII characters at all (no
    override needed -- the default font is fine), or if no installed font
    covers everything (a single `UserWarning` is emitted, naming a few of
    the missing characters, before returning `None`).

    Results are cached (keyed by the frozenset of needed codepoints) since
    the same label/legend/title text set is often resolved more than once
    per plot (e.g. once for labels, once for the legend).
    """
    codepoints = _non_ascii_codepoints(texts)
    if not codepoints:
        return None

    key = frozenset(codepoints)
    if key in _covering_font_cache:
        return _covering_font_cache[key]

    for entry in _ordered_font_entries():
        if _font_covers(entry.fname, codepoints):
            result = FontProperties(fname=entry.fname)
            _covering_font_cache[key] = result
            return result

    sample = ''.join(chr(cp) for cp in sorted(codepoints)[:5])
    warnings.warn(
        f"hypertools: no installed font covers the character(s) "
        f"{sample!r} (and possibly others) needed for this plot's text -- "
        f"it will likely render as 'tofu' (empty boxes). Pass "
        f"font=<family name or path to a .ttf/.otf/.ttc file> to "
        f"hyp.plot(...), or install a pan-Unicode font such as 'Noto Sans "
        f"CJK' (e.g. `apt-get install fonts-noto-cjk` on Debian/Ubuntu).",
        UserWarning,
        stacklevel=3,
    )
    _covering_font_cache[key] = None
    return None


def resolve_font(font, texts):
    """Resolve the user-facing `font=` kwarg into a
    `matplotlib.font_manager.FontProperties` (or `None`, meaning "use
    matplotlib's default -- no override needed").

    `font` may be:

    - `None` (default): auto-detect via `find_covering_font`, but ONLY
      when `texts` actually contains non-ASCII characters -- an
      ASCII-only plot gets `None` back (no override, byte-identical
      rendering to before this feature existed).
    - a `str`: either an installed font FAMILY NAME (resolved via
      matplotlib's font lookup) or a path to a `.ttf`/`.otf`/`.ttc` FILE
      (detected by `os.path.exists`, so relative and absolute paths both
      work).
    - a `matplotlib.font_manager.FontProperties` instance: passed through
      unchanged.

    Raises `ValueError` if `font` is a string that is neither an existing
    file path nor a family name matplotlib can resolve, listing what was
    tried.
    """
    if isinstance(font, FontProperties):
        return font

    if font is None:
        return find_covering_font(texts)

    if isinstance(font, str):
        _, ext = os.path.splitext(font)
        looks_like_path = ext.lower() in _FONT_FILE_EXTS or os.path.exists(font)
        if looks_like_path:
            if not os.path.exists(font):
                raise ValueError(
                    f"font={font!r} looks like a font file path (based on "
                    f"its extension) but no such file exists."
                )
            return FontProperties(fname=font)

        fp = FontProperties(family=font)
        try:
            font_manager.findfont(fp, fallback_to_default=False)
        except Exception as exc:
            raise ValueError(
                f"font={font!r} is not a recognized installed font family "
                f"and not an existing file path. Tried: matplotlib family "
                f"lookup ({exc}); os.path.exists({font!r}) -> False. Pass "
                f"a valid installed family name, a path to a .ttf/.otf/"
                f".ttc file, or a matplotlib.font_manager.FontProperties "
                f"instance."
            ) from exc
        return fp

    raise ValueError(
        f"font= must be None, a string (family name or file path), or a "
        f"matplotlib.font_manager.FontProperties instance; got "
        f"{type(font)!r}."
    )
