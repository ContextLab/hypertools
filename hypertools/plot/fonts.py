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
import pandas as pd
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

# The BUNDLED default face (vendored under hypertools/external/fonts, SIL
# OFL 1.1). matplotlib ships only DejaVu Sans, so without this the default
# look varied per machine; bundling one ~570 KB face makes hypertools render
# identically everywhere. It covers Latin/Greek/Cyrillic -- broader scripts
# (CJK, emoji, Indic) are far too large to bundle and are reached through the
# per-glyph fallback stack below instead.
_BUNDLED_FAMILY = 'Noto Sans'
_BUNDLED_FONT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'external', 'fonts')

# Good-looking sans-serif faces to prefer, in order, when they are installed.
# `_BUNDLED_FAMILY` leads because hypertools ships it (so it always resolves);
# the rest are common system faces that users may prefer/recognize. Anything
# NOT installed is filtered out before reaching matplotlib, because naming a
# missing family makes matplotlib log a `findfont: ... not found` message for
# every text artist.
_NICE_SANS_FAMILIES = [
    _BUNDLED_FAMILY,
    'Helvetica Neue',
    'Helvetica',
    'Arial',
    'Segoe UI',
    'Roboto',
    'Liberation Sans',
]

# matplotlib's own bundled face -- always installed, and the widest-coverage
# fallback we can rely on (math symbols, arrows, dingbats DejaVu has that Noto
# Sans does not), so it always anchors the end of the stack.
_LAST_RESORT_FAMILY = 'DejaVu Sans'

# module-level caches (GH #205): scanning every installed font's cmap is
# not free (each candidate needs an FT2Font open + a glyph-index lookup per
# codepoint), so both the final per-codepoint-set result and any font that
# failed to load are cached for the life of the process.
_covering_font_cache = {}
_font_load_failures = set()
_bundled_registered = False


def bundled_font_files():
    """Absolute paths of the font files vendored with hypertools."""
    if not os.path.isdir(_BUNDLED_FONT_DIR):
        return []
    return [os.path.join(_BUNDLED_FONT_DIR, fname)
            for fname in sorted(os.listdir(_BUNDLED_FONT_DIR))
            if fname.lower().endswith(_FONT_FILE_EXTS)]


def register_bundled_fonts():
    """Make the vendored face(s) visible to matplotlib's font manager.

    Idempotent (matplotlib's `addfont` appends unconditionally, so repeated
    calls would pile up duplicate entries). Additive only -- it registers an
    extra font, never changes the user's rcParams or removes anything.
    """
    global _bundled_registered
    if _bundled_registered:
        return
    _bundled_registered = True          # set first: never retry a bad file
    for path in bundled_font_files():
        try:
            font_manager.fontManager.addfont(path)
        except Exception:  # noqa: BLE001 - a bad/corrupt bundled file must
            pass           # never break plotting; the stack falls back below


def installed_families():
    """Set of family names matplotlib can currently resolve."""
    register_bundled_fonts()
    return {entry.name for entry in font_manager.fontManager.ttflist}


def sans_serif_stack(first=None, extra=None):
    """Ordered font families for matplotlib's PER-GLYPH fallback.

    matplotlib (>= 3.6) walks a ``font.family`` LIST per character, so a stack
    renders text whose glyphs live in different faces -- e.g. Latin from Noto
    Sans, Japanese from an installed CJK face, and math symbols from DejaVu
    Sans -- instead of drawing "tofu" boxes for whatever the single active font
    lacks. Ordering: `first` (an EXPLICIT ``font=``) if given, then the
    good-looking sans faces that are installed, then the pan-Unicode/CJK
    families, then `extra` (an AUTO-detected family filling a coverage gap --
    added as a FALLBACK so Noto stays primary rather than taking over ASCII),
    and finally DejaVu Sans as the widest-coverage anchor.

    Every entry is filtered against what is actually installed, so matplotlib
    never logs a ``findfont: ... not found`` message for a family in the stack.
    """
    available = installed_families()
    stack = []

    def _add(name):
        if name and name in available and name not in stack:
            stack.append(name)

    _add(first)                              # an explicit font= wins outright
    for family in _NICE_SANS_FAMILIES:
        _add(family)
    for family in _PREFERRED_FAMILIES:       # pan-Unicode / CJK coverage
        _add(family)
    _add(extra)                              # auto gap-filler: fallback only
    _add(_LAST_RESORT_FAMILY)
    if _LAST_RESORT_FAMILY not in stack:     # matplotlib always ships it, but
        stack.append(_LAST_RESORT_FAMILY)    # never return an empty stack
    return stack


def _iter_texts(obj):
    """Recursively yield every string found in `obj`, which may be a bare
    string, `None`, a numpy array, a pandas Series/Index/Categorical (e.g.
    a categorical `hue=` -- previously silently skipped, so CJK labels
    passed as a Series rendered as tofu while the identical list worked;
    release-1.0 audit, F24-014), a dict (its values are scanned), or an
    arbitrarily nested list/tuple of the above (the shapes `labels=`/
    `legend=`/colorbar tick lists/`hue=` can take). Non-string,
    non-container leaves (numbers, bools, ...) are ASCII-only by
    definition and are silently skipped."""
    if obj is None:
        return
    if isinstance(obj, str):
        yield obj
    elif isinstance(obj, (np.ndarray, pd.Series, pd.Index, pd.Categorical)):
        for item in np.asarray(obj).tolist():
            yield from _iter_texts(item)
    elif isinstance(obj, dict):
        for item in obj.values():
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


# Unicode NONCHARACTERS -- guaranteed by the standard never to be assigned,
# so no legitimate text font maps them. Universal-fallback/placeholder fonts
# (e.g. macOS's LastResort, whose format-13 cmap maps EVERY codepoint to a
# missing-glyph box) do claim them -- which is exactly how we detect and
# exclude such fonts: "covering" text with placeholder boxes is the tofu
# problem this module exists to solve, not a solution to it.
_NONCHARACTER_PROBES = (0xFDD0, 0x10FFFE)


def _font_covers(fname, codepoints):
    """Whether the font file at `fname` has a REAL glyph for every codepoint
    in `codepoints`, per its cmap (a nonzero `get_char_index` return). Fonts
    that also claim to cover Unicode noncharacters are universal-fallback/
    placeholder fonts (they'd render boxes, not glyphs) and never qualify.
    Fonts that fail to load (corrupt/unsupported files) are cached as
    failures and treated as non-covering rather than raising."""
    if fname in _font_load_failures:
        return False
    try:
        ft = FT2Font(fname)
        if any(ft.get_char_index(cp) != 0 for cp in _NONCHARACTER_PROBES):
            return False
        return all(ft.get_char_index(cp) != 0 for cp in codepoints)
    except Exception:
        _font_load_failures.add(fname)
        return False


def _codepoints_uncovered_by_stack(codepoints):
    """Subset of `codepoints` that NO family in `sans_serif_stack()` can draw.

    matplotlib walks the stack per glyph, so a character only renders as
    "tofu" if every family in it lacks that character -- which is a much
    narrower (and much rarer) condition than "no SINGLE font covers all of
    the text at once".
    """
    remaining = set(codepoints)
    for family in sans_serif_stack():
        if not remaining:
            break
        try:
            fname = font_manager.findfont(
                FontProperties(family=[family]), fallback_to_default=False)
        except Exception:  # noqa: BLE001 - unresolvable family covers nothing
            continue
        if fname in _font_load_failures:
            continue
        try:
            ft = FT2Font(fname)
            # skip universal-placeholder fonts, exactly as `_font_covers` does
            if any(ft.get_char_index(cp) != 0 for cp in _NONCHARACTER_PROBES):
                continue
            remaining = {cp for cp in remaining if ft.get_char_index(cp) == 0}
        except Exception:  # noqa: BLE001
            _font_load_failures.add(fname)
    return remaining


def _weight_style_key(entry):
    """Sort key preferring REGULAR-looking faces within a family: upright
    styles first, then weights closest to 400 (regular), then filename for
    determinism. Previously each family's files were ordered by filename
    alone, which on macOS put 'Hiragino ... W0.ttc' (weight 100,
    hairline-thin) ahead of the regular W4 face, so auto-detected CJK text
    rendered ultra-light next to regular-weight ASCII (release-1.0 audit,
    F24-008)."""
    weight = entry.weight
    if isinstance(weight, str):
        weight = font_manager.weight_dict.get(weight, 400)
    try:
        weight = int(weight)
    except (TypeError, ValueError):
        weight = 400
    style_penalty = 0 if entry.style == 'normal' else 1
    return (style_penalty, abs(weight - 400), entry.fname)


def _ordered_font_entries():
    """Every `ttflist` entry, preferred CJK/pan-Unicode families first (in
    `_PREFERRED_FAMILIES` order, each family's own multiple weight/style
    files sorted regular-weight-first via `_weight_style_key`), then
    everything else (sorted by family name, then the same
    regular-weight-first key)."""
    ttflist = font_manager.fontManager.ttflist
    by_name = {}
    for entry in ttflist:
        by_name.setdefault(entry.name, []).append(entry)

    ordered = []
    seen_names = set()
    for name in _PREFERRED_FAMILIES:
        entries = by_name.get(name)
        if entries:
            ordered.extend(sorted(entries, key=_weight_style_key))
            seen_names.add(name)

    remaining = [e for e in ttflist if e.name not in seen_names]
    ordered.extend(sorted(remaining,
                          key=lambda e: (e.name,) + _weight_style_key(e)))
    return ordered


def find_covering_font(texts):
    """Find an EXTRA font needed to cover text the default stack can't render.

    hypertools already draws every text surface through the Noto-first
    fallback stack (`sans_serif_stack`), which a per-glyph renderer walks one
    character at a time -- so ordinary accented Latin, Greek, Cyrillic, and
    common math symbols are already covered by the bundled Noto Sans (plus
    DejaVu Sans), and CJK is covered too on any machine whose installed CJK
    families are in the stack. This function therefore returns a font ONLY
    when the stack has a genuine COVERAGE GAP, so that a single accent or
    Greek letter never swaps the whole plot onto an unrelated platform font
    (maintainer font review).

    Returns `None` when `texts` is all ASCII, or when the default stack
    already covers every character (the common case -- no override needed).
    When the stack CANNOT draw some character, returns a
    `FontProperties` for an installed font covering that GAP (to be ADDED to
    the fallback stack, keeping Noto primary), or -- if nothing installed
    covers the gap either -- emits one `UserWarning` naming a few of the
    truly-missing characters and returns `None`.

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

    # (1) make the bundled face visible, then (2) ask what the normal
    # Noto-first stack cannot already render.
    register_bundled_fonts()
    gap = _codepoints_uncovered_by_stack(codepoints)
    if not gap:
        # (3) the stack covers everything -> no override; keep Noto primary.
        _covering_font_cache[key] = None
        return None

    # (4) a real gap: search installed fonts for one covering ONLY the gap
    # codepoints (not the whole string), to be added as an extra fallback.
    for entry in _ordered_font_entries():
        if _font_covers(entry.fname, gap):
            result = FontProperties(fname=entry.fname)
            _covering_font_cache[key] = result
            return result

    # nothing installed covers the gap either -> warn about those characters
    sample = ''.join(chr(cp) for cp in sorted(gap)[:5])
    warnings.warn(
        f"hypertools: no installed font covers the character(s) "
        f"{sample!r} (and possibly others) needed for this plot's text -- "
        f"they will render as 'tofu' (empty boxes). Pass "
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

    - `None` (default): auto-detect via `find_covering_font`, which returns
      a font ONLY when the default Noto-first fallback stack has a real
      COVERAGE GAP (a character no stack family can draw). For ASCII, and
      for accented Latin/Greek/Cyrillic/math and any CJK the stack already
      covers, it returns `None` -- no override, so the primary face stays
      the bundled Noto Sans. An auto-detected gap font is meant to be ADDED
      to the fallback stack (keeping Noto primary), NOT applied as a single
      face to whole text artists (see `hyp.plot`'s handling).
    - a `str`: either an installed font FAMILY NAME (resolved via
      matplotlib's font lookup; hyphenated and generic names like
      'sans-serif' work) or a path to a `.ttf`/`.otf`/`.ttc` FILE
      (detected by `os.path.exists`, so relative and absolute paths both
      work; the file is verified to be a loadable font HERE, not at
      draw time).
    - a `matplotlib.font_manager.FontProperties` instance: passed through
      unchanged.

    Raises `ValueError` if `font` is a string that is neither an existing
    loadable font file nor a family name matplotlib can resolve, listing
    what was tried.
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
            # validate NOW that the file is actually a loadable font --
            # otherwise the failure surfaces much later, at draw/save time,
            # as a cryptic ft2font 'RuntimeError: Can not load face' that
            # never mentions font= (release-1.0 audit, F24-003)
            try:
                FT2Font(font)
            except Exception as exc:
                raise ValueError(
                    f"font={font!r} exists but is not a loadable font file "
                    f"({exc}); pass a valid .ttf/.otf/.ttc font file, an "
                    f"installed font family name, or a "
                    f"matplotlib.font_manager.FontProperties instance."
                ) from exc
            return FontProperties(fname=font)

        # family passed as a LIST: a bare string family is parsed by
        # matplotlib as a fontconfig PATTERN, so any hyphenated name --
        # including the generic 'sans-serif' -- crashed with an uncaught
        # ParseException before the guarded lookup below ever ran
        # (release-1.0 audit, F24-001)
        fp = FontProperties(family=[font])
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
