#!/usr/bin/env python

"""
Helper functions
"""

##PACKAGES##
import numpy as np
import itertools
import pandas as pd
from matplotlib.lines import Line2D

# NOTE: seaborn and scipy.interpolate are imported lazily inside the functions
# that use them -- together they added ~1s to `import hypertools`.
#
# 2026-07 audit (X7-code-org-rest-002): this module used to call
# np.seterr(divide='ignore', invalid='ignore') at import time, permanently
# silencing numpy divide/invalid warnings PROCESS-WIDE for anyone who
# imported hypertools -- masking real numerical errors in users' own
# analysis code. Suppression is now scoped locally (np.errstate) at the
# specific call sites that intentionally divide by possibly-zero ranges.


def center(x):
    """Mean-center a list of datasets around their shared (pooled) mean.

    Parameters
    ----------
    x : list of numpy.ndarray
        List of 2D arrays (observations x features). All arrays are
        stacked together to compute a single pooled mean, which is then
        subtracted from each array individually.

    Returns
    -------
    list of numpy.ndarray
        Centered copies of each array in `x`, same shapes as the inputs.
    """
    assert isinstance(x, list), "Input data to center must be list"
    x_stacked = np.vstack(x)
    return [i - np.mean(x_stacked, 0) for i in x]


def scale(x):
    """Rescale a list of datasets into the range [-1, 1] using shared bounds.

    Parameters
    ----------
    x : list of numpy.ndarray
        List of 2D arrays. All arrays are stacked together to compute a
        single pooled min/max, which is used to rescale each array.

    Returns
    -------
    list of numpy.ndarray
        Rescaled copies of each array in `x`, same shapes as the inputs,
        with values mapped into [-1, 1] based on the pooled min/max.
    """
    assert isinstance(x, list), "Input data to scale must be list"
    x_stacked = np.vstack(x)
    m1 = np.min(x_stacked)
    m2 = np.max(x_stacked - m1)

    def f(a):
        """Rescale one array into [-1, 1] using the pooled min/range.

        Constant data has zero range (m2 == 0): 0/0 is intentionally
        allowed to produce NaN/inf here without warning -- suppression is
        scoped to this division only (never process-wide; see note above).
        """
        with np.errstate(divide='ignore', invalid='ignore'):
            return 2 * (np.divide(a - m1, m2)) - 1

    return [f(i) for i in x]


def group_by_category(vals):
    """Map each value in `vals` to an integer code for its category.

    Categories are ordered by first appearance in `vals` (not sorted by
    value), so the code for the first-seen category is 0, the next
    distinct category is 1, and so on.

    Parameters
    ----------
    vals : list, or list of lists
        Values to categorize. If a list of lists is given, it is
        flattened first via `itertools.chain`.

    Returns
    -------
    list of int
        Integer category code for each (flattened) entry in `vals`.
    """
    if any(isinstance(el, list) for el in vals):
        vals = list(itertools.chain(*vals))
    val_set = list(sorted(set(vals), key=list(vals).index))
    return [val_set.index(val) for val in vals]


def vals2colors(vals, cmap='GnBu',res=100):
    """Maps values to colors
    Args:
    values (list or list of lists) - list of values to map to colors
    cmap (str) - color map (default is 'GnBu')
    res (int) - resolution of the color map (default: 100)
    Returns:
    list of rgb tuples
    """
    # flatten if list of lists
    if any(isinstance(el, list) for el in vals):
        vals = list(itertools.chain(*vals))

    # get palette from seaborn
    import seaborn as sns
    palette = np.array(sns.color_palette(cmap, res))
    vmin, vmax = np.min(vals), np.max(vals)
    if vmax == vmin:
        return [tuple(palette[0])] * len(vals)
    # bin edges span [vmin, vmax] exactly so the palette's full range is
    # used (a stray max+1 edge previously left the top of the map unused);
    # clip keeps vals == vmax inside the last palette slot
    edges = np.linspace(vmin, vmax, res + 1)
    ranks = np.clip(np.digitize(vals, edges) - 1, 0, res - 1)
    return [tuple(i) for i in palette[ranks, :]]


def vals2bins(vals,res=100):
    """Maps values to bins
    Args:
    values (list or list of lists) - list of values to map to colors
    res (int) - resolution of the color map (default: 100)
    Returns:
    list of numbers representing bins
    """
    # flatten if list of lists
    if any(isinstance(el, list) for el in vals):
        vals = list(itertools.chain(*vals))
    vmin, vmax = np.min(vals), np.max(vals)
    if vmax == vmin:
        return [0] * len(vals)
    edges = np.linspace(vmin, vmax, res + 1)
    return list(np.clip(np.digitize(vals, edges) - 1, 0, res - 1))


def interp_array(arr, interp_val=10):
    """Upsample a 1D array via PCHIP (monotonic cubic) interpolation.

    Parameters
    ----------
    arr : array-like
        1D sequence of values, indexed by integer position.
    interp_val : int, optional
        Number of interpolated samples per original sample (default: 10).
        The output has `interp_val` times as many points as `arr`
        (minus a small remainder from the final step).

    Returns
    -------
    numpy.ndarray
        The interpolated (upsampled) array.
    """
    from scipy.interpolate import PchipInterpolator as pchip
    x=np.arange(0, len(arr), 1)
    xx=np.arange(0, len(arr)-1, 1/interp_val)
    q=pchip(x,arr)
    return q(xx)


def interp_array_list(arr_list, interp_val=10):
    """Apply `interp_array` independently to each array in a list.

    Parameters
    ----------
    arr_list : list of numpy.ndarray
        List of 1D (or column-wise) arrays to interpolate.
    interp_val : int, optional
        Number of interpolated samples per original sample, passed
        through to `interp_array` (default: 10).

    Returns
    -------
    list of numpy.ndarray
        Interpolated version of each array in `arr_list`, in the same
        order.
    """
    smoothed= [np.zeros(arr_list[0].shape) for item in arr_list]
    for idx,arr in enumerate(arr_list):
        smoothed[idx] = interp_array(arr,interp_val)
    return smoothed


def parse_kwargs(x, kwargs):
    """Broadcast each kwarg in `kwargs` across the drawn traces in `x`: a
    scalar value is repeated for every trace; a list/tuple value is
    distributed one-entry-per-trace and MUST match `len(x)` exactly.

    Note `x` here is the list of DRAWN traces, which is not always the list
    of INPUT datasets: `hue=`/`cluster=`/`n_clusters=`/MultiIndex regroup the
    data (a categorical line splits each dataset into one trace per
    contiguous run, GH #291), so `plot()` propagates any per-input-dataset
    style list to the runs BEFORE calling this, leaving only trace-length
    lists to distribute here.

    GH #206: a mismatched-length list previously degraded SILENTLY to
    `None` for every trace (a user's `color=['red', 'blue']` against 3
    traces would silently plot with no color at all, no error/warning ever
    raised) -- this now raises a clear ``ValueError`` naming the kwarg, the
    length actually given, and the number of traces it needed to match,
    exactly as the original GH #206 request specified.
    """
    n = len(x)
    kwargs_list = []
    for i, item in enumerate(x):
        tmp = {}
        for kwarg in kwargs:
            val = kwargs[kwarg]
            if isinstance(val, (tuple, list)):
                if len(val) != n:
                    raise ValueError(
                        f"{kwarg}= was given as a list/tuple of length "
                        f"{len(val)}, but there are {n} trace(s) to draw; "
                        f"pass either a single value (broadcast to every "
                        f"trace) or a list/tuple of length {n}."
                    )
                tmp[kwarg] = val[i]
            else:
                tmp[kwarg] = val
        kwargs_list.append(tmp)
    return kwargs_list


def reshape_data(x, hue, labels):
    """Regroup stacked data and labels by category (for per-category plotting).

    Stacks `x` into a single array, then splits its rows back out into
    one sub-array per distinct value of `hue` (in first-seen order),
    carrying the corresponding `labels` entries along with them.

    This GLOBALLY merges every observation of a category into a single
    array regardless of its source dataset or position, so it is only
    appropriate for MARKER/scatter plots (which have no connecting edges).
    A LINE plot must use `segment_by_run` instead, which preserves order
    and dataset identity so a line never joins observations that were not
    actually adjacent in one input trajectory (GH #291).

    Parameters
    ----------
    x : list of numpy.ndarray
        Datasets to regroup; stacked row-wise before splitting.
    hue : sequence
        Category value for each row of the stacked `x` (same total
        length as `np.vstack(x)`).
    labels : sequence or None
        Per-row labels to carry along with the regrouping. If None,
        `None` is used for every row.

    Returns
    -------
    tuple of (list of numpy.ndarray, list of list)
        `x_reshaped` -- one array per distinct `hue` category, each
        containing the rows belonging to that category (stacked).
        `labels_reshaped` -- the corresponding labels for each category,
        in matching order.
    """
    categories = list(sorted(set(hue), key=list(hue).index))
    x_stacked = np.vstack(x)
    x_reshaped = [[] for _ in categories]
    labels_reshaped = [[] for _ in categories]
    if labels is None:
        labels = [None]*len(hue)
    for idx, (point, label) in enumerate(zip(hue, labels)):
        x_reshaped[categories.index(point)].append(x_stacked[idx])
        labels_reshaped[categories.index(point)].append(labels[idx])
    return [np.vstack(i) for i in x_reshaped], labels_reshaped


def segment_by_run(x, hue, labels=None):
    """Split datasets into contiguous same-category runs for LINE plots.

    Unlike `reshape_data` (which merges ALL observations of a category into
    one array regardless of position or source dataset), this walks each
    input dataset in order and starts a new segment whenever the category
    changes OR a dataset boundary is crossed. A line may then only ever
    connect observations that were genuinely adjacent within one input
    trajectory -- so separate datasets are never joined, and a category that
    recurs along a trajectory (``A A B B A A``) is not collapsed into one
    tangled polyline (GH #291).

    Parameters
    ----------
    x : list of numpy.ndarray
        Input datasets (each a 2-D array of observations, in order).
    hue : sequence
        Per-observation category id, length == total rows of ``vstack(x)``,
        in the same row order.
    labels : sequence or None
        Per-observation labels carried along with the segmentation. If None,
        each segment gets a list of ``None`` (matching `reshape_data`).

    Returns
    -------
    segments : list of numpy.ndarray
        One array per maximal run of consecutive same-category observations
        within a single input dataset, in original order.
    seg_labels : list of list
        The labels of each segment's observations, parallel to `segments`.
    seg_category : list
        The category id of each segment, parallel to `segments`.
    seg_bridge : list of bool
        ``seg_bridge[i]`` is True when segments i and i+1 are consecutive
        runs of the SAME input dataset, so a line may be bridged from i into
        i+1 (a colour transition within one trajectory); False at dataset
        boundaries. Length is ``len(segments) - 1``. Pass the complementary
        indices to ``patch_lines(breaks=...)``.
    seg_dataset : list of int
        The source input-dataset index of each segment, parallel to
        `segments`. Lets the caller propagate a per-INPUT-DATASET style
        (fmt/linewidth/marker/...) to every run that dataset produced (GH
        #291 follow-up), rather than forcing callers to know the run count.
    """
    hue = list(hue)
    labels = [None]*len(hue) if labels is None else list(labels)
    segments, seg_labels, seg_category, seg_dataset = [], [], [], []
    row = 0
    for di, xi in enumerate(x):
        arr = np.asarray(xi)
        n = arr.shape[0]
        start = 0
        while start < n:
            cat = hue[row + start]
            end = start + 1
            while end < n and hue[row + end] == cat:
                end += 1
            segments.append(arr[start:end])
            seg_labels.append(labels[row + start:row + end])
            seg_category.append(cat)
            seg_dataset.append(di)
            start = end
        row += n
    seg_bridge = [seg_dataset[i] == seg_dataset[i + 1]
                  for i in range(len(segments) - 1)]
    return segments, seg_labels, seg_category, seg_bridge, seg_dataset


def patch_lines(x, breaks=None):
    """Bridge each group's line to the start of the next group.

    Extending every group with the first point of the NEXT group makes a
    line format render one continuous curve across group (colour)
    transitions. `breaks` is an optional iterable of group indices that must
    NOT be bridged INTO from their predecessor -- used to keep a line from
    crossing a dataset boundary (GH #291), e.g. the run segments produced by
    `segment_by_run` are bridged only where ``seg_bridge`` is True.
    """
    breaks = set() if breaks is None else set(breaks)
    for idx in range(len(x)-1):
        if (idx + 1) in breaks:
            continue
        x[idx] = np.vstack([x[idx], x[idx+1][0,:]])
    return x


def is_line(format_str):
    """True if the format string draws pure lines (no markers).

    Notes: linestyle tokens are stripped BEFORE checking for marker
    characters so that '-.' (dash-dot) is recognized as a line rather than
    a '.' marker, mirroring matplotlib's own fmt grammar. The "no marker"
    sentinel keys ('', ' ', 'None', 'none') are excluded from the marker
    set -- '' is a substring of every string, which previously made this
    function return False for ALL format strings (silently disabling line
    interpolation on matplotlib versions whose Line2D.markers includes '').
    """
    if isinstance(format_str, np.bytes_):
        format_str = format_str.decode('utf-8')
    if format_str is None:
        return True
    if isinstance(format_str, (list, tuple, np.ndarray)):
        return all(is_line(f) for f in format_str)
    remainder = format_str
    for linestyle in ('-.', '--', '-', ':'):  # two-char styles first
        remainder = remainder.replace(linestyle, '')
    markers = [str(symbol) for symbol in Line2D.markers.keys()
               if str(symbol) not in ('', ' ', 'None', 'none')]

    return all(symbol not in remainder for symbol in markers)


def has_line_component(format_str):
    """True if the format string includes a LINE component (any linestyle
    token), regardless of whether it ALSO has a marker. Companion to
    `is_line` (which additionally requires the ABSENCE of a marker).

    Used to gate line-smoothing interpolation (GH #141): marker+line
    combo styles (e.g. 'o-') must get exactly the same connecting-line
    smoothing/interpolation as pure line styles (e.g. '-') -- previously
    the interpolation step was gated on `is_line`, so any format string
    with a marker character skipped interpolation entirely, leaving
    straight (unsmoothed) segments between raw points for 'o-' while '-'
    alone rendered a smoothed curve on identical data. Markers are then
    drawn separately, at the ORIGINAL (pre-interpolation) sample points --
    see `split_marker_line_fmt` and matplotlib_backend.py's static
    plot1D/2D/3D functions.
    """
    if isinstance(format_str, np.bytes_):
        format_str = format_str.decode('utf-8')
    if format_str is None:
        return True
    if isinstance(format_str, (list, tuple, np.ndarray)):
        return all(has_line_component(f) for f in format_str)
    return any(token in format_str for token in ('-.', '--', '-', ':'))


def split_marker_line_fmt(format_str):
    """Split a matplotlib format string into its LINE and MARKER
    components (GH #141), so a combined style like 'o-' can be drawn as
    two separate artists: a smoothed/interpolated line (no marker) plus
    markers at the true sample points (no connecting line between
    markers). Mirrors the token-stripping order `is_line` uses (longest
    linestyle tokens matched first, so '-.' isn't misread as '-' + '.').

    Returns
    -------
    (line_token, marker_char)
        `line_token` is the matplotlib linestyle substring (one of '-.',
        '--', '-', ':'), or None if `format_str` has no line component.
        `marker_char` is the single marker character (e.g. 'o', 's'), or
        None if there is no marker. `format_str=None` returns (None, None).
    """
    if format_str is None:
        return None, None
    if isinstance(format_str, np.bytes_):
        format_str = format_str.decode('utf-8')
    remainder = format_str
    line_token = None
    for token in ('-.', '--', '-', ':'):
        if token in remainder:
            line_token = token
            remainder = remainder.replace(token, '')
            break
    marker_char = None
    for ch in remainder:
        if str(ch) in Line2D.markers and str(ch) not in ('', ' ', 'None', 'none'):
            marker_char = ch
            break
    return line_token, marker_char


def get_type(data):
    """
    Checks what the data type is and returns it as a string label
    """
    from ..datageometry import DataGeometry

    if isinstance(data, list):
        if len(data) == 0:
            return 'list_num'  # empty list -> empty numeric dataset
        if isinstance(data[0], (str, bytes)):
            return 'list_str'
        # bools count as numbers (release-1.0 audit, F08-plot-inputs-013):
        # a python list of bools is the same data as np.array([True, ...]),
        # which has always been accepted (dtype kind 'b' -> 'arr_num').
        # np.bool_ is listed explicitly because it is NOT an np.number
        # subclass (and, under numpy >= 2, not a python bool either).
        elif isinstance(data[0], (bool, int, float, np.number, np.bool_)):
            return 'list_num'
        elif isinstance(data[0], np.ndarray):
            return 'list_arr'
        else:
            # name the offending element type (release-1.0 audit,
            # F08-plot-inputs-008); keep the 'Unsupported data type' prefix
            # (existing callers/tests match on it).
            raise TypeError(
                f"Unsupported data type: list containing "
                f"'{type(data[0]).__name__}' elements. A list dataset may "
                "hold strings (text data), numbers/bools (a single 1-D "
                "dataset), or numpy arrays (multiple datasets). Supported "
                "per-dataset types: numpy array, pandas DataFrame, pandas "
                "Series, str, list of strings, list of numbers, or a "
                "(possibly nested) list/tuple of arrays/DataFrames.")
    elif isinstance(data, np.ndarray):
        # classify by dtype rather than indexing data[0][0] -- the latter
        # crashed on 1-D arrays (data[0] is a scalar, so data[0][0] raised
        # "invalid index to scalar variable") and on empty arrays (QC 2026-07).
        # A plain 1-D feature vector is a natural input; format_data reshapes it
        # to a column below.
        if data.dtype.kind in ('U', 'S'):
            return 'arr_str'
        if (data.dtype.kind == 'O' and data.size
                and isinstance(data.reshape(-1)[0], (str, bytes))):
            return 'arr_str'
        return 'arr_num'
    elif isinstance(data, pd.DataFrame):
        return 'df'
    elif isinstance(data, (str, bytes)):
        return 'str'
    elif isinstance(data, DataGeometry):
        return 'geo'
    else:
        # name the received type (release-1.0 audit, F08-plot-inputs-008);
        # keep the 'Unsupported data type' prefix (existing callers/tests
        # match on it). pandas Series and tuples are accepted by
        # `format_data` (which converts them before calling get_type).
        raise TypeError(
            f"Unsupported data type '{type(data).__name__}'. Supported "
            "types: numpy array, pandas DataFrame, pandas Series, str, "
            "list of strings, list of numbers, or a (possibly nested) "
            "list/tuple of arrays/DataFrames.")


def convert_text(data):
    """Reshape raw string/text data into a column vector for downstream processing.

    Parameters
    ----------
    data : list, numpy.ndarray, str, or bytes
        Input data, as accepted by `get_type`.

    Returns
    -------
    numpy.ndarray or original type
        If `data` is a string, bytes, or a list of strings (`get_type`
        returns `'str'` or `'list_str'`), returns `data` reshaped into an
        (n, 1) numpy array. Otherwise, `data` is returned unchanged.
    """
    dtype = get_type(data)
    if dtype in ['list_str', 'str']:
        data = np.array(data).reshape(-1, 1)
    return data


def get_dtype(data):
    """
    Checks what the data type is and returns it as a string label
    """
    from ..datageometry import DataGeometry

    if isinstance(data, list):
        return 'list'
    elif isinstance(data, np.ndarray):
        return 'arr'
    elif isinstance(data, pd.DataFrame):
        return 'df'
    elif isinstance(data, (str, bytes)):
        return 'str'
    elif isinstance(data, DataGeometry):
        return 'geo'
    else:
        raise TypeError('Unsupported data type passed. Supported types: '
                        'Numpy Array, Pandas DataFrame, String, List of strings'
                        ', List of numbers')
