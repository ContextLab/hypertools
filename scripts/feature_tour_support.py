"""Embedded, self-contained review helpers for the local/Colab feature tour.

The notebook setup supplies hyp, np, pd, plt, SETTINGS, CASES, SCRATCH and the
standard-library/display imports. Keep its helper cell synchronized using
scripts/update_feature_tour.py. This file is not part of the installed package.
"""

# ruff: noqa: F821
# Run state is injected by the notebook setup; its interface is annotated below.
import base64
import uuid
from collections import Counter
from IPython.display import Image, clear_output
import hashlib
import html
import importlib.util
import inspect
import json
from pathlib import Path
import shutil
import subprocess
import time
import traceback
import warnings
import zipfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import hypertools as hyp
from IPython.display import display, HTML

# Supplied by the notebook setup; annotations document the embedded interface
# without overwriting the run's dictionaries when this cell is executed.
RESULTS: dict
MANUAL: dict
ENVIRONMENT: dict
CASES: list
SETTINGS: dict
BACKENDS: list
SCRATCH: Path
IN_COLAB: bool
NOTEBOOK_SOURCE_SHA256: str

INTERACTIVE_PLOTS = {}
CURRENT_CASE = None


def source_hash(source):
    return hashlib.sha256(source.strip().encode()).hexdigest()


def report_rows(payload=None):
    rows = list(RESULTS.values()) if payload is None else payload["cases"]
    manual = MANUAL if payload is None else payload.get("manual_review", {})
    return [
        dict(
            row,
            visual=manual.get(row["id"], {}).get("verdict", "not applicable"),
            visual_notes=manual.get(row["id"], {}).get("notes", ""),
        )
        for row in rows
    ]


def save_report():
    payload = {
        "environment": ENVIRONMENT,
        "cases": list(RESULTS.values()),
        "manual_review": MANUAL,
        "inventory": CASES,
        "notebook_source_sha256": NOTEBOOK_SOURCE_SHA256,
        "inventory_sha256": source_hash(json.dumps(CASES, sort_keys=True)),
        "interactive_plots": list(INTERACTIVE_PLOTS),
    }
    (SCRATCH / "results.json").write_text(json.dumps(payload, indent=2, default=str))
    if RESULTS:
        pd.DataFrame(report_rows()).drop(columns=["traceback"], errors="ignore").to_csv(
            SCRATCH / "results.csv", index=False
        )
    with zipfile.ZipFile(
        SCRATCH / "hypertools-feature-review.zip", "w", zipfile.ZIP_DEFLATED
    ) as archive:
        for name in ["results.json", "results.csv"]:
            if (SCRATCH / name).exists():
                archive.write(SCRATCH / name, arcname=name)


def run_case(case_id, fn):
    global CURRENT_CASE
    CURRENT_CASE = case_id
    for key in list(INTERACTIVE_PLOTS):
        if key.startswith(case_id + "-"):
            del INTERACTIVE_PLOTS[key]
    if "_LIVE_OUTPUT" in globals():
        with _LIVE_OUTPUT:
            clear_output(wait=False)
    entry = next(c for c in CASES if c["id"] == case_id)
    # Approval belongs to one execution. Even a failed/blocked rerun invalidates it.
    execution_id = uuid.uuid4().hex
    if entry["visual"]:
        MANUAL[case_id] = {
            "verdict": "not reviewed",
            "notes": "",
            "inspect": entry["visual"],
            "execution_id": execution_id,
        }
    row = dict(
        id=case_id,
        title=entry["title"],
        covers=entry["covers"],
        status="PASS",
        seconds=0.0,
        warnings=[],
        detail="",
        traceback="",
        execution_id=execution_id,
        source_sha256=source_hash(inspect.getsource(fn)),
    )
    start = time.perf_counter()
    caught = []
    try:
        missing = []
        for name in entry["requires"]:
            try:
                available = importlib.util.find_spec(name) is not None
            except ModuleNotFoundError:
                available = False
            if not available:
                missing.append(name)
        if entry["gate"] and not SETTINGS[entry["gate"]]:
            row.update(status="SKIP", detail=f"Disabled setting: {entry['gate']}")
        elif entry.get("backend") and entry["backend"] not in BACKENDS:
            row.update(status="SKIP", detail="Backend disabled in BACKENDS")
        elif missing:
            row.update(
                status="BLOCKED",
                detail="Missing prerequisite modules: " + ", ".join(missing),
            )
        else:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                fn()
    except KeyboardInterrupt:
        row.update(
            status="INTERRUPTED",
            detail="Execution interrupted; this check did not pass.",
        )
        raise
    except Exception as exc:
        missing_binary = [
            name for name in entry.get("binaries", []) if shutil.which(name) is None
        ]
        status = (
            "BLOCKED"
            if missing_binary or type(exc).__name__ == "ChromeNotFoundError"
            else "FAIL"
        )
        row.update(
            status=status,
            detail=f"{type(exc).__name__}: {exc}",
            traceback=traceback.format_exc(),
        )
        if missing_binary:
            row["detail"] += "; missing binaries: " + ", ".join(missing_binary)
        print(row["traceback"])
    finally:
        counts = Counter(
            (w.category.__name__, str(w.message), w.filename, w.lineno) for w in caught
        )
        row["warnings"] = [
            dict(category=c, message=m, filename=f, line=line_number, count=n)
            for (c, m, f, line_number), n in counts.items()
        ]
        row["seconds"] = round(time.perf_counter() - start, 3)
        RESULTS[case_id] = row
        if case_id != "GUI-native":
            plt.close("all")
        save_report()
        CURRENT_CASE = None
    tint = {"PASS": "#137333", "FAIL": "#b31412", "SKIP": "#666", "BLOCKED": "#975600"}[
        row["status"]
    ]
    display(
        HTML(
            f'<p style="border-left:5px solid {tint};padding:8px"><b>{html.escape(case_id)}: '
            f"{row['status']}</b> · {row['seconds']} s<br>{html.escape(row['detail'])}</p>"
        )
    )
    if row["warnings"]:
        display(
            HTML(
                "<details><summary>Warnings (retained in report)</summary><pre>"
                + html.escape(json.dumps(row["warnings"], indent=2))
                + "</pre></details>"
            )
        )


def expect_error(kind, fn, contains=None):
    try:
        fn()
    except kind as exc:
        if contains is not None:
            assert contains.lower() in str(exc).lower(), str(exc)
        print(f"Expected {type(exc).__name__}: {exc}")
        return exc
    raise AssertionError(f"Expected {kind}, but the call succeeded")


def finite(value, shape=None):
    array = np.asarray(value)
    if shape is not None:
        assert array.shape == shape, (array.shape, shape)
    assert np.isfinite(array).all(), "Result contains NaN or infinity"
    return array


def download_artifact(path):
    """Download bytes, not an absolute filesystem URL into a notebook server."""
    path = Path(path)
    if IN_COLAB:
        import ipywidgets as widgets
        from google.colab import files

        button = widgets.Button(
            description="Download " + path.name, layout={"width": "auto"}
        )
        button.on_click(lambda _: files.download(str(path)))
        display(button)
    else:
        data = base64.b64encode(path.read_bytes()).decode()
        display(
            HTML(
                f'<a download="{html.escape(path.name, quote=True)}" '
                f'href="data:application/octet-stream;base64,{data}">Download {html.escape(path.name)}</a>'
            )
        )


def show_result(obj):
    if isinstance(obj, dict) and "fig" in obj:
        obj = obj.get("animation") or obj["fig"]
    if isinstance(obj, hyp.HyperAnimation):
        assert obj.n_frames > 0
        display(
            HTML(obj.to_html5_video() if shutil.which("ffmpeg") else obj.to_jshtml())
        )
    elif hasattr(obj, "to_plotly_json") and hasattr(obj, "write_image"):
        from hypertools._shared.lazy_import import ensure_kaleido_chrome
        import plotly.graph_objects as go

        ensure_kaleido_chrome()
        key = f"{CURRENT_CASE or 'plot'}-{uuid.uuid4().hex[:8]}"
        path = SCRATCH / (key + ".html")
        obj.write_html(str(path), include_plotlyjs=True, auto_play=False)
        INTERACTIVE_PLOTS[key] = str(path)
        preview = go.Figure(obj)
        # A still preview of the final frame, with no browser WebGL context.
        if obj.frames:
            last = obj.frames[-1]
            for index, trace in zip(
                last.traces if last.traces is not None else range(len(last.data)),
                last.data,
            ):
                preview.data[index].update(trace.to_plotly_json())
            preview.update_layout(last.layout)
        preview.frames = []
        preview.update_layout(updatemenus=[], sliders=[])
        image_path = SCRATCH / (key + ".png")
        preview.write_image(str(image_path))
        display(Image(filename=str(image_path)))
        print(
            "Interactive controls/playback: select",
            key,
            "in the single viewer at the end of this notebook.",
        )
        download_artifact(path)
    else:
        display(obj)


def interactive_viewer():
    """Only one live iframe: replacing/removing it disposes its WebGL contexts."""
    global _LIVE_OUTPUT
    if "_LIVE_OUTPUT" in globals():
        with _LIVE_OUTPUT:
            clear_output(wait=False)
    import ipywidgets as widgets

    picker = widgets.Dropdown(options=list(INTERACTIVE_PLOTS), layout={"width": "95%"})
    open_button = widgets.Button(description="Open selected plot")
    close_button = widgets.Button(description="Close interactive plot")
    output = widgets.Output()
    _LIVE_OUTPUT = output

    def close(_):
        with output:
            clear_output(wait=False)

    def show(_):
        close(None)
        source = Path(INTERACTIVE_PLOTS[picker.value]).read_text()
        with output:
            display(
                HTML(
                    '<iframe title="Interactive HyperTools review" style="width:100%;height:750px;border:0" '
                    'srcdoc="' + html.escape(source, quote=True) + '"></iframe>'
                )
            )

    open_button.on_click(show)
    close_button.on_click(close)
    display(widgets.VBox([picker, widgets.HBox([open_button, close_button]), output]))


def verify_export(path, animated=False):
    """Decode fresh artifacts; require distinct frames, not just nonempty files."""
    from PIL import Image as PILImage, ImageSequence

    path = Path(path)
    assert path.is_file() and path.stat().st_size > 100
    suffix = path.suffix.lower()
    if suffix in [".png", ".gif", ".apng"]:
        with PILImage.open(path) as im:
            decoded = [
                (frame.convert("RGB").tobytes(), frame.info.get("duration", 0))
                for frame in ImageSequence.Iterator(im)
            ]
            frames = [frame for frame, _ in decoded]
            if animated:
                assert len(frames) >= 4, f"{path.name}: only {len(frames)} frames"
                assert len(set(frames)) >= 3, "Frames do not show distinct motion"
                assert sum(duration for _, duration in decoded) > 0, (
                    "Animation has no positive playback duration"
                )
            else:
                im.load()
    elif suffix in [".mp4", ".mov", ".avi", ".webm", ".m4v", ".mkv"]:
        assert shutil.which("ffmpeg"), "ffmpeg is required to decode this movie"
        raw = subprocess.check_output(
            [
                "ffmpeg",
                "-v",
                "error",
                "-i",
                str(path),
                "-vf",
                "scale=64:64",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "rgb24",
                "-",
            ]
        )
        size = 64 * 64 * 3
        frames = [raw[i : i + size] for i in range(0, len(raw), size)]
        assert len(frames) >= 4 and len(set(frames)) >= 3, (
            "Movie lacks distinct decodable frames"
        )
    elif suffix == ".svg":
        import xml.etree.ElementTree as ET

        assert ET.parse(path).getroot().tag.endswith("svg")
    elif suffix == ".pdf":
        assert path.read_bytes().startswith(b"%PDF-")
    elif suffix == ".html":
        text = path.read_text()
        assert "<html" in text.lower() or "<div" in text.lower()
        if animated:
            assert "Plotly.addFrames(" in text or "new Animation(" in text, (
                "HTML has no animation payload"
            )
    download_artifact(path)


def view(data, backend, **kwargs):
    result = hyp.plot(data, backend=backend, show=False, **kwargs)
    show_result(result)
    return result


def fixtures():
    a = hyp.load("random_walk", n_samples=48, n_features=6, random_state=41)
    b = hyp.load("random_walk", n_samples=48, n_features=6, random_state=42)
    return np.asarray(a), np.asarray(b)


TEXTS = [
    "the dog plays with a ball",
    "a puppy runs in the park",
    "the cat sleeps beside the dog",
    "kittens sleep in the sun",
    "stocks rise after the report",
    "bond prices change with rates",
    "investors buy stocks and bonds",
    "markets react to economic news",
]


def assert_hosted_contract(name, data):
    """Assert documented user-facing containers and dimensions."""
    assert len(data) > 0
    if name in ("wiki", "nips", "sotus"):
        assert isinstance(data, list) and all(isinstance(v, str) for v in data)
        if name == "wiki":
            assert len(data) == 3136
        if name == "sotus":
            assert len(data) == 29
    elif name == "datasaurus":
        assert isinstance(data, list) and len(data) == 13
        assert all(isinstance(v, pd.DataFrame) and v.shape == (142, 2) for v in data)
        assert all(list(v.columns) == ["x", "y"] and v.index.is_unique for v in data)
    elif name in ("weights", "weights_avg", "weights_sample", "spiral"):
        assert isinstance(data, list) and all(
            isinstance(v, np.ndarray) and v.ndim == 2 for v in data
        )
        if name == "spiral":
            assert len(data) == 2 and all(v.shape == (1000, 3) for v in data)
    elif name == "mushrooms":
        assert isinstance(data, pd.DataFrame) and len(data) == 8124
    else:
        assert np.asarray(data).ndim == 2 and np.asarray(data).shape[1] == 3
        finite(data)
