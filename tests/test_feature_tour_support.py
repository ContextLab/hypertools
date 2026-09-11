"""Exercise the actual embedded review runner, including failed reruns."""

import inspect
import json
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import hypertools as hyp


def good():
    assert hyp.load(np.ones((3, 2))).shape == (3, 2)


def bad():
    hyp.predict(np.empty((0, 2)))


def test_failed_rerun_invalidates_visual_verdict_and_csv_keeps_both(tmp_path):
    root = Path(__file__).resolve().parents[1]
    ns = dict(
        globals(),
        SCRATCH=tmp_path,
        RESULTS={},
        MANUAL={},
        SETTINGS={},
        BACKENDS=["matplotlib"],
        ENVIRONMENT={},
        NOTEBOOK_SOURCE_SHA256="source",
        IN_COLAB=False,
        CASES=[
            dict(
                id="check",
                title="Real API",
                covers=[],
                requires=[],
                gate=None,
                visual="Inspect output",
            )
        ],
    )
    source = (root / "scripts/feature_tour_support.py").read_text(encoding="utf-8")
    exec(compile(source, str(root / "scripts/feature_tour_support.py"), "exec"), ns)
    ns["run_case"]("check", good)
    first = ns["MANUAL"]["check"]["execution_id"]
    ns["MANUAL"]["check"].update(verdict="pass", notes="Previous rendering accepted")
    ns["run_case"]("check", bad)
    assert ns["RESULTS"]["check"]["status"] == "FAIL"
    assert ns["MANUAL"]["check"]["verdict"] == "not reviewed"
    assert ns["MANUAL"]["check"]["execution_id"] != first
    ns["MANUAL"]["check"].update(verdict="fail", notes="Broken display")
    ns["save_report"]()
    csv = pd.read_csv(tmp_path / "results.csv")
    assert csv.loc[0, "visual"] == "fail" and csv.loc[0, "status"] == "FAIL"
    report = json.loads((tmp_path / "results.json").read_text(encoding="utf-8"))
    assert ns["report_rows"](report)[0]["visual_notes"] == "Broken display"
    assert report["cases"][0]["source_sha256"] == ns["source_hash"](
        inspect.getsource(bad)
    )


def test_export_check_rejects_single_frame_image(tmp_path):
    from PIL import Image

    root = Path(__file__).resolve().parents[1]
    ns = dict(globals(), IN_COLAB=False)
    exec(
        compile(
            (root / "scripts/feature_tour_support.py").read_text(encoding="utf-8"),
            str(root / "scripts/feature_tour_support.py"),
            "exec",
        ),
        ns,
    )
    path = tmp_path / "still.gif"
    Image.new("RGB", (100, 100), "steelblue").save(path)
    import pytest

    with pytest.raises(AssertionError, match="only 1 frames"):
        ns["verify_export"](path, animated=True)


def test_unicode_export_opens_in_the_actual_viewer(tmp_path):
    """Windows' default cp1252 cannot decode this actual UTF-8 Plotly file."""
    import html
    import pytest

    go = pytest.importorskip("plotly.graph_objects")
    pytest.importorskip("ipywidgets")
    from IPython.core.interactiveshell import InteractiveShell
    from IPython.utils.capture import capture_output

    root = Path(__file__).resolve().parents[1]
    source = (root / "scripts/feature_tour_support.py").read_text(encoding="utf-8")
    ns = dict(SCRATCH=tmp_path, IN_COLAB=False)
    exec(compile(source, str(root / "scripts/feature_tour_support.py"), "exec"), ns)
    label = "測定 “脳” — café"
    path = tmp_path / "unicode-plot.html"
    fig = go.Figure(
        data=[go.Scatter(x=[0, 1], y=[0, 1])],
        frames=[go.Frame(name="end", data=[go.Scatter(x=[0, 1], y=[1, 0])])],
        layout={"title": label},
    )
    fig.write_html(
        str(path),
        include_plotlyjs=True,
        auto_play=False,
        post_script=f"document.title = {json.dumps(label, ensure_ascii=False)};",
    )
    assert label in path.read_text(encoding="utf-8")
    with pytest.raises(UnicodeDecodeError):
        path.read_bytes().decode("cp1252")
    ns["INTERACTIVE_PLOTS"]["unicode"] = str(path)
    had_shell = InteractiveShell.initialized()
    InteractiveShell.instance()
    try:
        with capture_output(display=True) as captured, \
                warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            ns["verify_export"](path, animated=True)
            viewer = ns["interactive_viewer"]()
            viewer.children[1].children[0].click()
        # Colab printed IPython's "Consider using IPython.display.IFrame"
        # above every opened plot (2026-09-11)
        assert not [w for w in caught if "IFrame" in str(w.message)]
        frames = [
            o.data["text/html"]
            for o in captured.outputs
            if "text/html" in o.data and "<iframe" in o.data["text/html"]
        ]
        assert len(frames) == 1
        assert label in html.unescape(frames[0])
        assert "Plotly.newPlot(" in html.unescape(frames[0])
        viewer.children[1].children[1].click()
    finally:
        if not had_shell:
            InteractiveShell.clear_instance()


def test_summary_cell_shows_the_interactive_viewer_once():
    """interactive_viewer() displays its widget AND returns it; as a cell's
    bare last expression Jupyter displayed it a second time, so Colab showed
    two viewers (fresh-Colab review 2026-09-11)."""
    import ast
    repo = Path(__file__).resolve().parents[1]
    source = (repo / "scripts" / "update_feature_tour.py").read_text(encoding="utf-8")
    templates = [
        node.value for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
        and node.value.startswith("summary=pd.DataFrame")
        and "interactive_viewer" in node.value
    ]
    assert len(templates) == 1
    last = ast.parse(templates[0]).body[-1]
    bare_viewer = (isinstance(last, ast.Expr) and isinstance(last.value, ast.Call)
                   and getattr(last.value.func, "id", None) == "interactive_viewer")
    assert not bare_viewer
