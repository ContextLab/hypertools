"""Exercise the actual embedded review runner, including failed reruns."""

import inspect
import json
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
    source = (root / "scripts/feature_tour_support.py").read_text()
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
    report = json.loads((tmp_path / "results.json").read_text())
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
            (root / "scripts/feature_tour_support.py").read_text(),
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
