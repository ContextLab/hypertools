#!/usr/bin/env python
"""Smoke-test an INSTALLED hypertools artifact (release qualification, CI #8).

Run with the fresh-venv interpreter that pip-installed the built artifact
(wheel OR sdist -- both are exercised in CI, since their build/discovery
paths differ), from a directory that is NOT the repo, so an accidental
import of the source tree cannot mask a packaging gap:

    /tmp/fresh/bin/python /path/to/repo/scripts/wheel_smoke_test.py

Exercises that the public API is importable from the installed package, that
the bundled config.ini shipped (defaults resolve, not silently empty), and
that a real minimal pipeline runs end-to-end.
"""
import os
import sys

os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
import hypertools as hyp

PUBLIC = (
    "plot", "analyze", "reduce", "align", "normalize", "cluster", "manip",
    "predict", "impute", "load", "save", "apply_model", "Pipeline",
    "describe", "set_interactive_backend", "HyperAnimation",
    "supported_models", "HypertoolsError", "HypertoolsIOError",
)


def main():
    # the installed wheel -- not the source checkout
    assert "site-packages" in hyp.__file__, (
        f"hypertools imported from {hyp.__file__}, not the installed wheel")

    for name in PUBLIC:
        assert hasattr(hyp, name), f"missing public export: {name}"

    # config.ini shipped in the wheel -> published defaults resolve
    from hypertools.core.configurator import get_default_options
    assert get_default_options()["reduce"] != {}, "config.ini missing from wheel"

    # a real minimal pipeline runs end-to-end on the installed package
    rng = np.random.default_rng(0)
    x = np.cumsum(rng.standard_normal((60, 8)), axis=0)
    fig = hyp.plot(x, ".", reduce="PCA", ndims=3, show=False)
    assert fig is not None
    assert np.asarray(hyp.reduce(x, "PCA", ndims=3)).shape == (60, 3)
    assert "KMeans" in hyp.supported_models()

    print(f"wheel smoke test OK; hypertools {hyp.__version__} @ {hyp.__file__}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:  # surface a clear non-zero exit for CI
        print(f"WHEEL SMOKE TEST FAILED: {type(e).__name__}: {e}",
              file=sys.stderr)
        raise
