#!/usr/bin/env python
"""Regenerate the built-in example datasets as PLAIN pickles (no DataGeometry).

HyperTools 2.0 removes the ``DataGeometry`` class. The hosted example datasets
(``weights``, ``spiral``, ``mushrooms``, ...) are currently stored on Google
Drive / Dropbox as *pickled DataGeometry objects*, so once the class is deleted
``hyp.load('spiral')`` can no longer unpickle them.

This script must run WHILE ``DataGeometry`` still exists. For every entry in
``EXAMPLE_DATA`` it loads the dataset, and — if the loaded object is a
``DataGeometry`` — extracts its raw ``.get_data()`` payload and writes it as a
plain pickle under ``rehost/``. Datasets that already load as plain
arrays/DataFrames/sklearn pipelines are copied through unchanged and flagged as
"no re-host needed".

**Manual step (Jeremy):** upload the files listed in ``rehost/MANIFEST.txt`` to
replace the corresponding Google-Drive/Dropbox files (keeping the same share
IDs), OR host them at new IDs and update ``EXAMPLE_DATA`` in
``hypertools/io/load.py`` accordingly. Only AFTER the re-hosted plain pickles are
live can ``DataGeometry`` be deleted (Plan 7 Task 7).

Run: ``.venv/bin/python scripts/rehost_example_datasets.py``
"""
import os
import pickle
import traceback

import numpy as np
import pandas as pd

import hypertools as hyp
from hypertools.io.load import EXAMPLE_DATA
from hypertools.datageometry import DataGeometry

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "rehost")
os.makedirs(OUT_DIR, exist_ok=True)


def _describe(obj):
    """One-line type/shape description of a loaded object."""
    if isinstance(obj, list):
        shapes = [getattr(x, "shape", type(x).__name__) for x in obj[:3]]
        return f"list[{len(obj)}] e.g. {shapes}"
    if isinstance(obj, (np.ndarray, pd.DataFrame)):
        return f"{type(obj).__name__}{getattr(obj, 'shape', '')}"
    return type(obj).__name__


def main():
    manifest = []
    for name, source in EXAMPLE_DATA.items():
        try:
            loaded = hyp.load(name)
        except Exception as e:  # noqa: BLE001 — record and continue per dataset
            manifest.append((name, source, "LOAD FAILED", f"{type(e).__name__}: {e}", ""))
            print(f"[FAIL ] {name}: {type(e).__name__}: {e}")
            traceback.print_exc()
            continue

        if isinstance(loaded, DataGeometry):
            plain = loaded.get_data()  # raw array / list-of-arrays / DataFrame
            out_path = os.path.abspath(os.path.join(OUT_DIR, f"{name}.pkl"))
            with open(out_path, "wb") as f:
                pickle.dump(plain, f, protocol=pickle.HIGHEST_PROTOCOL)
            # verify the plain pickle round-trips WITHOUT DataGeometry
            with open(out_path, "rb") as f:
                back = pickle.load(f)
            assert type(back).__name__ != "DataGeometry", f"{name} still geo!"
            manifest.append((name, source, "RE-HOST (was geo)", _describe(plain), out_path))
            print(f"[REHOST] {name}: geo -> {_describe(plain)} -> {out_path}")
        else:
            manifest.append((name, source, "no re-host needed (already plain)",
                             _describe(loaded), ""))
            print(f"[PLAIN ] {name}: {_describe(loaded)} (no re-host needed)")

    manifest_path = os.path.abspath(os.path.join(OUT_DIR, "MANIFEST.txt"))
    with open(manifest_path, "w") as f:
        f.write("HyperTools 2.0 example-dataset re-host manifest\n")
        f.write("=" * 70 + "\n")
        f.write("Upload each RE-HOST file to replace the file at its share ID\n")
        f.write("(or host at a new ID and update EXAMPLE_DATA in io/load.py).\n")
        f.write("Only datasets marked RE-HOST were DataGeometry pickles.\n\n")
        for name, source, action, desc, path in manifest:
            f.write(f"- {name}\n")
            f.write(f"    source : {source}\n")
            f.write(f"    action : {action}\n")
            f.write(f"    payload: {desc}\n")
            if path:
                f.write(f"    file   : {path}\n")
            f.write("\n")
    print(f"\nManifest written to {manifest_path}")
    n_rehost = sum(1 for m in manifest if m[2].startswith("RE-HOST"))
    print(f"{n_rehost} dataset(s) need re-hosting (were DataGeometry pickles).")


if __name__ == "__main__":
    main()
