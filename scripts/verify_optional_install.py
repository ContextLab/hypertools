"""Verify missing-extra errors and actual installation in an isolated venv.

Usage: .venv/bin/python scripts/verify_optional_install.py --output /tmp/optional-check
Installs the current source with base dependencies into a NEW temporary venv;
never uninstalls or changes packages in the caller's interpreter.
"""

import argparse
import json
import os
from pathlib import Path
import subprocess
import tempfile
import venv


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    repo = Path(__file__).resolve().parents[1]
    with tempfile.TemporaryDirectory(prefix="hypertools-optional-env-") as directory:
        root = Path(directory)
        venv.EnvBuilder(with_pip=True).create(root)
        python = root / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        with (args.output / "install.log").open("w") as log:
            subprocess.run(
                [str(python), "-m", "pip", "install", str(repo)],
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
            )
        source = """import importlib.util, json
from pathlib import Path
from importlib.metadata import version
import numpy as np
import hypertools as hyp
assert importlib.util.find_spec('openpyxl') is None, 'Expected a base-only environment'
path=Path('roundtrip.xlsx')
with hyp.set_autoinstall(False):
    try: hyp.save(np.arange(12).reshape(4,3),str(path))
    except hyp.HypertoolsIOError as exc:
        assert isinstance(exc.__cause__, ImportError)
        assert 'hypertools[io]' in str(exc), str(exc)
        print('DISABLED:',str(exc))
    else: raise AssertionError('Missing extra unexpectedly succeeded')
assert importlib.util.find_spec('openpyxl') is None
with hyp.set_autoinstall(True):
    hyp.save(np.arange(12).reshape(4,3),str(path))
    restored=hyp.load(str(path))
np.testing.assert_array_equal(restored,np.arange(12).reshape(4,3))
print(json.dumps({'status':'PASS','hypertools':hyp.__version__,'openpyxl':version('openpyxl'),
                  'roundtrip_shape':list(restored.shape)}))
"""
        with (args.output / "verification.log").open("w") as log:
            subprocess.run(
                [str(python), "-c", source],
                cwd=root,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
            )
        (args.output / "result.json").write_text(
            json.dumps(
                {
                    "status": "PASS",
                    "source": str(repo),
                    "git_commit": subprocess.check_output(
                        ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
                    ).strip(),
                    "source_diff_sha256": __import__("hashlib")
                    .sha256(
                        subprocess.check_output(
                            ["git", "-C", str(repo), "diff", "HEAD"]
                        )
                    )
                    .hexdigest(),
                },
                indent=2,
            )
        )
    print("PASS:", args.output)


if __name__ == "__main__":
    main()
