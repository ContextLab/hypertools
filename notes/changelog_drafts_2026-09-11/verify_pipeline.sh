#!/bin/zsh
# Release verification pipeline (2026-09-11 final review). Run from repo root
# with nohup; poll LOG for step markers and the final DONE line.
set -u
REPO=/Users/jmanning/hypertools
SP=/private/tmp/claude-501/-Users-jmanning-hypertools/f69d921d-f10e-4fa8-aafb-f55b01cff0f9/scratchpad
LOG=$SP/verify.log
PY=$REPO/.venv/bin/python
export PLOTLY_RENDERER=json MPLBACKEND=Agg
cd $REPO
: > $LOG
step() { printf '%s\n' "=== STEP $1 $(date -u +%H:%M:%S) ===" >> $LOG; }
step "0 head $(git rev-parse --short HEAD)"
$PY -m pip install -q --no-deps -e . >> $LOG 2>&1

step "1 tutorials"
for nb in docs/tutorials/*.ipynb; do
  printf '%s\n' "--- $nb" >> $LOG
  $PY scripts/execute_tutorial.py "$nb" >> $LOG 2>&1 || printf '%s\n' "TUTORIAL-FAILED $nb" >> $LOG
done

step "2 pytest"
$PY -m pytest -q -p no:cacheprovider -o addopts="-m 'not bigdata'" -rfE > $SP/verify_pytest.log 2>&1
printf '%s\n' "pytest rc=$? $(tail -1 $SP/verify_pytest.log)" >> $LOG

step "3 sphinx html"
rm -rf docs/auto_examples docs/_build
(cd docs && $PY -m sphinx -b html -W -E -a . _build/html > $SP/verify_sphinx.log 2>&1)
printf '%s\n' "sphinx html rc=$? warnings=$(grep -c 'WARNING' $SP/verify_sphinx.log)" >> $LOG

step "4 sphinx doctest"
(cd docs && HYPERTOOLS_DOCS_PLOT_GALLERY=0 $PY -m sphinx -b doctest -W . _build/doctest > $SP/verify_doctest.log 2>&1)
printf '%s\n' "sphinx doctest rc=$? $(grep -E 'passed|failed' $SP/verify_doctest.log | tail -2 | tr '\n' ' ')" >> $LOG
git checkout -- docs/hypertools.FrameContext.rst docs/hypertools.io.LSLStream.rst 2>/dev/null

step "5 gallery thumbs"
$PY scripts/generate_gallery_thumbs.py >> $LOG 2>&1
printf '%s\n' "thumbs rc=$?" >> $LOG

step "6 example smoke"
HYPERTOOLS_EXAMPLE_SMOKE=1 $PY -m pytest -q -p no:cacheprovider tests/test_examples_are_native.py -k end_to_end > $SP/verify_smoke.log 2>&1
printf '%s\n' "smoke rc=$? $(tail -1 $SP/verify_smoke.log)" >> $LOG

# tour runs separately, after the re-executed notebooks are committed
# (its SET-01 case requires a clean tracked checkout at REVIEW_COMMIT)

printf '%s\n' "DONE $(date -u +%H:%M:%S)" >> $LOG
