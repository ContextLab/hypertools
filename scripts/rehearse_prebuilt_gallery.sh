#!/usr/bin/env bash
# Rehearse the pre-built gallery sequence end to end against a LOCAL bare
# remote: build -> publish -> record -> commit -> fresh clone -> fetch ->
# Read the Docs recipe build -> overwrite branch -> fetch must reject.
#   scripts/rehearse_prebuilt_gallery.sh <source-repo> <workdir> [mini]
# "mini" keeps three fast examples so the whole thing takes a few minutes;
# without it the already-built docs/auto_examples of <source-repo> is used.
set -u
SRC=$1; R=$2; MODE=${3:-full}
PY=$SRC/.venv/bin/python
export MPLBACKEND=Agg GIT_AUTHOR_NAME=rehearsal GIT_AUTHOR_EMAIL=r@example.invalid GIT_COMMITTER_NAME=rehearsal GIT_COMMITTER_EMAIL=r@example.invalid
step() { echo; echo "== $*"; }
die() { echo "REHEARSAL FAILED: $*"; exit 1; }

rm -rf $R; mkdir -p $R
git init -q --bare $R/remote.git
step "work clone at $(git -C $SRC rev-parse --short HEAD) (+ uncommitted tracked changes)"
git clone -q $SRC $R/work || die clone
# carry the working tree's uncommitted tracked changes, then commit them
(cd $SRC && git diff HEAD --binary) > $R/wip.patch
if [ -s $R/wip.patch ]; then (cd $R/work && git apply $R/wip.patch && git commit -q -am "rehearsal: working tree") || die "apply wip"; fi

if [ "$MODE" = mini ]; then
  step "mini: keep three fast examples and build the gallery for real"
  (cd $R/work/examples && ls *.py | grep -vE '^(plot_basic|plot_2D|plot_impute)\.py$' | xargs git rm -q) || die "trim examples"
  (cd $R/work && git commit -q -m "rehearsal: three examples") || die commit
  (cd $R/work/docs && $PY -m sphinx -b html -d _build/doctrees . _build/html > $R/build_work.log 2>&1) || die "gallery build (see $R/build_work.log)"
else
  step "full: reuse the gallery built in $SRC"
  cp -R $SRC/docs/auto_examples $R/work/docs/auto_examples || die "copy gallery"
fi
BUILT_FROM=$(git -C $R/work rev-parse HEAD)
echo "gallery built from $BUILT_FROM; $(ls $R/work/docs/auto_examples/*.py.md5 | wc -l | tr -d ' ') md5 files"

step "publish to the bare remote, which writes the record"
(cd $R/work && $PY scripts/publish_prebuilt_gallery.py --push --remote $R/remote.git) || die publish
cat $R/work/docs/prebuilt_gallery.json
step "commit the record (the one commit allowed after the build)"
(cd $R/work && git add docs/prebuilt_gallery.json && git commit -q -m "release: record the pre-built gallery") || die "commit record"
RELEASE=$(git -C $R/work rev-parse HEAD)
echo "release commit $RELEASE; changed since the build: $(git -C $R/work diff --name-only $BUILT_FROM $RELEASE | tr '\n' ' ')"
[ "$(git -C $R/work diff --name-only $BUILT_FROM $RELEASE)" = "docs/prebuilt_gallery.json" ] || die "more than the record changed"

step "fresh clone of the release commit, as Read the Docs has; fetch"
git clone -q $R/work $R/fresh || die "fresh clone"
[ ! -e $R/fresh/docs/auto_examples ] || die "fresh clone already has a gallery"
(cd $R/fresh && $PY docs/fetch_prebuilt_gallery.py --require --remote $R/remote.git) || die "fetch --require"

step "Read the Docs recipe in the fresh clone"
START=$(date +%s)
(cd $R/fresh/docs && READTHEDOCS=True READTHEDOCS_GIT_IDENTIFIER=master READTHEDOCS_VERSION=latest READTHEDOCS_OUTPUT=$R/out \
  $PY -m sphinx -T -b html -d _build/doctrees -D language=en . $R/out/html > $R/build_fresh.log 2>&1) || die "sphinx (see $R/build_fresh.log)"
(cd $R/fresh && READTHEDOCS=True READTHEDOCS_GIT_IDENTIFIER=master READTHEDOCS_VERSION=latest READTHEDOCS_OUTPUT=$R/out $PY docs/post_build.py > $R/post_build.log 2>&1) || die "post_build"
echo "sphinx + post_build: $(( $(date +%s) - START )) s"
# an executed example rewrites its md5 and its page; an unexecuted one does not
CHANGED=$(cd $R/fresh/docs/auto_examples && for f in *.rst; do cmp -s $f $R/work/docs/auto_examples/$f || echo $f; done | grep -v '^sg_execution_times\|^index.rst' | tr '\n' ' ')
echo "gallery pages rewritten by this build: [${CHANGED}]"
grep -E 'computation time|Sphinx-Gallery successfully executed|executed [0-9]+ out of' $R/build_fresh.log | head -3
EXECUTED=$(grep -oE 'successfully executed [0-9]+ out of [0-9]+' $R/build_fresh.log | head -1)
echo "sphinx-gallery says: ${EXECUTED:-<no execution summary>}"
case "$EXECUTED" in *"executed 0 out of"*|"") ;; *) die "examples were executed: $EXECUTED";; esac
[ -z "$CHANGED" ] || die "pages were rewritten: $CHANGED"
N=$(ls $R/out/html/auto_examples/*.html | wc -l | tr -d ' '); echo "$N gallery html pages built"

step "overwrite the gallery branch; a fresh clone must not use it"
mkdir -p $R/evil/auto_examples && echo '.. include:: /etc/passwd' > $R/evil/auto_examples/plot_basic.rst && echo '{}' > $R/evil/manifest.json
(cd $R/evil && git init -q && git add -A && git commit -q -m evil && git push -q --force $R/remote.git HEAD:refs/heads/$(git -C $R/remote.git for-each-ref --format='%(refname:short)' | head -1)) || die "overwrite"
git clone -q $R/work $R/fresh2
(cd $R/fresh2 && $PY docs/fetch_prebuilt_gallery.py --remote $R/remote.git)
[ ! -e $R/fresh2/docs/auto_examples ] || die "the overwritten branch was used"
(cd $R/fresh2 && $PY docs/fetch_prebuilt_gallery.py --require --remote $R/remote.git > /dev/null); [ $? -eq 1 ] || die "--require should fail"

echo; echo "REHEARSAL PASSED ($MODE)"
