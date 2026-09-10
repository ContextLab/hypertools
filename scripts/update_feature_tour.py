"""Synchronize the local review notebook's helpers and content after review.

This is a development utility, never a publisher. The tour lives in notes/colab.
Run from the repository root. Executed outputs are invalidated on source edits.
"""

import ast
import hashlib
import json
import pprint
import re
from pathlib import Path

PATH = Path("notes/colab/hypertools_1.1_feature_tour.ipynb")
nb = json.loads(PATH.read_text())


def text(cell):
    return "".join(cell["source"])


def set_source(cell, source):
    if text(cell) != source:
        cell["source"] = source.splitlines(keepends=True)
        if cell["cell_type"] == "code":
            cell["outputs"] = []
            cell["execution_count"] = None


def code_cell(source):
    return dict(
        cell_type="code",
        metadata={},
        source=source.splitlines(keepends=True),
        outputs=[],
        execution_count=None,
    )


def markdown(source):
    return dict(
        cell_type="markdown", metadata={}, source=source.splitlines(keepends=True)
    )


def case_cell(case_id):
    return next(
        c
        for c in nb["cells"]
        if c["cell_type"] == "code" and f"run_case('{case_id}', demo)" in text(c)
    )


config = next(c for c in nb["cells"] if "REVIEW_COMMIT = " in text(c))
s = text(config)
if "if IN_COLAB:\n    spec =" in s:
    start = s.index("if IN_COLAB:\n    spec =")
    end = s.index("os.environ.setdefault('HF_HUB_DISABLE_PROGRESS_BARS'", start)
    install = s[start:end]
    set_source(config, s[:start] + s[end:])
    cell = code_cell(install)
    cell["metadata"]["tags"] = ["hypertools-install"]
    nb["cells"].insert(nb["cells"].index(config) + 1, cell)

setup = next(c for c in nb["cells"] if "STARTED = datetime.datetime" in text(c))
set_source(
    setup,
    re.sub(
        r"'status','--porcelain'(?:,'--untracked-files=no')*",
        "'status','--porcelain','--untracked-files=no'",
        text(setup),
    ),
)
s = text(setup)
if "'optional_versions'" not in s:
    s += """\noptional_packages = ['plotly','kaleido','Pillow','scipy','statsmodels','torch',
 'sentence-transformers','transformers','chronos-forecasting','gensim','skaters',
 'polars','pyarrow','ipywidgets','ipympl','pylsl','kagglehub','scikit-image','openpyxl']
ENVIRONMENT['optional_versions'] = {}
for package in optional_packages:
    try: ENVIRONMENT['optional_versions'][package] = metadata.version(package)
    except metadata.PackageNotFoundError: ENVIRONMENT['optional_versions'][package] = None
ENVIRONMENT['binaries'] = {name:shutil.which(name) for name in ['ffmpeg','ffprobe','google-chrome','chromium']}
ENVIRONMENT['run_id'] = hashlib.sha256(STARTED.encode()).hexdigest()[:16]
"""
    set_source(setup, s)
for install_cell in nb["cells"]:
    if "hypertools-install" in install_cell.get("metadata", {}).get("tags", []):
        set_source(
            install_cell,
            """# Preserve the full installer transcript even if setup fails.
from pathlib import Path
import tempfile
INSTALL_LOG = Path(tempfile.mkdtemp(prefix='hypertools-install-')) / 'install.log'
if IN_COLAB:
    spec = f'hypertools[{EXTRAS}] @ git+https://github.com/ContextLab/hypertools.git@{REVIEW_COMMIT}'
    command = [sys.executable, '-m', 'pip', 'install', spec,
               'pyarrow', 'polars', 'ipywidgets', 'xlrd', 'xlwt']
    with INSTALL_LOG.open('w') as log:
        process = subprocess.Popen(command, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True)
        for line in process.stdout:
            print(line, end='')
            log.write(line)
        status = process.wait()
    print('Retained installation log:', INSTALL_LOG)
    if status:
        from google.colab import files
        files.download(str(INSTALL_LOG))
        raise RuntimeError(f'Candidate installation failed ({status}); see {INSTALL_LOG}')
else:
    INSTALL_LOG.write_text('Local candidate verification: installer intentionally not run.\\n')
    print('Local mode: retaining the installed package / editable checkout.')
    print('Optional full environment:', f'python -m pip install -e ".[{EXTRAS}]" pyarrow polars ipywidgets xlrd xlwt')
""",
        )
helper = next(c for c in nb["cells"] if "def run_case(" in text(c))
set_source(helper, Path("scripts/feature_tour_support.py").read_text())

inventory = next(
    c for c in nb["cells"] if text(c).startswith("# Explicit coverage inventory")
)
cases = ast.literal_eval(ast.parse(text(inventory)).body[0].value)
for entry in cases:
    if entry["id"].startswith("EXPORT-anim-") and entry["id"].split("-")[2] in (
        "mp4",
        "mov",
        "avi",
        "webm",
        "m4v",
        "mkv",
    ):
        entry["binaries"] = ["ffmpeg"]
    aliases = {
        "AL-SRM": "SharedResponseModel",
        "AL-DetSRM": "DeterministicSharedResponseModel",
        "AL-RSRM": "RobustSharedResponseModel",
    }
    if (
        entry["id"] in aliases
        and "align:" + aliases[entry["id"]] not in entry["covers"]
    ):
        entry["covers"].append("align:" + aliases[entry["id"]])
    if entry["id"].startswith("PLOT-1d") or "separate feature lines" in entry["visual"]:
        entry["visual"] = entry["visual"].replace(
            "separate feature lines", "one reduced signal"
        )
    if entry["id"] == "ANIM-companion":
        entry["visual"] = (
            "Both markers and the date advance together. The right curve stays fully visible; black is a trailing three-sample mean."
        )

# Real export demonstrations: at least twelve frames, unique destination on rerun.
for cell in nb["cells"]:
    if cell["cell_type"] != "code":
        continue
    s = text(cell)
    s = s.replace("display(FileLink(str(path)))", "download_artifact(path)")
    if "run_case('EXPORT-" in s and "path=SCRATCH/" in s:
        s = re.sub(
            r"path=SCRATCH/'([^']+)'", r"path=SCRATCH/(uuid.uuid4().hex+'-\1')", s
        )
        if "run_case('EXPORT-anim-" in s:
            s = s.replace(
                "duration=.5,frame_rate=4", "duration=2,frame_rate=6"
            ).replace("fps=4", "fps=6")
            # Resolve generated literal branches while retaining the real backend path.
            s = re.sub(
                r"    if 'matplotlib'=='matplotlib':(.*?)\n    else:.*?\n                  show=False,save_path=str\(path\)\)",
                r"    \1",
                s,
                flags=re.S,
            )
            s = re.sub(
                r"    if 'plotly'=='matplotlib':.*?\n    else:", r"    ", s, flags=re.S
            )
            s = s.replace(
                "    assert path.is_file() and path.stat().st_size>100\n    download_artifact(path);print(path.stat().st_size,'bytes')",
                "    verify_export(path, animated=True)",
            )
        else:
            s = s.replace(
                "    assert path.is_file() and path.stat().st_size>100\n    download_artifact(path);print(path.stat().st_size,'bytes')",
                "    verify_export(path)",
            )
    s = re.sub(
        r"HashingVectorizer\(n_features=128,alternate_sign=False\) if '([^']+)'=='HashingVectorizer' else '[^']+'",
        lambda m: (
            "HashingVectorizer(n_features=128,alternate_sign=False)"
            if m[1] == "HashingVectorizer"
            else repr(m[1])
        ),
        s,
    )
    set_source(cell, s)

set_source(
    case_cell("ANIM-companion"),
    """def demo():
    a=hyp.load('helix',n_samples=30,random_state=0)
    data=pd.DataFrame(a,index=pd.date_range('2026-01-01',periods=30))
    contexts=[]
    obj=hyp.plot(data,backend='matplotlib',animate=True,duration=5,frame_rate=6,fmt='o-',markersize=4,
        companion=[{'data':a[:,0],'smooth':3,'position':'bottom','xlabel':'Sample','ylabel':'Helix x; black: trailing mean'},
                   {'data':a[:,1],'position':'right','reveal':False,'xlabel':'Sample','ylabel':'Helix y'}],
        title='{index:%Y-%m-%d}',on_frame=contexts.append,show=False)
    assert len(obj.figure.axes)==3
    for i in range(obj.n_frames):
        obj.draw_frame(i)
        for ax in obj.figure.axes[1:]:
            assert list(ax.lines[-1].get_xdata())==[i]
        assert obj.figure.axes[0].get_title()==data.index[i].strftime('%Y-%m-%d')
        assert contexts[-1].revealed_counts==(i+1,)
    show_result(obj)

run_case('ANIM-companion', demo)""",
)

# Native GUI must live in its own process, outside the inline notebook backend.
set_source(
    case_cell("GUI-native"),
    """def demo():
    if IN_COLAB: raise RuntimeError('Native desktop GUI is unavailable in Colab.')
    script = SCRATCH/'native_gui_check.py'
    script.write_text("import matplotlib\\nmatplotlib.use('QtAgg')\\nimport hypertools as hyp\\nimport matplotlib.pyplot as plt\\nhyp.set_interactive_backend('QtAgg')\\nhyp.plot(hyp.load('helix'), backend='matplotlib', interactive=True, explore=True, show=False)\\nplt.show(block=True)\\n")
    process = subprocess.Popen([sys.executable,str(script)])
    time.sleep(2)
    assert process.poll() is None, 'Native window process failed; install a Qt binding and inspect its error.'
    print('Separate native window remains open for inspection. Close it manually after testing hover/rotation/zoom.')

run_case('GUI-native', demo)""",
)

# No implicit trust of ad-hoc remote pickles in Run all. Require an explicitly
# vetted content digest; raw bytes are verified before any deserialization.
for case_id, source in [
    ("SOURCE-drive", "1nHAusn2VsQinJk35xvJSd7CtWPC1uOwK"),
    ("SOURCE-dropbox", "https://www.dropbox.com/s/7d9vo9idqk1hn31/bunny.pkl?dl=0"),
]:
    cell = case_cell(case_id)
    # Existing vetted built-in bunny has the same resolver routes; preserve
    # actual connector coverage via a safe CSV URL for Dropbox separately.
    # Until a digest is vetted, the dangerous calls must be explicit skips.
    for e in cases:
        if e["id"] == case_id:
            e["gate"] = "trusted_remote_pickle"
    s = text(cell)
    if "Explicitly authorized" not in s:
        s = s.replace(
            "def demo():",
            'def demo():\n    print("Explicitly authorized remote pickle: loading can execute code from the source owner.")',
        )
        set_source(cell, s)
if "'trusted_remote_pickle'" not in text(config):
    set_source(
        config,
        text(config).replace(
            "'native_gui': False,",
            "'trusted_remote_pickle': False, # opt in only after vetting Drive/Dropbox pickle owners\n    'native_gui': False,",
        ),
    )


# Additional behavior demos identified by the reviewers.
def add(
    case_id, title, source, covers, visual="", gate=None, requires=(), backend=None
):
    if any(c["id"] == case_id for c in cases):
        set_source(
            case_cell(case_id), source.strip() + f"\n\nrun_case('{case_id}', demo)"
        )
        return
    cases.insert(
        -1,
        dict(
            id=case_id,
            title=title,
            covers=covers,
            gate=gate,
            backend=backend,
            requires=list(requires),
            visual=visual,
        ),
    )
    anchor = nb["cells"].index(case_cell("COVERAGE")) - 1
    nb["cells"][anchor:anchor] = [
        markdown(
            f'<a id="case-{case_id}"></a>\n\n### {case_id} · {title}\n\n'
            + ("**Inspect:** " + visual + "\n" if visual else "")
        ),
        code_cell(source.strip() + f"\n\nrun_case('{case_id}', demo)"),
    ]


for backend in ("matplotlib", "plotly"):
    add(
        "PLOT-series-" + backend,
        "All input columns as time series and datetime limits",
        f"""def demo():
    a,_=fixtures()
    data=pd.DataFrame(a[:20,:3],index=pd.date_range('2026-01-01',periods=20))
    obj=hyp.plot(data,ndims=1,reduce=None,backend='{backend}',xlim=(data.index[2],data.index[-3]),show=False,return_model=True)
    assert len(obj['trace_data'])==3
    show_result(obj)""",
        ["plot:ndims", "plot:xlim", "behavior:multicolumn-series"],
        "Three feature curves, with the displayed date range limited at both ends.",
        requires=["plotly"] if backend == "plotly" else (),
        backend=backend,
    )
    add(
        "PLOT-bundle-" + backend,
        "Returned coordinates, colors and observation ownership",
        f"""def demo():
    a,b=fixtures()
    bundle=hyp.plot([a,b],backend='{backend}',hue=['A']*len(a)+['B']*len(b),return_model=True,show=False)
    assert set(bundle)>={{'fig','xform_data','trace_data','trace_metadata','colors','pipeline'}}
    assert len(bundle['trace_data'])==2
    display({{key:type(value).__name__ for key,value in bundle.items()}})
    print('Colors:',bundle['colors']);print('Trace metadata:',bundle['trace_metadata'])
    show_result(bundle)""",
        ["behavior:trace-bundle", "plot:return_model"],
        "Returned color mapping agrees with the two visible groups.",
        requires=["plotly"] if backend == "plotly" else (),
        backend=backend,
    )

add(
    "PIPE-raw-fitted",
    "Reuse a fitted raw sklearn stage without refitting",
    """def demo():
    from sklearn.decomposition import PCA
    a,b=fixtures(); pca=PCA(n_components=3).fit(a); before=pca.components_.copy()
    pipe=hyp.Pipeline([('reduce',pca)])
    bundle=hyp.plot(b,pipeline=pipe,show=False,return_model=True)
    np.testing.assert_allclose(bundle['xform_data'][0],pca.transform(b))
    np.testing.assert_array_equal(pca.components_,before)
    show_result(bundle)""",
    ["plot:pipeline", "behavior:fitted-sklearn"],
    visual="Held-out data projected through the fitted training transform.",
)

add(
    "ANIM-clock",
    "Actual reveal bounds and callable titles across animation modes",
    """def demo():
    a=hyp.load('helix',n_samples=12)
    for mode in ['parallel','window','spin']:
        contexts=[]
        obj=hyp.plot(a,animate=mode,duration=2,frame_rate=6,show=False,
                     title=lambda ctx:f"{ctx.style}: {ctx.progress:.0%}",on_frame=contexts.append)
        for frame in range(obj.n_frames):
            obj.draw_frame(frame);ctx=contexts[-1]
            assert ctx.progress==frame/(obj.n_frames-1)
            if mode!='spin':
                for artist,(start,end),data in zip(ctx.artists,ctx.window_bounds,ctx.datasets):
                    xyz=np.column_stack(artist.get_data_3d())
                    if len(xyz):np.testing.assert_allclose(xyz[-1],data[end-1])
        show_result(obj)""",
    ["behavior:frame-context", "plot:title", "plot:on_frame"],
    visual="Each title runs from 0% to 100%; parallel/window reveal and spin rotates.",
)

add(
    "IO-passthrough",
    "Already-loaded inputs and unknown-argument rejection",
    """def demo():
    a,_=fixtures();df=pd.DataFrame(a)
    np.testing.assert_allclose(hyp.load(df),df)
    np.testing.assert_allclose(hyp.load(a),a)
    expect_error(TypeError,lambda:hyp.load('helix',definitely_not_an_option=True),contains='definitely_not_an_option')""",
    ["behavior:load-passthrough", "behavior:load-invalid-kwargs"],
)

for model in ("wiki_model", "nips_model", "sotus_model"):
    add(
        "IO-model-" + model,
        "Hosted fitted topic pipeline: " + model,
        f"""def demo():
    from sklearn.pipeline import Pipeline
    model=hyp.load('{model}')
    assert isinstance(model,Pipeline)
    output=finite(model.transform(TEXTS[:2]),shape=(2,50))
    assert 'steps' in model.get_params()
    print(model);display(pd.DataFrame(output))""",
        ["io:hosted-models"],
        gate="large_data",
    )

add(
    "MAN-warmup",
    "Trailing smoothing warm-up versus partial windows",
    """def demo():
    data=pd.DataFrame({'signal':np.arange(8,dtype=float)})
    full=hyp.manip(data,model='Smooth',kernel='boxcar',kernel_width=3,center=False,maintain_bounds=False)
    partial=hyp.manip(data,model='Smooth',kernel='boxcar',kernel_width=3,center=False,maintain_bounds=False,min_periods=1)
    assert np.isnan(np.asarray(full)[:2]).all()
    np.testing.assert_allclose(np.asarray(partial).ravel(),data.signal.rolling(3,min_periods=1).mean())
    display(pd.concat({'full_window':full,'partial_window':partial},axis=1))""",
    ["behavior:smoothing-warmup"],
)

# Review gaps: explicit behavior checks, not just exported-name coverage.
add(
    "COLOR-luminance",
    "Image palette luminance controls",
    """def demo():
    from PIL import Image as PILImage
    from hypertools.plot.colors import image_palette,luminance
    pixels=np.zeros((20,60,3),dtype=np.uint8)
    pixels[:,:40]=[250,250,230];pixels[:,40:]=[20,60,100]
    path=SCRATCH/'luminance.png';PILImage.fromarray(pixels).save(path)
    colors=image_palette(path,n_colors=2,max_luminance=.5)
    assert np.max(np.atleast_1d(luminance(colors)))<=.5
    display(PILImage.open(path));print('Selected dark palette:',colors)""",
    ["behavior:image-luminance"],
    visual="Bright background is excluded from the selected palette.",
)

add(
    "PLOT-cjk",
    "Multibyte titles, labels and legends",
    """def demo():
    a,_=fixtures()
    obj=hyp.plot(a,title='日本語 · Ελληνικά · café',legend=['測定'],xlabel='時間',show=False)
    # Rasterization is required: glyph warnings may be delayed until draw.
    obj.canvas.draw()
    show_result(obj)""",
    ["behavior:multibyte-fonts"],
    visual="Japanese, Greek and accented characters render without missing-glyph boxes. Check recorded font warnings.",
)

add(
    "PLOT-panel-bundle",
    "Panel return values and independent fitted models",
    """def demo():
    a,b=fixtures()
    bundle=hyp.plot([a,b],panels=True,return_model=True,show=False)
    assert bundle['panels']==(1,2)
    assert len(bundle['axes'])==len(bundle['panel_models'])==2
    assert all(ax.figure is bundle['fig'] for ax in bundle['axes'])
    assert bundle['colors'] is not None
    print('Bundle:',list(bundle));show_result(bundle)""",
    ["behavior:panel-bundle"],
    visual="Both panel axes belong to the returned figure; each model corresponds to its dataset.",
)

add(
    "PLOT-hierarchy-metadata",
    "Hierarchy trace ownership",
    """def demo():
    a,b=fixtures()
    columns=pd.MultiIndex.from_product([['group'],['left','right'],['x','y','z']],names=['root','subject','feature'])
    data=pd.DataFrame(a,columns=columns)
    bundle=hyp.plot(data,return_model=True,show=False)
    assert bundle['trace_metadata'] is not None
    assert len(bundle['trace_metadata']['keys'])==len(bundle['trace_data'])
    display(bundle['trace_metadata']);show_result(bundle)""",
    ["behavior:trace-metadata"],
    visual="Trace keys identify the hierarchy curves shown.",
)

add(
    "IO-sniff-compressed",
    "Compressed CSV and extensionless content detection",
    """def demo():
    import gzip
    frame=pd.DataFrame({'x':[1,2,3],'y':[4,5,6]})
    compressed=SCRATCH/'sample.csv.gz'
    compressed.write_bytes(gzip.compress(frame.to_csv(index=False).encode()))
    pd.testing.assert_frame_equal(hyp.load(str(compressed)),frame)
    plain=SCRATCH/'extensionless';plain.write_text(frame.to_csv(index=False))
    pd.testing.assert_frame_equal(hyp.load(str(plain)),frame)""",
    ["behavior:compressed-load", "behavior:extensionless-load"],
)

for cell in nb["cells"]:
    if cell["cell_type"] != "code":
        continue
    s = text(cell)
    match = re.search(r"run_case\('DATA-([^']+)'", s)
    if match and "assert_hosted_contract" not in s:
        s = s.replace(
            "    assert len(data)>0", f"    assert_hosted_contract('{match[1]}',data)"
        )
        set_source(cell, s)
stream = case_cell("STREAM-03")
s = text(stream).replace(
    "path=SCRATCH/'stream.gif'", "path=SCRATCH/(uuid.uuid4().hex+'-stream.gif')"
)
s = s.replace(
    "    assert path.stat().st_size>100", "    verify_export(path,animated=True)"
)
set_source(stream, s)
policy = case_cell("API-policy")
s = text(policy).replace(
    "    with hyp.set_autoinstall(False):\n        assert callable(hyp.reduce)",
    """    from hypertools._shared.lazy_import import auto_install_enabled
    before=auto_install_enabled()
    with hyp.set_autoinstall(False):
        assert auto_install_enabled() is False
    assert auto_install_enabled()==before
    print('Real missing-extra installation is checked separately with scripts/verify_optional_install.py in an isolated environment.')""",
)
set_source(policy, s)
errors = case_cell("ERR-input")
s = text(errors).replace(
    "expect_error((ValueError,TypeError),lambda:hyp.plot(a,group=[0]*len(a),show=False))",
    "expect_error(TypeError,lambda:hyp.plot(a,group=[0]*len(a),show=False),contains='hue=')",
)
s = s.replace(
    "expect_error((ValueError,TypeError),lambda:hyp.align([a,a],align=True))",
    "expect_error(TypeError,lambda:hyp.align([a,a],align=True),contains='model=')",
)
s = s.replace(
    "expect_error(TypeError,lambda:hyp.align",
    "expect_error(ValueError,lambda:hyp.align",
)
set_source(errors, s)

add(
    "IO-legacy-xls",
    "Read a real legacy binary spreadsheet",
    """def demo():
    import xlwt
    workbook=xlwt.Workbook();sheet=workbook.add_sheet('observations')
    for row,values in enumerate([['time','value'],[1,2],[3,4]]):
        for col,value in enumerate(values):sheet.write(row,col,value)
    path=SCRATCH/'legacy.xls';workbook.save(str(path))
    result=hyp.load(str(path))
    pd.testing.assert_frame_equal(result,pd.DataFrame({'time':[1,3],'value':[2,4]}))
    display(result)""",
    ["behavior:legacy-xls"],
    requires=["xlrd", "xlwt"],
)

# Registry guards count declarations separately from successful executions.
coverage = case_cell("COVERAGE")
s = text(coverage)
if "AUTOENCODER_NAMES" not in s:
    s = s.replace(
        "    missing=sorted(expected-planned)",
        """    from hypertools.manip.manip import MANIPULATORS
    from hypertools.align.align import ALIGNERS
    from hypertools.impute.impute import IMPUTERS
    from hypertools.predict.predict import FORECASTERS
    from hypertools.reduce.common import AUTOENCODER_NAMES
    for prefix,models in [('manip',MANIPULATORS),('align',ALIGNERS),('impute',IMPUTERS),('predict',FORECASTERS)]:
        expected|={prefix+':'+model.__name__ for model in models}
    expected|={'model:'+name for name in AUTOENCODER_NAMES}
    successful={feature for row in RESULTS.values() if row['status']=='PASS' for feature in row['covers']}
    print('Declared but not backed by a successful case:',sorted(expected-successful))
    missing=sorted(expected-planned)""",
    )
    set_source(coverage, s)

summary = next(c for c in nb["cells"] if text(c).startswith("summary=pd.DataFrame"))
set_source(
    summary,
    """summary=pd.DataFrame(report_rows())
display(summary[['id','title','status','visual','seconds','detail','visual_notes']])
print('Automatic counts:',summary.status.value_counts().to_dict())
print('Visual counts:',summary.visual.value_counts().to_dict())
print('Not run:',sorted({c['id'] for c in CASES}-set(RESULTS)))
attention=summary[(summary.status!='PASS') | (summary.visual.isin(['fail','not reviewed']))]
display(attention[['id','status','visual','detail','visual_notes']])
save_report()
print('Use one interactive viewer at a time; Close disposes its browser resources.')
interactive_viewer()""",
)
comparison = next(c for c in nb["cells"] if text(c).startswith("OTHER_RESULTS ="))
set_source(
    comparison,
    """OTHER_RESULTS = '' # downloaded results.json from the other environment
if OTHER_RESULTS:
    other=json.loads(Path(OTHER_RESULTS).read_text())
    print('Other environment:',other['environment'])
    print('This environment:',ENVIRONMENT)
    print('Same notebook source:',other.get('notebook_source_sha256')==NOTEBOOK_SOURCE_SHA256)
    print('Same inventory:',other.get('inventory_sha256')==source_hash(json.dumps(CASES,sort_keys=True)))
    columns=['id','status','visual','visual_notes','source_sha256','warnings']
    left=pd.DataFrame(report_rows(other)).reindex(columns=columns).set_index('id').map(str)
    right=pd.DataFrame(report_rows()).reindex(columns=columns).set_index('id').map(str)
    left,right=left.align(right,join='outer',fill_value='NOT RUN')
    display(left.compare(right,result_names=('other','this run')))
else:
    print('Set OTHER_RESULTS to compare automatic/visual outcomes, notes, case sources and warnings.')
save_report()""",
)
for c in nb["cells"]:
    if c["cell_type"] == "code" and "for path in [SCRATCH/" in text(c):
        set_source(
            c,
            text(c).replace("display(FileLink(str(path)))", "download_artifact(path)"),
        )
    if c["cell_type"] == "markdown":
        s = text(c).replace(
            "The lower panel’s head follows the trajectory and date title; the right panel remains fully visible.",
            "Both markers follow the trajectory and date title; the right curve remains fully visible. Black is a trailing mean.",
        )
        set_source(c, s)
set_source(
    inventory,
    "# Explicit coverage inventory; declarations are not passes.\nCASES = "
    + pprint.pformat(cases, width=100, sort_dicts=False),
)


# Remove generated constant branches while retaining real runtime conditions.
class LiteralBranches(ast.NodeTransformer):
    changed = False

    def visit_If(self, node):
        node = self.generic_visit(node)
        test = node.test
        if isinstance(test, ast.Compare) and len(test.ops) == 1:
            try:
                left = ast.literal_eval(test.left)
                right = ast.literal_eval(test.comparators[0])
            except (ValueError, TypeError):
                return node
            op = test.ops[0]
            if isinstance(op, ast.Eq):
                result = left == right
            elif isinstance(op, ast.In):
                result = left in right
            else:
                return node
            self.changed = True
            return node.body if result else node.orelse
        return node


for cell in nb["cells"]:
    if cell["cell_type"] == "code" and "run_case('" in text(cell):
        visitor = LiteralBranches()
        tree = visitor.visit(ast.parse(text(cell)))
        if visitor.changed:
            set_source(cell, ast.unparse(tree))
    elif cell["cell_type"] == "markdown":
        match = re.search(r"### ([A-Za-z0-9_-]+) ·", text(cell))
        entry = next((e for e in cases if match and e["id"] == match[1]), None)
        if entry and entry["visual"]:
            set_source(
                cell,
                re.sub(
                    r"\*\*Inspect:\*\* [^\n]*",
                    "**Inspect:** " + entry["visual"],
                    text(cell),
                ),
            )

# Keep the illustrative reducers well-conditioned; retain expected time/model warnings.
for case_id, old, new in [
    ("RED-NMF", "'max_iter': 30", "'max_iter': 1000"),
    (
        "RED-SpectralEmbedding",
        "'random_state': 0",
        "'random_state': 0, 'n_neighbors': 24",
    ),
    ("RED-UMAP", "'n_neighbors': 8", "'n_neighbors': 8, 'n_jobs': 1"),
]:
    cell = case_cell(case_id)
    if new not in text(cell):
        set_source(cell, text(cell).replace(old, new))
mds = case_cell("RED-MDS")
if "inspect.signature(MDS)" not in text(mds):
    set_source(
        mds,
        """def demo():
    from sklearn.manifold import MDS
    a,_=fixtures()
    kwargs={'random_state':0,'max_iter':300,'n_init':1}
    if 'init' in inspect.signature(MDS).parameters:kwargs['init']='random'
    out=hyp.reduce(a,reduce={'model':'MDS','kwargs':kwargs},ndims=2)
    finite(out,(48,2));print('MDS',np.shape(out))

run_case('RED-MDS', demo)""",
    )

# Mixture weights encode continuous blends, so there is no categorical legend.
for backend in ["matplotlib", "plotly"]:
    cell = case_cell("HIER-mixture-" + backend)
    set_source(
        cell,
        text(cell).replace(
            "legend=True,title='Means blend", "legend=False,title='Means blend"
        ),
    )

# Hash normalized sources, excluding the hash declaration itself and all outputs.
setup_source = re.sub(r"\nNOTEBOOK_SOURCE_SHA256 = '[^']*'", "", text(setup)).rstrip()
digest = hashlib.sha256(
    json.dumps(
        [
            (c["cell_type"], setup_source if c is setup else text(c))
            for c in nb["cells"]
        ],
        ensure_ascii=False,
    ).encode()
).hexdigest()
set_source(setup, setup_source + f"\nNOTEBOOK_SOURCE_SHA256 = '{digest}'\n")
for index, cell in enumerate(nb["cells"]):
    cell.setdefault(
        "id", hashlib.sha256((str(index) + text(cell)).encode()).hexdigest()[:12]
    )
PATH.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n")
print(PATH, len(cases), "cases; source", digest)
