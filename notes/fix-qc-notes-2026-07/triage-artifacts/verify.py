import os
os.environ["MPLBACKEND"] = "Agg"
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import hypertools as hyp

OUT = "/private/tmp/claude-501/-Users-jmanning-hypertools/7e6531b3-066a-4ce2-b1f6-7c07c5e87b15/scratchpad/triage"
rng = np.random.default_rng(42)

def line(label, ok, extra=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {label}: {extra}")

# identity checks: public IS internal object?
from hypertools.reduce.reduce import reduce as _reduce
from hypertools.cluster.cluster import cluster as _cluster
from hypertools.reduce.describe import describe as _describe
from hypertools.predict.predict import predict as _predict
from hypertools.impute.impute import impute as _impute
from hypertools.manip.manip import manip as _manip
from hypertools.tools.normalize import normalize as _normalize
from hypertools.align.align import align as _align
from hypertools.tools.analyze import analyze as _analyze
from hypertools.core.model import apply_model as _apply_model
print("=== IDENTITY (public IS internal) ===")
for n, a, b in [("reduce",hyp.reduce,_reduce),("cluster",hyp.cluster,_cluster),
                ("describe",hyp.describe,_describe),("predict",hyp.predict,_predict),
                ("impute",hyp.impute,_impute),("manip",hyp.manip,_manip),
                ("normalize",hyp.normalize,_normalize),("align",hyp.align,_align),
                ("analyze",hyp.analyze,_analyze),("apply_model",hyp.apply_model,_apply_model)]:
    print(f"  hyp.{n} is internal: {a is b}")

print("\n=== PART B calls ===")
# reduce PCA
try:
    r = hyp.reduce(rng.normal(size=(80,10)), reduce='PCA', ndims=2)
    line("reduce PCA ndims=2", getattr(r,'shape',None)==(80,2), f"shape={getattr(r,'shape',None)}")
except Exception as e:
    line("reduce PCA", False, repr(e))
# reduce GaussianMixture
try:
    r = hyp.reduce(rng.normal(size=(80,10)), reduce='GaussianMixture', ndims=2)
    line("reduce GaussianMixture ndims=2", getattr(r,'shape',None) is not None, f"shape={getattr(r,'shape',None)}")
except Exception as e:
    line("reduce GaussianMixture", False, repr(e))
# cluster KMeans
Xc = rng.normal(size=(100,5))
try:
    c = hyp.cluster(Xc, cluster='KMeans', n_clusters=10)
    arr=np.asarray(c); line("cluster KMeans n_clusters=10", True, f"len={len(arr)} uniq={len(np.unique(arr))}")
except Exception as e:
    line("cluster KMeans", False, repr(e))
# cluster GaussianMixture
try:
    c = hyp.cluster(Xc, cluster='GaussianMixture', n_clusters=2)
    arr=np.asarray(c); line("cluster GaussianMixture n_clusters=2", True, f"len={len(arr)} uniq={len(np.unique(arr))}")
except Exception as e:
    line("cluster GaussianMixture", False, repr(e))
# describe show=False
Xd = rng.normal(size=(50,12))
try:
    d = hyp.describe(Xd, reduce='PCA', max_dims=10, show=False)
    line("describe show=False", isinstance(d,dict) and 'average' in d, f"keys={list(d.keys())} avg_len={len(d['average'])}")
except Exception as e:
    line("describe", False, repr(e))
# predict Kalman
try:
    p = hyp.predict(np.cumsum(rng.normal(size=(40,2)),axis=0), model='Kalman', t=10)
    line("predict Kalman t=10", True, f"type={type(p).__name__} shape={getattr(np.asarray(p),'shape',None)}")
except Exception as e:
    line("predict Kalman", False, repr(e))
# predict GP
try:
    p = hyp.predict(np.cumsum(rng.normal(size=(40,2)),axis=0), model='GP', t=10)
    line("predict GP t=10", True, f"shape={getattr(np.asarray(p),'shape',None)}")
except Exception as e:
    line("predict GP", False, repr(e))
# predict ARIMA
try:
    p = hyp.predict(np.cumsum(rng.normal(size=(40,2)),axis=0), model='ARIMA', t=10)
    line("predict ARIMA t=10", True, f"shape={getattr(np.asarray(p),'shape',None)}")
except Exception as e:
    line("predict ARIMA", False, repr(e))
# impute PPCA
Xi = rng.normal(size=(60,8))
mask = rng.random(Xi.shape) < 0.10
Xi_missing = Xi.copy(); Xi_missing[mask] = np.nan
nan_before = int(np.isnan(Xi_missing).sum())
try:
    imp = hyp.impute(Xi_missing, model='PPCA')
    nan_after = int(np.isnan(np.asarray(imp)).sum())
    line("impute PPCA", nan_after==0, f"nan_before={nan_before} nan_after={nan_after} shape={np.asarray(imp).shape}")
except Exception as e:
    line("impute PPCA", False, repr(e))
# manip
try:
    m = hyp.manip(rng.normal(size=(30,4)), model='ZScore')
    line("manip ZScore", True, f"shape={np.asarray(m).shape} mean~0={np.allclose(np.asarray(m).mean(0),0,atol=1e-6)}")
except Exception as e:
    line("manip ZScore", False, repr(e))
# normalize
try:
    nm = hyp.normalize(rng.normal(size=(30,4)))
    line("normalize", True, f"shape={np.asarray(nm).shape}")
except Exception as e:
    line("normalize", False, repr(e))
# align
try:
    al = hyp.align([rng.normal(size=(20,5)), rng.normal(size=(20,5))])
    line("align (2 datasets)", True, f"n={len(al)} shape0={np.asarray(al[0]).shape}")
except Exception as e:
    line("align", False, repr(e))
# analyze
try:
    an = hyp.analyze(rng.normal(size=(40,6)), reduce='PCA', ndims=2)
    line("analyze reduce=PCA ndims=2", True, f"shape={np.asarray(an).shape}")
except Exception as e:
    line("analyze", False, repr(e))
# apply_model
try:
    from sklearn.decomposition import PCA
    am = hyp.apply_model(rng.normal(size=(40,6)), PCA(n_components=2))
    line("apply_model PCA", True, f"shape={np.asarray(am).shape}")
except Exception as e:
    line("apply_model", False, repr(e))

print("\n=== describe show=True screenshot ===")
try:
    plt.close('all')
    d = hyp.describe(rng.normal(size=(50,12)), reduce='PCA', max_dims=10, show=True)
    fig = plt.gcf()
    fig.savefig(f"{OUT}/describe_current.png", dpi=110, bbox_inches='tight')
    ax = fig.axes[0] if fig.axes else None
    if ax:
        sp = {k: ax.spines[k].get_visible() for k in ('top','right','left','bottom')}
        print(f"  spines visibility: {sp}")
    print("  saved describe_current.png")
except Exception as e:
    print("  describe show=True FAILED:", repr(e))
