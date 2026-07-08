import os, sys, traceback
os.environ['MPLBACKEND'] = 'Agg'
import warnings
warnings.simplefilter('ignore')
import numpy as np
import hypertools as hyp

rng = np.random.RandomState(0)

def mk(n=40, d=6, shift=0.0):
    return rng.randn(n, d) + shift

def hdr(t):
    print("\n" + "="*72)
    print(t)
    print("="*72)

def show(label, fn):
    try:
        out = fn()
        print(f"[OK]   {label}")
        if isinstance(out, tuple):
            out = out[0]
        arr = np.asarray(out if not isinstance(out, list) else out[0])
        print(f"       result type={type(out).__name__} shape/first={getattr(arr,'shape',None)} sample={arr.ravel()[:4]}")
        return out
    except Exception as e:
        print(f"[FAIL] {label}")
        tb = traceback.format_exc().strip().splitlines()
        # last frame + error
        for ln in tb[-6:]:
            print("       " + ln)
        return None

X = mk(shift=0.0)
Y = mk(shift=5.0)   # clearly different data for reuse detection
Xlist = [mk(shift=0.0), mk(shift=1.0)]
Ylist = [mk(shift=5.0), mk(shift=6.0)]

# ---- REDUCE reuse (bare Reducer from return_model=True) ----
hdr("1. reduce reuse: hyp.reduce(new, reduce=fitted_reducer)")
r_out, r_model = hyp.reduce(X, reduce='PCA', ndims=3, return_model=True)
print("fitted reducer type:", type(r_model).__name__, "is_fitted:", getattr(r_model,'is_fitted',None))
# refit-vs-reuse probe: transform Y with reused model, compare to fresh fit on Y
reuse = show("reduce(Y, reduce=fitted_reducer)", lambda: hyp.reduce(Y, reduce=r_model, ndims=3))
fresh = hyp.reduce(Y, reduce='PCA', ndims=3)
if reuse is not None:
    same = np.allclose(np.asarray(reuse), np.asarray(fresh))
    print(f"       reuse==freshfit? {same}  (False => genuinely reused fitted basis, correct)")

# ---- CLUSTER reuse: bare fitted sklearn estimator ----
hdr("2a. cluster reuse: hyp.cluster(new, cluster=<bare fitted sklearn estimator>)")
from sklearn.cluster import KMeans
km = KMeans(n_clusters=3, n_init=10).fit(np.vstack([X]))
show("cluster(Y, cluster=<fitted sklearn KMeans instance>)", lambda: hyp.cluster(Y, cluster=km))

# ---- CLUSTER reuse: fitted hyp Clusterer (single-stage return_model) ----
hdr("2b. cluster reuse: hyp.cluster(new, cluster=fitted_Clusterer) single-stage")
c_out, c_model = hyp.cluster(X, cluster='KMeans', n_clusters=3, return_model=True)
print("single-stage model type:", type(c_model).__name__, "is_fitted:", getattr(c_model,'is_fitted',None))
show("cluster(Y, cluster=fitted_Clusterer)", lambda: hyp.cluster(Y, cluster=c_model))

# ---- CLUSTER reuse: fitted hyp Pipeline (cross-module return_model) THE BUG ----
hdr("2c. cluster reuse: hyp.cluster(new, cluster=fitted_Pipeline) [Jeremy's bug]")
labels_cm, cluster_model = hyp.cluster(X, cluster='KMeans', n_clusters=3,
                                       reduce='PCA', ndims=3, manip='ZScore', return_model=True)
print("cross-module model type:", type(cluster_model).__name__)
# exact repro from the ticket:
show("cluster(Y, cluster=cluster_model, reduce='PCA', ndims=3, manip='ZScore')",
     lambda: hyp.cluster(Y, cluster=cluster_model, reduce='PCA', ndims=3, manip='ZScore'))
# also try passing the Pipeline as cluster= WITHOUT the extra stages
show("cluster(Y, cluster=cluster_model)  [pipeline as bare cluster spec]",
     lambda: hyp.cluster(Y, cluster=cluster_model))

# ---- ALIGN reuse ----
hdr("3. align reuse: hyp.align(new_list, model=fitted_aligner)")
a_out, a_model = hyp.align(Xlist, model='HyperAlign', return_model=True)
print("fitted aligner type:", type(a_model).__name__, "is_fitted:", getattr(a_model,'is_fitted',None))
show("align(Ylist, model=fitted_aligner)", lambda: hyp.align(Ylist, model=a_model))

# ---- MANIP reuse ----
hdr("4. manip reuse: hyp.manip(new, model=fitted_manip)")
m_out, m_model = hyp.manip(X, model='ZScore', return_model=True)
print("fitted manip type:", type(m_model).__name__, "is_fitted:", getattr(m_model,'is_fitted',None))
mreuse = show("manip(Y, model=fitted_manip)", lambda: hyp.manip(Y, model=m_model))
mfresh = hyp.manip(Y, model='ZScore')
if mreuse is not None:
    same = np.allclose(np.asarray(mreuse), np.asarray(mfresh))
    print(f"       reuse==freshfit? {same} (False => reused fit-time mean/std, correct)")

# ---- NORMALIZE reuse (P0-1 regression check) ----
hdr("5. normalize reuse: hyp.normalize(new, normalize=fitted_Normalizer)")
n_out, n_model = hyp.normalize(X, normalize='across', return_model=True)
print("fitted normalizer type:", type(n_model).__name__, "is_fitted:", getattr(n_model,'is_fitted',None))
nreuse = show("normalize(Y, normalize=fitted_Normalizer)", lambda: hyp.normalize(Y, normalize=n_model))
nfresh = hyp.normalize(Y, normalize='across')
if nreuse is not None:
    same = np.allclose(np.asarray(nreuse), np.asarray(nfresh))
    print(f"       reuse==freshfit? {same} (False => reused fit-time mean/std across, correct)")
# also .transform directly
show("fitted_Normalizer.transform(Y)", lambda: n_model.transform(Y))

# ---- CROSS-MODULE combos ----
hdr("6a. cross-module: cluster(new, cluster=fitted_bare_Clusterer, reduce='PCA', manip='ZScore')")
# single-stage fitted Clusterer reused but WITH cross-module stages this time
show("cluster(Y, cluster=<single-stage fitted Clusterer>, reduce='PCA', ndims=3, manip='ZScore')",
     lambda: hyp.cluster(Y, cluster=c_model, reduce='PCA', ndims=3, manip='ZScore'))

hdr("6b. cross-module: reduce(new, reduce=fitted_reducer, align='HyperAlign')")
show("reduce(Ylist, reduce=r_model, align='HyperAlign')",
     lambda: hyp.reduce(Ylist, reduce=r_model, align='HyperAlign'))

# reduce cross-module return_model gives a Pipeline; test reusing THAT
hdr("6c. reduce cross-module return_model=Pipeline, then reuse")
rp_out, rp_model = hyp.reduce(X, reduce='PCA', ndims=3, manip='ZScore', return_model=True)
print("reduce cross-module model type:", type(rp_model).__name__)
show("reduce(Y, reduce=rp_model_Pipeline, manip='ZScore', ndims=3)",
     lambda: hyp.reduce(Y, reduce=rp_model, ndims=3, manip='ZScore'))
show("reduce(Y, reduce=rp_model_Pipeline)  [pipeline as bare reduce spec]",
     lambda: hyp.reduce(Y, reduce=rp_model, ndims=3))

# align cross-module return_model gives Pipeline; reuse
hdr("6d. align cross-module return_model=Pipeline, then reuse")
ap_out, ap_model = hyp.align(Xlist, model='HyperAlign', reduce='PCA', ndims=3, return_model=True)
print("align cross-module model type:", type(ap_model).__name__)
show("align(Ylist, model=ap_model_Pipeline, reduce='PCA', ndims=3)",
     lambda: hyp.align(Ylist, model=ap_model, reduce='PCA', ndims=3))

# manip Pipeline reuse (list spec -> Pipeline)
hdr("6e. manip list-spec return_model=Pipeline, then reuse")
mp_out, mp_model = hyp.manip(X, model=['ZScore', {'model':'Smooth','kwargs':{'kernel_width':5}}], return_model=True)
print("manip list model type:", type(mp_model).__name__)
show("manip(Y, model=mp_model_Pipeline)", lambda: hyp.manip(Y, model=mp_model))

print("\nDONE")
