# Core plotting
hyp.plot on a single array renders a 3-D figure
hyp.plot on a list of arrays colors each dataset distinctly
hyp.plot with ndims=2 renders a clean 2-D figure
backend='plotly' produces an interactive figure matching the matplotlib one
fmt / marker / markers / color / colors / linestyle style kwargs all take effect
hue= colors points by a grouping variable; legend= and title= render correctly
labels= point annotations appear; label_alpha= changes their background opacity (#103)
xlabel= / ylabel= / zlabel= set axis titles; zlabel on a 2-D plot errors clearly (#154)
# Dimensionality reduction
hyp.reduce with PCA / IncrementalPCA / UMAP / TSNE returns the target ndims
mixture models as reducers (GaussianMixture) return (n, ndims) membership rows (#174)
autoencoder reducers (Autoencoder, VariationalAutoencoder) train and reduce (#162)
return_model=True gives back a fitted reducer that .transform()s new data without refitting
# Alignment
hyp.align with 'hyper' brings datasets into a shared space (correlation rises)
hyp.align with 'SRM' produces a sensible alignment (CONFIRM new single-fit semantics — task B1)
a fitted aligner .transform()s held-out data of the same shape; wrong shape errors clearly (#227)
# Clustering
hyp.cluster('KMeans', n_clusters=k) returns k discrete labels
mixture (soft) clustering colors points by blended membership
# Manip + chaining
hyp.manip single step (ZScore / Smooth / Resample / Normalize) works
manip list chaining runs the [Smooth, Resample, ZScore] spec end-to-end (#274)
Smooth kernel='gaussian' / 'boxcar' / 'savgol' visibly differ; jumps reduced
# Pipelines & cross-module
hyp.Pipeline: build, fit, transform new data, inverse_transform round-trip (#227/#161)
plot(..., pipeline=fitted) reapplies a pipeline to new data with no refit
cross-module kwargs: hyp.cluster(x, reduce=..., manip=...) and hyp.reduce(x, align=...) (#138)
hyp.analyze / hyp.apply_model / hyp.describe each return the expected shape
# Predict & impute
hyp.predict forecasts future rows; return_model reuse works
hyp.impute fills missing values (PPCA / Kalman / sklearn imputers)
# Animation
animate='spin' rotates at constant speed; duration= controls length (#275)
animate='window' shows a sliding opaque window; focused= sets its length (#275)
animate='morph' morphs between datasets (Hungarian correspondence)
2-D animation works for all styles except spin (spin on 2-D errors) (#123)
animate= dict form {'style':...,'duration':...} matches the equivalent flat kwargs (#154)
# Data loaders
hyp.load('iris') / hyp.load('digits') return sklearn datasets with a target column (#273)
hyp.load('penguins') returns the seaborn dataset (#273)
hyp.load('fivethirtyeight/bechdel') returns the 538 dataset (#116)
hyp.load('kaggle/uciml/iris') downloads anonymously (needs kagglehub) (#116)
# Text
hyp.plot(list_of_strings, vectorizer='Word2Vec') embeds + plots documents via gensim (#198)
sklearn->gensim->HF parse order: a name in sklearn resolves to sklearn (#198)
# Streaming
hyp.io.lsl_stream feeds hyp.plot from a live LSL outlet (needs a running outlet) (#130)
# Earlier 1.0 features
colorbar=True renders continuous + discrete colorbars in legend order (#100)
surface=True draws smooth lit convex-hull surfaces (2-D + 3-D) (#109)
density=True overlays subtle KDE shading, off by default (#108/#191)
MultiIndex DataFrames plot leaf trajectories + per-level group means (#95)
multibyte / CJK labels render without tofu boxes; font= kwarg works (#205)
# Save / backend
hyp.save round-trips a figure/data object
set_interactive_backend switches the matplotlib interactive backend
# Documentation (from the audit — spot-check the fixes once applied)
CLAUDE.md architecture section matches the current module layout (task A1)
API docstrings: align default, ndims, zoom, return_model bundle correct (task A2)
README mat2colors path + What's-new coverage (task A4)
