"""
==============================================
Fit once, reuse: pipelines and return_model
==============================================

Every stage dispatcher (`hyp.manip`, `hyp.normalize`, `hyp.reduce`,
`hyp.align`, `hyp.cluster`, `hyp.analyze`, `hyp.impute`, `hyp.predict`) and
`hyp.plot` accept ``return_model=True`` to get back the *fitted* model
alongside the transformed result: a single fitted wrapper
when only one stage ran, or a fitted `hyp.Pipeline` when multiple stages
ran together (e.g. via the cross-module ``normalize=``/``reduce=``/
``align=``/``cluster=`` kwargs). The fitted model/`Pipeline` can
then be applied to held-out data via ``.transform()`` -- WITHOUT refitting
-- so a train/test split, or streaming new data through an established
projection, only ever fits once.

`hyp.plot`/`hyp.analyze` additionally accept the fitted `Pipeline` back in
via ``pipeline=``, so the exact multi-stage pipeline fit on one dataset can
be replayed on new data with a single call.
"""

# Code source: Contextual Dynamics Laboratory
# License: MIT

import numpy as np
import hypertools as hyp

# NOTE: train and test have the SAME number of rows on purpose -- the
# multi-stage demo below includes an alignment stage, and aligners (e.g.
# HyperAlign) operate on the rows common to all datasets, trimming any
# extras. Equal-length datasets mean nothing is discarded.
rng = np.random.default_rng(42)
train = np.cumsum(rng.standard_normal((200, 12)), axis=0)
test = np.cumsum(rng.standard_normal((200, 12)), axis=0)

# %% single-stage return_model: hyp.reduce returns the fitted Reducer
reduced, model = hyp.reduce(train, reduce='PCA', ndims=3, return_model=True)
print(f'single-stage model: {type(model).__name__}')

# apply the SAME fitted PCA to held-out data -- no refitting
test_reduced = model.transform(test)
print(f'train: {reduced.shape}, test (via .transform): {test_reduced.shape}')

# %% multi-stage return_model: normalize= + align= alongside reduce= makes
# hyp.reduce run all three stages (canonical order) and hands back
# a fitted hyp.Pipeline. (Alignment keeps only the rows common to all
# datasets -- pass equal-length datasets to avoid dropping data.)
multi_out, pipeline = hyp.reduce(
    [train, test], reduce='PCA', ndims=3,
    normalize='across', align='HyperAlign', return_model=True)
print(f'multi-stage model: {pipeline!r}')
print(f'multi-stage output shapes: {[m.shape for m in multi_out]}')

# %% hyp.plot's own return_model bundle exposes a 'pipeline' key that can be
# fed straight back into pipeline= to replay the exact fitted pipeline on
# new data -- no need to re-specify normalize=/reduce=/align=
bundle = hyp.plot([train], normalize='across', reduce='PCA', ndims=3,
                   align='HyperAlign', show=False, return_model=True)
replayed = hyp.plot([test], pipeline=bundle['pipeline'], show=False)
print('replayed a fitted 3-stage pipeline on held-out data via pipeline=')
