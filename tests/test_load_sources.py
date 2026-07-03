# -*- coding: utf-8 -*-
"""Universal loader: hyp.load / DataGeometry resolve strings through
builtin -> local file -> Hugging Face -> Google Drive -> Dropbox -> URL,
and lists of strings resolve to lists of datasets. All tests use real
files and real network calls (no mocks)."""

import numpy as np
import pandas as pd
import pytest

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import hypertools as hyp
from hypertools._shared.exceptions import HypertoolsIOError
from hypertools.tools.sources import is_loadable_string

IRIS_CSV = 'raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv'
# legacy hypertools 'spiral' pickle, still hosted on Google Drive
DRIVE_SPIRAL_ID = '1nHAusn2VsQinJk35xvJSd7CtWPC1uOwK'
DROPBOX_BUNNY = 'https://www.dropbox.com/s/7d9vo9idqk1hn31/bunny.pkl?dl=0'


def test_is_loadable_string_discrimination():
    assert is_loadable_string('spiral')                      # builtin
    assert is_loadable_string('https://example.com/x.csv')   # URL
    assert is_loadable_string(IRIS_CSV)                      # schemeless URL
    assert is_loadable_string('scikit-learn/iris')           # HF id
    assert is_loadable_string(DRIVE_SPIRAL_ID)               # bare Drive id
    assert is_loadable_string(DROPBOX_BUNNY)
    # raw text must NOT be treated as a data source
    assert not is_loadable_string('the dog is happy')
    assert not is_loadable_string('')
    assert not is_loadable_string('nonexistent_dataset_name')


def test_load_local_formats(tmp_path):
    arr = np.random.default_rng(0).standard_normal((12, 4))
    np.save(tmp_path / 'a.npy', arr)
    np.savez(tmp_path / 'b.npz', x=arr, y=arr * 2)
    pd.DataFrame(arr).to_csv(tmp_path / 'c.csv', index=False)
    pd.DataFrame(arr).to_csv(tmp_path / 'd.tsv', sep='\t', index=False)
    pd.DataFrame(arr).to_json(tmp_path / 'e.json')
    pd.to_pickle(pd.DataFrame(arr), tmp_path / 'f.pkl')

    np.testing.assert_allclose(hyp.load(str(tmp_path / 'a.npy')), arr)
    npz = hyp.load(str(tmp_path / 'b.npz'))
    assert isinstance(npz, list) and len(npz) == 2
    assert hyp.load(str(tmp_path / 'c.csv')).shape == (12, 4)
    assert hyp.load(str(tmp_path / 'd.tsv')).shape == (12, 4)
    assert hyp.load(str(tmp_path / 'e.json')).shape == (12, 4)
    assert hyp.load(str(tmp_path / 'f.pkl')).shape == (12, 4)


def test_load_huggingface_dataset():
    pytest.importorskip('datasets')
    df = hyp.load('scikit-learn/iris')
    assert isinstance(df, pd.DataFrame)
    assert df.shape[0] == 150


def test_load_huggingface_streaming_flows_to_plot():
    pytest.importorskip('datasets')
    ds = hyp.load('scikit-learn/iris', streaming=True)
    from hypertools.tools.streaming import is_stream
    assert is_stream(ds)
    ds = ds.select_columns(['SepalLengthCm', 'SepalWidthCm',
                            'PetalLengthCm', 'PetalWidthCm'])
    geo = hyp.plot(ds, '.', show=False, stream_init=50, stream_chunk=50)
    assert geo.stream_info['n_samples'] == 150
    plt.close('all')


def test_load_streaming_rejects_load_time_transforms():
    pytest.importorskip('datasets')
    with pytest.raises(ValueError, match='stream'):
        hyp.load('scikit-learn/iris', streaming=True, ndims=3)


def test_load_google_drive_id_and_url():
    geo = hyp.load(DRIVE_SPIRAL_ID)
    assert type(geo).__name__ == 'DataGeometry'
    url = ('https://drive.google.com/uc?export=download&id='
           + DRIVE_SPIRAL_ID)
    geo2 = hyp.load(url)
    assert type(geo2).__name__ == 'DataGeometry'


def test_load_dropbox_url_forms():
    a = hyp.load(DROPBOX_BUNNY)                       # dl=0 -> normalized
    b = hyp.load('www.dropbox.com/s/7d9vo9idqk1hn31/bunny.pkl')
    c = hyp.load('s/7d9vo9idqk1hn31/bunny.pkl')       # shared-link path
    for o in (a, b, c):
        assert np.asarray(o).shape[1] == 3            # x, y, z point cloud


def test_load_generic_url_with_and_without_scheme():
    df1 = hyp.load('https://' + IRIS_CSV)
    df2 = hyp.load(IRIS_CSV)
    assert df1.shape == df2.shape == (150, 5)


def test_load_list_of_strings(tmp_path):
    arr = np.random.default_rng(1).standard_normal((9, 3))
    np.save(tmp_path / 'x.npy', arr)
    out = hyp.load(['spiral', str(tmp_path / 'x.npy')])
    assert isinstance(out, list) and len(out) == 2
    assert type(out[0]).__name__ == 'DataGeometry'
    np.testing.assert_allclose(out[1], arr)


def test_datageometry_plot_accepts_source_strings():
    geo = hyp.load('spiral')
    g2 = geo.plot(data=IRIS_CSV, show=False)
    assert type(g2).__name__ == 'DataGeometry'
    plt.close('all')


def test_plot_mixed_dtype_dataframe():
    """Regression: pandas>=2.0 get_dummies returns bool, which made
    df2mat produce object arrays that crashed np.isnan in format_data."""
    df = pd.DataFrame({'a': np.random.rand(20), 'b': np.random.rand(20),
                       'c': ['x', 'y'] * 10})
    from hypertools.tools.df2mat import df2mat
    m = df2mat(df)
    assert m.dtype == np.float64
    geo = hyp.plot(df, '.', show=False)
    plt.close('all')


def test_load_unresolvable_string_lists_attempts():
    with pytest.raises(HypertoolsIOError, match='Tried, in order'):
        hyp.load('no_such_dataset_or_file_xyz123')
