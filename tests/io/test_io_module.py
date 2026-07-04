def test_io_load_importable():
    from hypertools.io.load import load
    assert callable(load)


def test_io_sources_importable():
    from hypertools.io.sources import is_loadable_string
    assert callable(is_loadable_string)


def test_io_streaming_data_side_importable():
    from hypertools.io.streaming import is_stream, row_to_vector
    assert callable(is_stream) and callable(row_to_vector)
