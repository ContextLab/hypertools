def test_io_load_importable():
    from hypertools.io.load import load
    assert callable(load)


def test_io_sources_importable():
    from hypertools.io.sources import is_loadable_string
    assert callable(is_loadable_string)


def test_io_streaming_data_side_importable():
    from hypertools.io.streaming import is_stream, row_to_vector
    assert callable(is_stream) and callable(row_to_vector)


def test_tools_shims_are_same_objects():
    from hypertools.io.load import load as new_load
    from hypertools.tools.load import load as old_load
    from hypertools.io.streaming import is_stream as new_is_stream
    from hypertools.tools.streaming import is_stream as old_is_stream
    assert new_load is old_load and new_is_stream is old_is_stream
