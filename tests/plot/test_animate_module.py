def test_animate_module_exposes_save_helpers():
    from hypertools.plot.animate import save_animation
    assert callable(save_animation)


def test_animate_reexports_svg_combiner():
    from hypertools.plot.animate import combine_frames_svg
    assert callable(combine_frames_svg)
