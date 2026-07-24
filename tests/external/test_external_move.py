def test_external_ppca_importable():
    from hypertools.external.ppca import PPCA
    assert hasattr(PPCA, "fit")


def test_external_brainiak_importable():
    from hypertools.external.brainiak import SRM, DetSRM
    assert hasattr(SRM, "fit") and hasattr(DetSRM, "fit")


def test_externals_shims_are_same_objects():
    from hypertools.external.ppca import PPCA as new_ppca
    from hypertools._externals.ppca import PPCA as old_ppca
    from hypertools.external.brainiak import SRM as new_srm
    from hypertools._externals.srm import SRM as old_srm
    assert new_ppca is old_ppca
    assert new_srm is old_srm
