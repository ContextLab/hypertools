import os
import pytest

@pytest.fixture
def fig_dir():
    return os.path.join(os.path.split(os.path.dirname(__file__))[0], 'reference_figures')