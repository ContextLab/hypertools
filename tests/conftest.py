import os
import pytest

@pytest.fixture
def fig_dir():
    return os.path.join(os.path.dirname(__file__), 'reference_figures')