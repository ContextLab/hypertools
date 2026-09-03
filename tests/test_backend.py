# -*- coding: utf-8 -*-
"""Tests for hypertools.plot.backend module, particularly version parsing."""



class TestVersionParsing:
    """Test the notebook version parsing logic used in _get_jupyter_frontend()."""

    def _parse_version(self, version_output):
        """
        Replicate the version parsing logic from backend.py.
        Returns tuple of ints or None if parsing fails.
        """
        try:
            version_line = version_output.split("\n")[0].strip()
            version_parts = version_line.split(".")
            return tuple(
                int(
                    p.split("-")[0]
                    .split("+")[0]
                    .split("a")[0]
                    .split("b")[0]
                    .split("rc")[0]
                )
                for p in version_parts[:3]
            )
        except (ValueError, IndexError):
            return None

    def test_standard_version(self):
        assert self._parse_version("7.0.0") == (7, 0, 0)
        assert self._parse_version("6.5.4") == (6, 5, 4)
        assert self._parse_version("10.2.1") == (10, 2, 1)

    def test_prerelease_versions(self):
        assert self._parse_version("7.0.0-rc1") == (7, 0, 0)
        assert self._parse_version("7.0.0a1") == (7, 0, 0)
        assert self._parse_version("7.0.0b2") == (7, 0, 0)
        assert self._parse_version("7.2.3+build123") == (7, 2, 3)

    def test_colab_path_output(self):
        """
        Test handling of Colab's malformed version output.
        In Colab, 'jupyter notebook --version' can return a path like
        '/usr/local/lib/python3' instead of a version number.
        """
        assert self._parse_version("/usr/local/lib/python3") is None
        assert self._parse_version("/usr/local/lib/python3.12/dist-packages") is None

    def test_garbage_input(self):
        assert self._parse_version("some garbage") is None
        assert self._parse_version("") is None
        assert self._parse_version("not a version") is None

    def test_multiline_output(self):
        assert self._parse_version("\n7.0.0") is None
        assert self._parse_version("7.0.0\nsome extra line") == (7, 0, 0)


class TestBackendInitialization:
    """Test that backend initialization doesn't crash on import."""

    def test_import_hypertools(self):
        """Verify hypertools can be imported without error."""
        import hypertools

        assert hypertools is not None

    def test_backend_module_globals(self):
        """Verify backend module globals are initialized."""
        from hypertools.plot import backend

        assert backend.BACKEND_MAPPING is not None
        assert backend.HYPERTOOLS_BACKEND is not None
        assert backend.IS_NOTEBOOK is not None
