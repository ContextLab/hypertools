# HyperTools v0.8.1 Release Notes (July, 2025)

This is a minor release that updates HyperTools for use with [NumPy 2.0+](https://numpy.org/devdocs/release/2.0.0-notes.html) and adds CI/CD infrastructure.

## Changes

### NumPy 2.0+ and Pandas 2.0+ Support
- Enhanced NumPy 2.0+ compatibility with additional deprecation fixes
- Fixed `pkg_resources` deprecation by replacing with `importlib.metadata`
- Fixed `scipy.stats.stats` deprecated import to use `scipy.stats`
- Updated dependency requirements for compatibility with NumPy 2.0+ and Pandas 2.0+

### CI/CD Infrastructure
- Added comprehensive GitHub Actions workflow for automated testing
- Multi-platform testing across Ubuntu, Windows, and macOS
- Multi-Python version testing (Python 3.9, 3.10, 3.11, 3.12)
- Coverage reporting with Codecov integration

### Bug Fixes
- Fixed t-SNE test error by setting appropriate perplexity parameter for small datasets

## Platform Support

### Python Versions
- Removed support for Python ≤ 3.8
- Added support for Python 3.10, 3.11, and 3.12

### Dependencies
- NumPy ≥ 2.0.0
- Pandas ≥ 2.2.0
- Updated all major dependencies to latest compatible versions

## Additional Notes
- This release brings back support for running HyperTools in Colaboratory notebooks, which now use NumPy 2.0+
- All 129 tests pass across all supported platforms and Python versions
- Comprehensive CI/CD ensures quality and prevents regressions

## Credits
- @terrafying - Original NumPy 2.0+ compatibility implementation
- @jeremymanning - Release management
- Claude Code (claude.ai/code) - Enhanced compatibility, CI/CD, and testing infrastructure