# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

HyperTools is a Python library for visualizing and manipulating high-dimensional data. It provides a unified interface for dimensionality reduction, data alignment, clustering, and visualization, built on top of matplotlib, plotly (interactive backend), scikit-learn, seaborn, and pydata-wrangler (core data-wrangling).

## Key Commands

### Testing
- `pytest` - Run all tests (run from the repo root; `pyproject.toml` sets `testpaths = ["tests"]`)
- `pytest tests/test_<module>.py` - Run tests for a specific module
- `pytest tests/test_<module>.py::test_<function>` - Run a specific test function

### Development Setup
- `pip install -e ".[dev]"` - Install in development mode with test/dev dependencies. `[dev]` does NOT pull in every optional extra: `text` (pydata-wrangler[hf], HF text embeddings) and `predict-hf` (chronos-forecasting) are left out, and their code paths are `importorskip`-guarded in tests.
- `pip install -r docs/doc_requirements.txt` - Install documentation dependencies

### Documentation
- `cd docs && make html` - Build HTML documentation
- `cd docs && make clean` - Clean documentation build files

## Code Architecture

### Core Components

**DataGeometry Class** (`hypertools/datageometry.py`)
- INTERNAL, unpickle-only legacy shell (its own docstring: "not part of the public API"); not used by any current API function
- Kept solely so `hypertools.load()` can unpickle hosted example-dataset geo files (created by hypertools < 1.0) and extract their raw data via `get_data()`
- In 1.0, `plot()` returns a matplotlib `Figure` and `load()` returns raw data -- no `DataGeometry` is ever constructed or returned to users

**Main API Functions** (`hypertools/__init__.py`)
- `plot()` - Primary visualization function
- `analyze()` - Classic manip -> normalize -> reduce -> align -> cluster pipeline dispatcher
- `reduce()` - Dimensionality reduction utilities
- `align()` - Data alignment across datasets
- `normalize()` - Data normalization
- `describe()` - Data description and summary
- `cluster()` - Clustering functionality
- `manip()` - Manipulator dispatcher (Normalize/ZScore/Smooth/Resample, etc.)
- `predict()` - Timeseries forecasting (ARIMA, Kalman, GP, autoregression, Chronos, etc.)
- `impute()` - Missing-data imputation (PPCA, Kalman, sklearn imputers)
- `load()` - Data loading utilities
- `save()` - Save data/results to disk
- `apply_model()` - Apply an arbitrary sklearn-API model to data
- `Pipeline` - Class for chaining/reusing manip/reduce/align/cluster stages
- `set_interactive_backend()` - Switch the plotting backend (matplotlib/plotly)
- `io` submodule - I/O helpers, including `io.lsl_stream()` for Lab Streaming Layer input

**Tools Module** (`hypertools/tools/`)
- `align.py` - Classic array/mode `align()` API; thin compatibility wrapper over `hypertools/align/`
- `analyze.py` - `analyze()` pipeline dispatcher
- `normalize.py` - `normalize()` (z-scoring) with cross-module stage kwargs
- `format_data.py` - Data preprocessing and formatting
- `text2mat.py` - Text-to-matrix conversion
- `df2mat.py` - DataFrame-to-matrix conversion
- `gensim_models.py` - sklearn-API wrappers around gensim topic/embedding models
- `missing_inds.py` - Missing data handling

The dev-1.0 refactor moved several tools into their own top-level subpackages (each a folder with a main module plus helpers):
- `hypertools/reduce/` - Dimensionality reduction (`reduce.py`, `describe.py`, `autoencoders.py`, `common.py`)
- `hypertools/cluster/` - Clustering (`cluster.py`, `common.py`)
- `hypertools/align/` - Alignment (`align.py`, `hyperalign.py`, `procrustes.py`, `srm.py`, `null.py`, `common.py`)
- `hypertools/manip/` - Manipulators (`manip.py`, `normalize.py`, `zscore.py`, `smooth.py`, `resample.py`, `common.py`)
- `hypertools/io/` - Loading/saving/streaming (`load.py`, `save.py`, `sources.py`, `streaming.py`, `lsl.py`)
- `hypertools/predict/` - Forecasting models (`predict.py`, `arima.py`, `autoreg.py`, `gp.py`, `kalman.py`, `laplace.py`, `chronos.py`, `common.py`)
- `hypertools/impute/` - Imputation models (`impute.py`, `ppca.py`, `kalman.py`, `sklearn_imputers.py`, `common.py`)
- `hypertools/core/` - Shared config/exceptions and `apply_model()`/`Pipeline` (`configurator.py`, `exceptions.py`, `model.py`, `pipeline.py`, `shared.py`)

**Plot Module** (`hypertools/plot/`)
- `plot.py` - Main plotting interface and logic
- `backend.py` - Backend selection (`set_interactive_backend()`)
- `matplotlib_backend.py` - Low-level matplotlib drawing (`draw.py` is now a 3-line compatibility shim over this)
- `plotly_backend.py` - Low-level plotly drawing (interactive backend)
- `interactive.py` - Interactive-mode helpers
- `animate.py` - Animation support
- `colors.py` - Color handling
- `density.py` / `surface.py` - Density and iso-surface rendering
- `morph.py` / `trails.py` - Morph and trail effects
- `meshutil.py` - Mesh utilities
- `multiindex.py` - Pandas MultiIndex handling
- `fonts.py` - Font handling

**External Dependencies** (`hypertools/_externals/`)
- `ppca.py` - Probabilistic Principal Component Analysis
- `srm.py` - Shared Response Model

### Data Flow

1. **Input Processing**: Data is formatted and validated through `format_data()`
2. **Normalization**: Optional data normalization via `normalize()`
3. **Alignment**: Optional cross-dataset alignment via `align()`
4. **Dimensionality Reduction**: Data is reduced via `reduce()`
5. **Clustering**: Optional clustering via `cluster()`
6. **Visualization**: Final plotting through `plot()`

### Key Design Patterns

- **Modular Architecture**: Each major operation (align, reduce, normalize, etc.) is in its own module
- **Unified Interface**: All functions accept similar input formats (lists of arrays, DataFrames, etc.)
- **Flexible Data Types**: Supports numpy arrays, pandas DataFrames, text data, and mixed inputs
- **Matplotlib Integration**: Deep integration with matplotlib for customizable visualizations
- **Animation Support**: Built-in support for animated visualizations

## Development Notes

- The package follows a functional programming style with separate modules for each operation
- All major functions are designed to work with multiple input formats
- `plot()` returns a matplotlib `Figure`; `load()` returns raw data -- there is no central state-container object in 1.0 (`DataGeometry` is an internal unpickle-only legacy shell; see above)
- Tests are located in `tests/` directory and follow pytest conventions
- Documentation is built with Sphinx and uses example galleries
- The codebase requires Python 3.10+ (`requires-python = ">=3.10"`)

## Testing Strategy

- Unit tests for individual tools and functions
- Integration tests for end-to-end workflows
- Example-based testing through documentation
- Visual regression testing for plot outputs