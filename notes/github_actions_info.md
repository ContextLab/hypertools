# GitHub Actions CI/CD Setup

## Test Matrix
The GitHub Actions workflow (`/.github/workflows/test.yml`) runs comprehensive tests on:

### Python Versions
- Python 3.9
- Python 3.10  
- Python 3.11
- Python 3.12

### Operating Systems
- Ubuntu Latest (Linux)
- Windows Latest
- macOS Latest

### Features
- **Dependency caching**: Pip cache is used to speed up builds
- **System dependencies**: FFmpeg is installed for animation support
- **Coverage reporting**: Coverage is collected on Ubuntu Python 3.12 and uploaded to Codecov
- **Matrix strategy**: Tests run in parallel across all combinations (12 total jobs)
- **Fail-fast disabled**: All combinations run even if one fails

## Triggers
- Push to `master` or `dev` branches
- Pull requests to `master` or `dev` branches

## Badge
Add this badge to README.md to show build status:
```markdown
[![Tests](https://github.com/ContextLab/hypertools/workflows/Tests/badge.svg)](https://github.com/ContextLab/hypertools/actions)
```

## Local Testing
To run the same tests locally:
```bash
pytest -v --tb=short
```

For coverage:
```bash
pytest --cov=hypertools --cov-report=xml --cov-report=term-missing
```