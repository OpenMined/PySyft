#!/bin/bash
set -e

# fix isort and format with black
# isort src/**/*.py tests/**/*.py
black src tests
pre-commit run --all-files
bandit -r src -ll
python setup.py test
