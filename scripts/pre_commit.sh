#!/bin/bash
set -e

# fix isort and format with black
isort src/**/*.py tests/**/*.py
black src tests

black --check --verbose src tests
python setup.py flake8
bandit -r src -ll
python setup.py test
