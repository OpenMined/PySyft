#!/bin/bash
set -e

./scripts/build_proto.sh

# fix isort and format with black
# isort src/**/*.py tests/**/*.py
black src tests
pre-commit run --all-files
bandit -r src -ll
python setup.py test

# run API documentation test notebooks
jupyter nbconvert --to script examples/api/*.ipynb
pytest examples/api